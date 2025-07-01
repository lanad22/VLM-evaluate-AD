import os
import json
import argparse
import pathlib
import time
import re
import ast
import subprocess

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

PROMPT_FOR_EVALUATION = """
ROLE: You are an expert Accessibility Consultant specializing in the quality assurance of audio description (AD) for video content.

CONTEXT: I am providing you with two assets:
1.  A video file.
2.  The structured JSON data of the existing audio description, which is included below.

**JSON DATA:**
```json
{json_data}
```

TASK: Analyze the video and the JSON data to evaluate the quality of the audio description track.

SCALE (1–5):
1 = very poor, 2 = poor, 3 = acceptable, 4 = good, 5 = exemplary.

CATEGORIES & CRITERIA:
Reads Text-on-Screen: Captures visible text accurately and at the right time. (If there is no on-screen text in the video, score = 5 with justification “no on-screen text present.”)
Inline Track Quality: Effectiveness of short ADs placed during natural pauses. (Inline ADs are preferred over extended ones when they can convey the same info.)
Extended Track Quality: Effectiveness of longer ADs inserted into pauses or gaps.
Strategic AD Type Selection: Optimal mix of brief (preferred) and in-depth AD.
Track Placement: Narration is well-timed and does not overlap original video dialog or music.


OUTPUT FORMAT:
You MUST return your response as a single, valid JSON object. Do not include any text, notes, or markdown formatting before or after the JSON block.
The JSON object should have the following structure:
{{
  "evaluation_summary": {{
    "overall_quality_rating": "A rating from 1 to 5, where 1 is poor and 5 is excellent.",
    "strengths": "A brief summary of what was done well.",
    "areas_for_improvement": "A brief summary of what could be improved."
  }},
  "criteria_ratings": {{
    "Reads Text-on-Screen": {{ "rating": "1-5", "justification": "..." }},
    "Inline Track Quality": {{ "rating": "1-5", "justification": "..."}},
    "Extended Track Quality": {{ "rating": "1-5", "justification": "..." }},
    "strategic_AD_type_selection": {{ "rating": "1-5", "justification": "..."}},
    "track_placement": {{ "rating": "1-5", "justification": "..." }}
  }}
}}
"""

def standardize_video_for_processing(input_path: str) -> str:
    input_dir = os.path.dirname(input_path)
    base_name, ext = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(input_dir, f"{base_name}_temp{ext}")
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", input_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed to convert {input_path}: {e.stderr.decode()}. Using original path.")
        return input_path

def create_video_chunk(video_path: str, start_time: float, duration: float, output_path: str) -> bool:
    """Create a video chunk using ffmpeg."""
    command = [
        "ffmpeg", "-y", "-loglevel", "error", 
        "-i", video_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
        output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create chunk {output_path}: {e.stderr.decode()}")
        return False

def get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe."""
    command = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        print(f"Could not determine duration for {video_path}")
        return 0.0

def clean_and_parse_json(text: str) -> dict:
    print("--- CLEANING AND PARSING RESPONSE ---")
    if not text:
        return {"error": "Received empty response from model."}
    try:
        text = re.sub(r'```json|```', '', text).strip()
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_text = text[start_idx:end_idx + 1]
            print(f"Extracted JSON snippet: {json_text[:200]}...")
            return json.loads(json_text)
        else: 
            return ast.literal_eval(text)
    except (json.JSONDecodeError, SyntaxError, ValueError) as e:
        print(f"Failed to parse JSON with standard methods: {e}")
        return {"error": "Failed to parse model output as JSON", "raw_response": text}

def process_single_chunk(messages: list, model_client: dict, chunk_index: int) -> str:
    """Process a single video chunk and return raw response text."""
    model = model_client['model']
    processor = model_client['processor']
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Clear GPU cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
            
            inputs = processor(
                text=[text], 
                images=image_inputs, 
                videos=video_inputs, 
                padding=True,
                return_tensors="pt"
            ).to(model.device)
            
            generation_config = {
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_config)
            
            input_token_len = inputs.input_ids.shape[1]
            generated_ids = output_ids[:, input_token_len:]
            response_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            
            print(f"Got response for chunk {chunk_index}")
            return response_text

        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM error for chunk {chunk_index} (attempt {attempt + 1}): {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if attempt == max_retries - 1:
                return None
            time.sleep(10)
            
        except Exception as e:
            print(f"Error processing chunk {chunk_index} (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(5)
    
    return None

def combine_chunk_responses(responses: list) -> dict:
    """Combine multiple chunk responses into a single comprehensive evaluation."""
    if not responses:
        return {"error": "No valid responses from chunks"}
    
    # Try to parse the first valid response as the primary evaluation
    for response in responses:
        parsed = clean_and_parse_json(response)
        if parsed and "evaluation_summary" in parsed:
            return parsed
    
    # If no valid parse, return error
    return {"error": "Could not parse any chunk responses", "raw_responses": responses}

def evaluate_video_with_qwen(video_path: str, json_data_str: str, model_client: dict) -> dict:
    """Evaluate the entire video by processing it in chunks but combining context."""
    model = model_client['model']
    processor = model_client['processor']
    
    final_prompt = PROMPT_FOR_EVALUATION.format(json_data=json_data_str)
    
    # Get video duration
    video_duration = get_video_duration(video_path)
    chunk_duration = 30.0  # Process in 30-second chunks
    
    all_responses = []
    chunk_start = 0.0
    chunk_index = 0
    temp_files = []
    
    try:
        while chunk_start < video_duration:
            chunk_end = min(chunk_start + chunk_duration, video_duration)
            actual_duration = chunk_end - chunk_start
            
            # Create chunk file
            chunk_filename = f"temp_chunk_{chunk_index}.mp4"
            chunk_path = os.path.join(os.path.dirname(video_path), chunk_filename)
            temp_files.append(chunk_path)
            
            print(f"Processing chunk {chunk_index}: {chunk_start:.1f}s - {chunk_end:.1f}s")
            
            if create_video_chunk(video_path, chunk_start, actual_duration, chunk_path):
                # Process this chunk with the FULL JSON context
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": final_prompt}, 
                    {"type": "video", "video": chunk_path}
                ]}]
                
                chunk_response = process_single_chunk(messages, model_client, chunk_index)
                if chunk_response:
                    all_responses.append(chunk_response)
            
            chunk_start = chunk_end
            chunk_index += 1
    
    finally:
        # Cleanup chunk files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
    
    # Combine all chunk responses into a single evaluation
    return combine_chunk_responses(all_responses)

def main():
    parser = argparse.ArgumentParser(description="Evaluate audio description using Qwen with video chunking.")
    parser.add_argument("video_folder", help="Path to the folder containing the video and JSON data.")
    parser.add_argument("--input_type", required=True, help="The source of the input JSON file.")
    args = parser.parse_args()
    
    folder_path = pathlib.Path(args.video_folder)
    video_id = os.path.basename(os.path.normpath(args.video_folder))
    video_path = folder_path / f"{video_id}.mp4"
    json_path = folder_path / f"final_data_{args.input_type}.json"

    if not video_path.is_file() or not json_path.is_file():
        print(f"Error: Missing video '{video_path}' or JSON '{json_path}'.")
        return

    print(f"Found video: {video_path}")
    print(f"Found input JSON: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        json_string_for_prompt = json.dumps(json.load(f), indent=2)

    # Initialize Qwen model
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model_path = "Qwen/Qwen2.5-VL-72B-Instruct"
    print(f"Initializing Qwen model: {model_path}")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", 
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir="../.cache",
            low_cpu_mem_usage=True,
            max_memory={0: "70GiB"}
        )
        processor = AutoProcessor.from_pretrained(model_path)
        model_client = {'model': model, 'processor': processor}
        
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
    except Exception as e:
        print(f"Failed to load Qwen model: {e}")
        return

    # Standardize video and evaluate
    standardized_video_path = None
    
    try:
        standardized_video_path = standardize_video_for_processing(str(video_path))
        
        # Evaluate the entire video using chunked processing
        evaluation_result = evaluate_video_with_qwen(
            standardized_video_path, json_string_for_prompt, model_client
        )
    
    finally:
        # Cleanup standardized video
        if standardized_video_path and standardized_video_path != str(video_path) and os.path.exists(standardized_video_path):
            try:
                os.remove(standardized_video_path)
                print(f"Cleaned up: {standardized_video_path}")
            except OSError as e:
                print(f"Warning: Could not remove {standardized_video_path}: {e}")

    # Save results
    if evaluation_result:
        output_filename = f"qwen_evaluate_{args.input_type}.json"
        output_path = folder_path / output_filename
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(evaluation_result, f, indent=4, ensure_ascii=False)
            print(f"\nEvaluation successfully saved to: {output_path}")
        except IOError as e:
            print(f"\nError saving file: {e}")
    else:
        print("No evaluation result to save.")

if __name__ == "__main__":
    main()