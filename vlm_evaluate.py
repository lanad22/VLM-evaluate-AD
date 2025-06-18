import os
import json
import argparse
import pathlib
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

PROMPT_FOR_EVALUATION = """
**ROLE:** You are an expert Accessibility Consultant specializing in the quality assurance of audio description (AD) for video content.

**CONTEXT:** I am providing you with two assets:
1.  A video file.
2.  The structured JSON data for the video, which is included below.

**JSON DATA:**
```json
{json_data}
```

**TASK:** Analyze the video and the JSON data to evaluate the quality of the audio description track.

**OUTPUT FORMAT:**
You MUST return your response as a single, valid JSON object. Do not include any text, notes, or markdown formatting before or after the JSON block.

The JSON object should have the following structure:
{{
  "evaluation_summary": {{
    "overall_quality_rating": "A rating from 1 to 5",
    "strengths": "A brief summary of what was done well.",
    "areas_for_improvement": "A brief summary of what could be improved."
  }},
  "criteria_ratings": {{
    "reads_text_on_screen": {{
      "rating": "A rating from 1 to 5",
      "justification": "Detailed analysis of how on-screen text was handled, with examples."
    }},
    "inline_track_quality": {{
      "rating": "A rating from 1 to 5",
      "justification": "Detailed analysis of the inline descriptions, with examples."
    }},
    "extended_track_quality": {{
      "rating": "A rating from 1 to 5",
      "justification": "Detailed analysis of the extended descriptions, with examples."
    }},
    "balance_of_inline_and_extended": {{
      "rating": "A rating from 1 to 5",
      "justification": "Analysis of the overall balance between short and long descriptions."
    }},
    "track_placement": {{
      "rating": "A rating from 1 to 5",
      "justification": "Critical analysis of whether AD overlaps with dialogue or key sounds, with specific timestamps if possible."
    }}
  }}
}}
"""

def wait_for_file_to_be_active(file) -> bool:
    print(f"Waiting for file '{file.display_name}' to be processed...")
    while file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(10)
        file = genai.get_file(name=file.name)

    if file.state.name == "FAILED":
        print(f"\nError: File processing failed for '{file.display_name}'.")
        return False

    print(f"\nFile '{file.display_name}' is now ACTIVE.")
    return True

def evaluate_audio_description(video_folder_path: str):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found.")
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error configuring API: {e}")
        return None

    try:
        video_id = os.path.basename(os.path.normpath(video_folder_path))
        video_path = pathlib.Path(video_folder_path) / f"{video_id}.mp4"
        json_path = pathlib.Path(video_folder_path) / "Melissa_adzYW5DZoWs_output.json"

        if not video_path.is_file() or not json_path.is_file():
            print(f"Error: Missing video or JSON file in '{video_folder_path}'.")
            return None
    except Exception as e:
        print(f"Error deriving file paths: {e}")
        return None

    print(f"Found video: {video_path}")
    print(f"Found JSON: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_content_as_dict = json.load(f)
        json_string_for_prompt = json.dumps(json_content_as_dict, indent=2)
        print("Successfully read JSON data.")
    except Exception as e:
        print(f"Error reading or parsing JSON file '{json_path}': {e}")
        return None

    print("\nUploading video file to the Gemini API...")
    try:
        video_file = genai.upload_file(path=video_path)
        if not wait_for_file_to_be_active(video_file):
            return None
    except Exception as e:
        print(f"An error occurred during file upload: {e}")
        return None

    final_prompt = PROMPT_FOR_EVALUATION.format(json_data=json_string_for_prompt)

    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    print("\nFile is ready. Generating the evaluation...")
    try:
        response = model.generate_content(
            [final_prompt, video_file],
            request_options={'timeout': 600}
        )
        
        print("\n--- RAW MODEL RESPONSE ---")
        print(response.text)
        print("--- END RAW RESPONSE ---\n")
        
        response_text = response.text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_text = response_text[start_idx:end_idx + 1]
            print(f"Extracted JSON: {json_text[:200]}...")
            return json.loads(json_text)
        else:
            return json.loads(response_text)
            
    except json.JSONDecodeError as e:
        print(f"\n--- JSON DECODE ERROR: {e} ---")
        print("Raw response text:")
        print(repr(response.text))
        print("\nSaving raw response for debugging...")
        
        debug_path = pathlib.Path(video_folder_path) / "debug_raw_response.txt"
        try:
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Raw response saved to: {debug_path}")
        except Exception as save_error:
            print(f"Could not save debug file: {save_error}")
            
        return {"error": "Failed to parse model output as JSON", "raw_response": response.text}
        
    except Exception as e:
        print(f"An error occurred during content generation: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate an audio description track using the Gemini 1.5 Pro model."
    )
    parser.add_argument(
        "video_folder",
        help="Path to the folder containing the video file and JSON data."
    )
    args = parser.parse_args()
    evaluation_result = evaluate_audio_description(args.video_folder)

    if evaluation_result:
        output_path = pathlib.Path(args.video_folder) / "gemini_evaluate_human.json"
        print(f"\nAttempting to save evaluation to: {output_path}")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(evaluation_result, f, indent=4, ensure_ascii=False, sort_keys=True)
            print(f"\nEvaluation successfully saved to: {output_path}")
        except IOError as e:
            print(f"\nError saving file: {e}")