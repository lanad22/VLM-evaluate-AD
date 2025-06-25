import os
import json
import argparse
import pathlib
import time
import re
import ast
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

load_dotenv()

'''
PROMPT_FOR_EVALUATION = """
**ROLE:** You are an expert Accessibility Consultant specializing in the quality assurance of audio description (AD) for video content.

**CONTEXT:** I am providing you with two assets:
1.  A video file.
2.  The structured JSON data of the existing audio description, which is included below.

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
    "overall_quality_rating": "A rating from 1 to 5, where 1 is poor and 5 is excellent.",
    "strengths": "A brief summary of what was done well.",
    "areas_for_improvement": "A brief summary of what could be improved."
  }},
  "criteria_ratings": {{
    "reads_text_on_screen": {{ "rating": "1-5", "justification": "Captures visible text accurately and at the right time." }},
    "inline_track_quality": {{ "rating": "1-5", "justification": "Effectiveness of short ADs placed during natural pauses." }},
    "extended_track_quality": {{ "rating": "1-5", "justification": "Effectiveness of longer ADs inserted into pauses or gaps." }},
    "balance_of_inline_and_extended": {{ "rating": "1-5", "justification": "Optimal mix of brief and in-depth descriptions." }},
    "track_placement": {{ "rating": "1-5", "justification": "Narration is well-timed and does not conflict with dialogue or music." }}
  }}
}}
"""
'''

PROMPT_FOR_EVALUATION = """
ROLE: 
You are an expert content evaluator specializing in audio descriptions.

INPUT:
A video file.
JSON data with original dialog and human-authored AD:
{json_data}.

SCALE (1–5):
1 = very poor, 2 = poor, 3 = acceptable, 4 = good, 5 = exemplary.

CATEGORIES & CRITERIA:
Reads Text-on-Screen: Captures visible text accurately and at the right time. (If there is no on-screen text in the video, score = 5 with justification “no on-screen text present.”)
Inline Track Quality: Effectiveness of short ADs placed during natural pauses. (Inline ADs are preferred over extended ones when they can convey the same info.)
Extended Track Quality: Effectiveness of longer ADs inserted into pauses or gaps.
Balance of Inline and Extended: Optimal mix of brief (preferred) and in-depth AD.
Track Placement: Narration is well-timed and does not overlap original video dialog or music.

EVALUATION SUMMARY: 
Overall Quality Rating: A rating from 1 to 5, where 1 is poor and 5 is excellent.,
Strengths: A brief summary of what was done well.,
Areas for improvement": A brief summary of what could be improved.
    
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

def evaluate_audio_description(video_folder_path: str, input_type: str):
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
        json_path = pathlib.Path(video_folder_path) / f"final_data_{input_type}.json"

        if not video_path.is_file() or not json_path.is_file():
            print(f"Error: Missing video '{video_path}' or JSON '{json_path}'.")
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

    model = genai.GenerativeModel(
        'models/gemini-1.5-pro-latest',
        system_instruction="You are an expert Accessibility Consultant specializing in the quality assurance of audio description (AD) for video content.",
        generation_config={
            "temperature": 0.6,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
        },
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    print("\nFile is ready. Generating the evaluation...")
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                [final_prompt, video_file],
                request_options={'timeout': 600}
            )
            
            print("\n--- RAW GEMINI RESPONSE ---")
            print(response.text)
            print("--- END RAW RESPONSE ---\n")
            
            return clean_and_parse_json(response.text)
            
        except Exception as e:
            print(f"Error during generation (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return {"error": f"Failed after {max_retries} attempts", "last_error": str(e)}
            time.sleep(5 * (attempt + 1))
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate an audio description track using the Gemini 1.5 Pro model."
    )
    parser.add_argument(
        "video_folder",
        help="Path to the folder containing the video file and JSON data."
    )
    parser.add_argument(
        "--input_type", 
        required=True,
        help="The source of the input JSON file (e.g., 'human', 'qwen', 'gemini')."
    )
    args = parser.parse_args()
    
    evaluation_result = evaluate_audio_description(args.video_folder, args.input_type)

    if evaluation_result:
        output_filename = f"gemini_evaluate_{args.input_type}.json"
        output_path = pathlib.Path(args.video_folder) / output_filename
        print(f"\nAttempting to save evaluation to: {output_path}")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(evaluation_result, f, indent=4, ensure_ascii=False)
            print(f"\nEvaluation successfully saved to: {output_path}")
        except IOError as e:
            print(f"\nError saving file: {e}")