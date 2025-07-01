import json
import argparse
from pathlib import Path

def prepare_dialogue(scenes):
    dialogue = []
    sequence_counter = 1
    last_dialogue_end = None
    continuing_dialogue = False
    
    for scene in scenes:
        scene_starttime = scene.get("start_time", 0)
        transcript = scene.get("transcript", [])
        
        for line in transcript:
            start = scene_starttime + line.get("start", 0)
            end = scene_starttime + line.get("end", 0)
            text = line.get("text", "")
            
            gap_threshold = 0.1
            
            if last_dialogue_end is not None and abs(start - last_dialogue_end) < gap_threshold:
                if dialogue and continuing_dialogue:
                    dialogue[-1]["end_time"] = end - 0.1
                    dialogue[-1]["duration"] = round(dialogue[-1]["end_time"] - dialogue[-1]["start_time"], 2)
                    continuing_dialogue = False 
                    last_dialogue_end = end 
                    continue 
            
            duration = round(end - start, 2)
            dialogue.append({
                "start_time": start,
                "end_time": end - 0.1,
                "duration": duration,
                "audio_text": text,
                "sequence_num": sequence_counter
            })
            sequence_counter += 1
            
            if end >= scene.get("end_time", 0) - gap_threshold:
                continuing_dialogue = True
            else:
                continuing_dialogue = False
                
            last_dialogue_end = end
            
    return dialogue

def generate_final_output(scenes_path, human_videoid_json_path, output_path):
    dialogue_timestamps = []
    audio_clips = []

    try:
        with open(scenes_path, mode='r', encoding='utf-8') as scenes_file:
            scenes_data = json.load(scenes_file)
        
        print(f"Successfully loaded scene data from {scenes_path}.")
        
        dialogue_timestamps = prepare_dialogue(scenes_data)
        print(f"Successfully prepared {len(dialogue_timestamps)} dialogue entries from scene data.")

    except FileNotFoundError:
        print(f"Error: The scenes file '{scenes_path}' was not found. Aborting.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{scenes_path}'. Please check file format.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while processing scene data: {e}")
        return

    try:
        with open(human_videoid_json_path, mode='r', encoding='utf-8') as human_json_file:
            human_data = json.load(human_json_file)
            if "audio_clips" in human_data:
                audio_clips = human_data["audio_clips"]
                audio_clips.sort(key=lambda clip: clip['start_time'])
                print(f"Successfully loaded and sorted {len(audio_clips)} audio clips from {human_videoid_json_path}.")
            else:
                print(f"Warning: 'audio_clips' key not found in {human_videoid_json_path}. Audio clips will be empty.")
    except FileNotFoundError:
        print(f"Error: The human_videoid JSON file '{human_videoid_json_path}' was not found. Aborting.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{human_videoid_json_path}'. Please check file format.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading audio clips from JSON: {e}")
        return

    # Combine and save the final data
    final_data = {
        "dialogue_timestamps": dialogue_timestamps,
        "audio_clips": audio_clips,
    }

    try:
        with open(output_path, mode='w', encoding='utf-8') as json_file:
            json.dump(final_data, json_file, indent=2, ensure_ascii=False)
        
        print(f"\nSuccessfully combined data sources.")
        print(f"Final JSON file saved to: {output_path}")
    except Exception as e:
        print(f"Error saving final JSON file to {output_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a final output JSON by preparing dialogue from scene_info.json and loading audio clips from human_videoid.json.")
    parser.add_argument("video_folder", help="The path to the main video folder (e.g., 'videos/adzYW5DZoWs').")
 
    args = parser.parse_args()
    
    video_folder_path = Path(args.video_folder)
    video_id = video_folder_path.name

    scenes_json_input_file = video_folder_path / f"{video_id}_scenes" / "scene_info.json"
    human_videoid_json_input_file = video_folder_path / f"human_{video_id}.json"
    
    final_output_file = video_folder_path / f"final_data_human.json"

    generate_final_output(str(scenes_json_input_file), str(human_videoid_json_input_file), str(final_output_file))