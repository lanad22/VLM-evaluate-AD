import csv
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
                "sequence_num": sequence_counter
            })
            sequence_counter += 1
            
            if end >= scene.get("end_time", 0) - gap_threshold:
                continuing_dialogue = True
            else:
                continuing_dialogue = False
                
            last_dialogue_end = end
            
    return dialogue

def generate_final_output(csv_path, scenes_path, output_path):
    try:
        with open(scenes_path, mode='r', encoding='utf-8') as scenes_file:
            scenes_data = json.load(scenes_file)
        
        print(f"Successfully loaded scene data from {scenes_path}.")
        
        dialogue_timestamps = prepare_dialogue(scenes_data)
        print(f"Successfully prepared {len(dialogue_timestamps)} dialogue entries.")

    except FileNotFoundError:
        print(f"Error: The scenes file '{scenes_path}' was not found. Aborting.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{scenes_path}'.")
        return

    audio_clips = []
    try:
        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            print(f"Successfully opened {csv_path} to generate audio clips.")

            for i, row in enumerate(csv_reader):
                if not row: continue
                try:
                    columns = row[0].split(',')
                    audio_clips.append({
                        "start_time": float(columns[15]),
                        "type": "Visual",
                        "description_style": columns[14],
                        "text": columns[18],
                    })
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not process row {i+1} in CSV. Error: {e}.")
        
        audio_clips.sort(key=lambda clip: clip['start_time'])
        print(f"Successfully parsed and sorted {len(audio_clips)} audio clips.")

    except FileNotFoundError:
        print(f"Error: The CSV file '{csv_path}' was not found. Aborting.")
        return


    final_data = {
        "dialogue_timestamps": dialogue_timestamps,
        "audio_clips": audio_clips,
    }

    with open(output_path, mode='w', encoding='utf-8') as json_file:
        json.dump(final_data, json_file, indent=2, ensure_ascii=False)
    
    print(f"\nSuccessfully combined data sources.")
    print(f"Final JSON file saved to: {output_path}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a final output JSON by processing a video's CSV and scene info.")
    parser.add_argument("video_folder", help="The path to the main video folder (e.g., 'videos/adzYW5DZoWs').")
 
    args = parser.parse_args()
    
    video_folder_path = Path(args.video_folder)

    video_id = video_folder_path.name

    csv_input_file = video_folder_path / f"human_{video_id}.csv"
    scenes_json_input_file = video_folder_path / f"{video_id}_scenes" / "scene_info.json"
    final_output_file = video_folder_path / f"human_{video_id}_output.json"

    generate_final_output(str(csv_input_file), str(scenes_json_input_file), str(final_output_file))
