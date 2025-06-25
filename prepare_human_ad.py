import csv
import json
import argparse
from pathlib import Path
import tempfile
import os

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

def convert_and_get_csv_path(csv_path):
    needs_conversion = False
    try:
        with open(csv_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            first_row = next(csv_reader, None)
            
            if first_row and len(first_row) == 1:
                needs_conversion = True
                print(f"Detected single-column CSV format. Converting...")
            else:
                print(f"Detected multi-column CSV format ({len(first_row) if first_row else 0} columns).")
                return csv_path
                
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    if not needs_conversion:
        return csv_path
    
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', prefix='converted_')
        os.close(temp_fd) 
        
        converted_rows = []
        skipped_rows = 0
        
        with open(csv_path, mode='r', encoding='utf-8') as input_file:
            csv_reader = csv.reader(input_file)
            
            for i, row in enumerate(csv_reader):
                if not row:
                    skipped_rows += 1
                    continue
                
                try:
                    converted_row = row[0].split(',')
                    converted_rows.append(converted_row)
                    
                except Exception as e:
                    print(f"Warning: Could not convert row {i+1}. Error: {e}")
                    skipped_rows += 1
        
        with open(temp_path, mode='w', encoding='utf-8', newline='') as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerows(converted_rows)
        
        print(f"Converted {len(converted_rows)} rows to proper CSV format")
        if skipped_rows > 0:
            print(f"Skipped {skipped_rows} empty/invalid rows")
        
        return temp_path
        
    except Exception as e:
        print(f"Error during CSV conversion: {e}")
        return None

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
    
    processed_csv_path = convert_and_get_csv_path(csv_path)
    if not processed_csv_path:
        print("Error: Could not process CSV file. Aborting.")
        return
    using_temp_file = processed_csv_path != csv_path
    
    try:
        audio_clips = []
        
        with open(processed_csv_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            print(f"Successfully opened CSV file to generate audio clips.")

            for i, row in enumerate(csv_reader):
                if not row: continue
                
                try:
                    if len(row) < 19:
                        print(f"Warning: Row {i+1} has only {len(row)} columns, expected at least 19. Skipping.")
                        continue
                    
                    audio_clips.append({
                        "start_time": float(row[15]),
                        "end_time": float(row[16]),
                        "type": "Visual",
                        "description_style": row[14],
                        "text": row[18],
                    })
                except (IndexError, ValueError) as e:
                    print(f"Warning: Could not process row {i+1} in CSV. Error: {e}.")
                    print(f"Row has {len(row)} columns")
        
        audio_clips.sort(key=lambda clip: clip['start_time'])
        print(f"Successfully parsed and sorted {len(audio_clips)} audio clips.")

    except FileNotFoundError:
        print(f"Error: The CSV file could not be found. Aborting.")
        return
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return
    finally:
        if using_temp_file and os.path.exists(processed_csv_path):
            try:
                os.unlink(processed_csv_path)
                print("Cleaned up temporary converted CSV file.")
            except:
                pass  

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
    final_output_file = video_folder_path / f"final_data_human.json"

    generate_final_output(str(csv_input_file), str(scenes_json_input_file), str(final_output_file))