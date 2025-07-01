import argparse
import csv
import json
import os
import sys

def extract_audio_clips(csv_path, video_id, audio_description_id):
    """
    Extracts audio clip transcripts from a CSV for a given video and audio description ID.

    Parameters:
        csv_path (str): Path to the CSV file.
        video_id (str): Youtube video ID
        audio_description_id (str): Audio Description ID.

    Returns:
        dict: A dictionary with a single key "audio_clips" containing its metadata.
        Format: dict: {"audio_clips": [ {start_time: float, end_time: float, description_style: str, text: str}, ... ]}
    """
    clips = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('youtube_id') == video_id and row.get('audio_description_id') == audio_description_id:
                clips.append({
                    "start_time": float(row.get("audio_clip_start_time")),
                    "end_time": float(row.get("audio_clip_end_time")),
                    "description_style": row.get("audio_clip_playback_type"),
                    "text": row.get("audio_clip_transcript")
                })

    # sort by start_time
    clips.sort(key=lambda clip: clip["start_time"])
    return {"audio_clips": clips}   


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio clips from a CSV by video_id and audio_description_id"
    )
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("video_id", help="Video ID to filter on")
    parser.add_argument("audio_description_id", help="Audio description ID to filter on")
    args = parser.parse_args()

    result = extract_audio_clips(args.csv_path, args.video_id, args.audio_description_id)
    json_output = json.dumps(result, indent=2, ensure_ascii=False)

    # Build output directory and file path:
    #   videos/{video_id}/human_{video_id}_{audio_description_id}.json
    output_dir = os.path.join("videos", args.video_id)
    os.makedirs(output_dir, exist_ok=True)

    file_name = f"human_{args.video_id}_{args.audio_description_id}.json"
    output_path = os.path.join(output_dir, file_name)

    try:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write(json_output)
        print(f"Written JSON output to {output_path}")
    except IOError as e:
        print(f"Error writing to {output_path}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()