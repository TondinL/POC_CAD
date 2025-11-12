import os
import sys
import ffmpeg

def analyze_video(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        total_frames = int(video_info['nb_frames'])
        fps = eval(video_info['r_frame_rate'])  # Evaluates the frame rate expression
        width = int(video_info['width'])
        height = int(video_info['height'])
        duration = float(video_info['duration'])

        video_name = os.path.basename(video_path).split('.')[0]

        print(f"Video: {video_name}")
        print(f"  Totale frame: {total_frames}")
        print(f"  FPS: {fps}")
        print(f"  Width: {width}")
        print(f"  Height: {height}")
        print(f"  Durata: {duration} secondi\n")

        return total_frames, duration

    except ffmpeg.Error as e:
        print(f"Errore: Impossibile ottenere informazioni dal video {video_path}")
        print(e)
        return 0, 0.0

def analyze_videos_in_folder(folder_path):
    total_duration = 0.0
    total_frames = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                video_path = os.path.join(root, file)
                frames, duration = analyze_video(video_path)
                total_frames += frames
                total_duration += duration

    print(f"Durata totale di tutti i video: {total_duration} secondi")
    print(f"Totale frame di tutti i video: {total_frames}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python analyze_videos.py <path_to_video_directory>")
    else:
        folder_path = sys.argv[1]
        analyze_videos_in_folder(folder_path)
