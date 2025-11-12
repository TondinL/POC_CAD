import os
import sys
from pathlib import Path
from moviepy import VideoFileClip

def convert_mp4_to_avi(video_path, output_path):
    try:
        video_path = Path(video_path)
        output_video_path = Path(os.path.splitext(output_path)[0] + ".avi")

        with VideoFileClip(str(video_path)) as clip:
            # usa lo stesso fps del sorgente; utile per VFR
            fps = clip.fps or 25

            # AVI + Xvid (qualità alta: qscale 2; 1=quasi lossless ma file enormi)
            clip.write_videofile(
                str(output_video_path),
                codec="libxvid",
                audio_codec="libmp3lame",   # in alternativa: "pcm_s16le" (WAV non compresso)
                fps=fps,
                ffmpeg_params=[
                    "-qscale:v", "2",       # qualità visiva alta; più basso = migliore
                    "-pix_fmt", "yuv420p"   # compatibilità player
                ],
                threads=0,                  # lascia ffmpeg usare tutti i core
            )

            print(f"OK: {video_path.name} → {output_video_path}")

    except Exception as e:
        print(f"Errore su {video_path}: {e}")

def process_videos_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for name in os.listdir(input_folder):
        if name.lower().endswith(".mp4"):
            src = os.path.join(input_folder, name)
            dst = os.path.join(output_folder, name)  # l’estensione verrà cambiata in .avi
            convert_mp4_to_avi(src, dst)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_mp4_to_avi.py <input_folder> <output_folder>")
        sys.exit(1)
    process_videos_in_folder(sys.argv[1], sys.argv[2])
