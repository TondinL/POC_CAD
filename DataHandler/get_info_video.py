import ffmpeg
import sys

def get_video_info(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        total_frames = int(video_info['nb_frames'])
        fps = eval(video_info['r_frame_rate'])  # Evaluates the frame rate expression
        width = int(video_info['width'])
        height = int(video_info['height'])
        duration = float(video_info['duration'])
        codec = video_info['codec_name']
        bitrate = int(probe['format']['bit_rate'])

        print(f"Totale frame: {total_frames}")
        print(f"FPS: {fps}")
        print(f"Larghezza: {width}")
        print(f"Altezza: {height}")
        print(f"Durata: {duration} secondi")
        print(f"Codec: {codec}")
        print(f"Bitrate: {bitrate} bps")

    except ffmpeg.Error as e:
        print(f"Errore: Impossibile ottenere informazioni dal video {video_path}")
        print(e)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python get_info_video_ffmpeg.py <percorso_video>")
    else:
        video_path = sys.argv[1]
        get_video_info(video_path)