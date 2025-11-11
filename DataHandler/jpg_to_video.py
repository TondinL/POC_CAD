import os
import sys
import subprocess


def create_video_from_images(image_folder, output_folder, output_video_name, framerate=30):
    # Verifica che la directory delle immagini esista
    if not os.path.isdir(image_folder):
        print(f"Errore: La directory {image_folder} non esiste")
        return

    # Verifica che la directory di output esista, altrimenti la crea
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Verifica che ci siano file JPG nella directory delle immagini
    if not any(fname.endswith('.jpg') for fname in os.listdir(image_folder)):
        print(f"Errore: Nessun file JPG trovato nella directory {image_folder}")
        return

    # Creazione del pattern per le immagini
    images_pattern = os.path.join(image_folder, "frame_%04d.jpg")
    output_video = os.path.join(output_folder, output_video_name)

    # Comando FFmpeg per creare il video
    ffmpeg_command = [
        "ffmpeg",
        "-framerate", str(framerate),
        "-i", images_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video
    ]
    # Esecuzione del comando FFmpeg
    subprocess.run(ffmpeg_command)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Uso: python jpg_to_video.py <path_to_image_directory> <path_to_output_directory> <output_video_name> <framerate>")
    else:
        image_folder = sys.argv[1]
        output_folder = sys.argv[2]
        output_video_name = sys.argv[3]
        framerate = int(sys.argv[4])
        create_video_from_images(image_folder, output_folder, output_video_name, framerate)
