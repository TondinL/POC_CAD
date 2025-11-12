import cv2
import os
import argparse

def process_video(input_video_path, output_video_path):
    # Apri il video con OpenCV
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Errore nell'apertura del video {input_video_path}")
        return

    # Ottieni framerate (fps) e dimensione dei frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Definisci il codec e il file di output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Puoi cambiare il codec se necessario
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Aggiungi il framerate in sovraimpressione
        frame_number += 1
        timestamp = frame_number / fps
        text = f"Frame: {frame_number}, FPS: {fps:.2f}, Time: {timestamp:.2f}s"
        
        # Posiziona il testo in alto a sinistra del frame
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Scrivi il frame nel file di output
        out.write(frame)

    cap.release()
    out.release()

def process_videos_in_folder(input_folder, output_folder):
    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Processa ogni video nella cartella
    for filename in os.listdir(input_folder):
        if filename.endswith(('.mp4', '.avi', '.mkv')):
            input_video_path = os.path.join(input_folder, filename)
            output_video_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_with_fps.avi")

            print(f"Processing video {filename}...")
            process_video(input_video_path, output_video_path)

    print("Elaborazione dei video completata.")

if __name__ == "__main__":
    # Definisci gli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Aggiungi framerate in sovraimpressione ai video.")
    parser.add_argument("--input_folder", type=str, required=True, help="Percorso della cartella contenente i video.")
    parser.add_argument("--output_folder", type=str, required=True, help="Cartella di output per i video processati.")
    
    args = parser.parse_args()

    # Esegui il processo di elaborazione dei video
    process_videos_in_folder(args.input_folder, args.output_folder)
