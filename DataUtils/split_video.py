import cv2
import sys
import os

def split_video(input_path, start_frame, end_frame, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Verifica i frame totali nel video
    if start_frame >= total_frames or end_frame > total_frames:
        print(f"Errore: I frame specificati superano il numero totale di frame del video (totale frame: {total_frames}).")
        return

    # Stampa delle informazioni sul video di input
    print(f"Informazioni sul video di input:")
    print(f" - Numero totale di frame: {total_frames}")
    print(f" - FPS: {fps}")
    print(f" - Larghezza: {width}")
    print(f" - Altezza: {height}")

    # Posiziona il video al frame di inizio
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    current_frame = start_frame
    frame_count = 0
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1
        frame_count += 1

    cap.release()
    out.release()

    # Stampa delle informazioni sul video di output
    print(f"Informazioni sul video di output:")
    print(f" - Numero totale di frame: {frame_count}")
    print(f'Video salvato con successo in "{output_path}"')

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Uso: python split_video.py <percorso_video> <start_frame> <end_frame> <output_path>")
    else:
        video_path = sys.argv[1]
        start_frame = int(sys.argv[2])
        end_frame = int(sys.argv[3])
        output_path = sys.argv[4]
        split_video(video_path, start_frame, end_frame, output_path)
