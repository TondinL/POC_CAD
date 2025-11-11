import cv2
import sys

def crop_video(input_path, output_path, x, y, w, h):
    # Carica il video di input
    cap = cv2.VideoCapture(input_path)

    # Controlla se il video è stato caricato correttamente
    if not cap.isOpened():
        print(f"Errore nell'apertura del file video {input_path}")
        sys.exit()

    # Ottieni le proprietà del video originale
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Stampa informazioni sul video originale
    print(f"Frame Rate: {fps} FPS")
    print(f"Dimensioni: {width}x{height}")
    print(f"Numero totale di frame: {total_frames}")

    # Imposta il codec e crea un oggetto VideoWriter per l'output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ritaglia il frame
        cropped_frame = frame[y:y+h, x:x+w]

        # Scrivi il frame ritagliato nel file di output
        out.write(cropped_frame)

    # Rilascia le risorse
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Utilizzo: python crop_video.py <input_path> <output_path> <x> <y> <w> <h>")
        sys.exit()

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    x = int(sys.argv[3])
    y = int(sys.argv[4])
    w = int(sys.argv[5])
    h = int(sys.argv[6])

    crop_video(input_path, output_path, x, y, w, h)
