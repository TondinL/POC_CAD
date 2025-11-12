import subprocess
import argparse
from pathlib import Path
import shutil

from joblib import Parallel, delayed

# prova a trovare ffprobe nel PATH, altrimenti metti il tuo percorso
FFPROBE = shutil.which("ffprobe") or r"C:/Users/tondi/Downloads/ffmpeg-8.0-essentials_build/ffmpeg-8.0-essentials_build/bin/ffprobe.exe"   # <-- cambia se diverso
FFMPEG  = shutil.which("ffmpeg")  or r"C:/Users/tondi/Downloads/ffmpeg-8.0-essentials_build/ffmpeg-8.0-essentials_build/bin/ffmpeg.exe"    # opzionale

def run(cmd):
    """Esegue un comando subprocess e mostra eventuali errori."""
    print(">>", " ".join(map(str, cmd)))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("⚠️ Errore ffmpeg/ffprobe:")
        print(p.stderr)
        raise RuntimeError(f"Comando fallito ({p.returncode})")
    return p


def video_process(video_file_path, dst_root_path, ext, fps=-1, size=240):
    """Estrae i frame da un singolo video e li salva in jpg."""
    video_file_path = Path(video_file_path)
    if video_file_path.suffix.lower() != ext.lower():
        print(f"Skip {video_file_path.name} (estensione diversa da {ext})")
        return

    # Crea cartella di output
    dst_dir_path = Path(dst_root_path) / video_file_path.stem
    dst_dir_path.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Ottieni metadati del video con ffprobe
    ffprobe_cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-of", "default=noprint_wrappers=1:nokey=1",
        "-show_entries", "stream=width,height,avg_frame_rate,duration",
        str(video_file_path)
    ]
    p = run(ffprobe_cmd)
    res = p.stdout.splitlines()
    if len(res) < 4:
        print(f"⚠️ ffprobe output insufficiente per {video_file_path}")
        return

    width, height = int(res[0]), int(res[1])
    num, den = (res[2].split('/') + ['1'])[:2]
    num = float(num)
    den = float(den) if float(den) != 0 else 1.0
    frame_rate = num / den
    try:
        duration = float(res[3])
    except Exception:
        duration = 0.0

    n_frames = int(frame_rate * duration) if frame_rate > 0 and duration > 0 else 0

    # Evita doppia elaborazione se i frame già esistono
    n_exist = len([x for x in dst_dir_path.iterdir() if x.suffix == ".jpg"])
    if n_frames and n_exist >= n_frames:
        print(f"✅ Già estratti: {video_file_path.name}")
        return

    # 2️⃣ Costruisci filtro video (ridimensionamento proporzionale)
    vf = f"scale=-1:{size}" if width > height else f"scale={size}:-1"

    # 3️⃣ Comando ffmpeg per estrarre i frame
    ffmpeg_cmd = [FFMPEG, "-i", str(video_file_path), "-vf", vf, "-qscale:v", "2"]
    if fps > 0:
        ffmpeg_cmd += ["-r", str(fps)]  # riscampiona FPS
    ffmpeg_cmd += ["-threads", "1", str(dst_dir_path / "image_%05d.jpg")]

    run(ffmpeg_cmd)
    print(f"✅ OK: {video_file_path.name} → {dst_dir_path}")


def class_process(class_dir_path, dst_root_path, ext, fps=-1, size=240):
    """Elabora tutti i video dentro una sottocartella (es. NORMAL/ABNORMAL)."""
    class_dir_path = Path(class_dir_path)
    if not class_dir_path.is_dir():
        print(f"Skip {class_dir_path}, non è una directory.")
        return

    dst_class_path = Path(dst_root_path) / class_dir_path.name
    dst_class_path.mkdir(parents=True, exist_ok=True)

    for video_file_path in sorted(class_dir_path.iterdir()):
        video_process(video_file_path, dst_class_path, ext, fps, size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", type=Path, help="Directory con i video di input")
    parser.add_argument("dst_path", type=Path, help="Directory dove salvare i frame")
    parser.add_argument("dataset", type=str,
                        help="Tipo dataset (kinetics | mit | ucf101 | hmdb51 | activitynet)")
    parser.add_argument("--n_jobs", default=-1, type=int, help="Numero di job paralleli")
    parser.add_argument("--fps", default=-1, type=int,
                        help="Frame rate di output (-1 = originale)")
    parser.add_argument("--size", default=240, type=int, help="Altezza dei frame di output")
    args = parser.parse_args()

    # Determina estensione in base al tipo di dataset
    if args.dataset in ["kinetics", "mit", "activitynet"]:
        ext = ".mp4"
    else:
        ext = ".avi"

    class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]
    test_set_video_path = args.dir_path / "test"
    if test_set_video_path.exists():
        class_dir_paths.append(test_set_video_path)

    Parallel(n_jobs=args.n_jobs, backend="threading")(
        delayed(class_process)(class_dir_path, args.dst_path, ext, args.fps, args.size)
        for class_dir_path in class_dir_paths
    )

    print("\n✅ Estrazione completata con successo!\n")