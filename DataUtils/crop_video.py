import cv2, os, sys, argparse
from pathlib import Path

# Target aspect ratio (UMN = 4:3)
TARGET_AR_W, TARGET_AR_H = 4, 3

def write_video(out_path, fps, size, codec="XVID"):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not vw.isOpened():
        raise RuntimeError(f"Impossibile aprire VideoWriter su {out_path}")
    return vw

def clamp_crop(frame, x, y, w, h):
    H, W = frame.shape[:2]
    x = max(0, min(x, W-1))
    y = max(0, min(y, H-1))
    w = max(1, min(w, W-x))
    h = max(1, min(h, H-y))
    return frame[y:y+h, x:x+w]

def crop_to_aspect_center(frame, ar_w=4, ar_h=3):
    H, W = frame.shape[:2]
    target_ar = ar_w / ar_h
    cur_ar = W / H
    if cur_ar > target_ar:
        # troppo largo → taglio ai lati
        target_w = int(round(H * target_ar))
        x0 = (W - target_w) // 2
        return frame[:, x0:x0+target_w]
    elif cur_ar < target_ar:
        # troppo alto → taglio sopra/sotto
        target_h = int(round(W / target_ar))
        y0 = (H - target_h) // 2
        return frame[y0:y0+target_h, :]
    else:
        return frame

def process_video_umn(src, dst, cut_top):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[SKIP] Non apro {src}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (W, H - cut_top)

    vw = write_video(dst, fps, out_size)
    while True:
        ret, f = cap.read()
        if not ret: break
        f2 = clamp_crop(f, 0, cut_top, W, H-cut_top)
        vw.write(f2)
    cap.release(); vw.release()
    print(f"[OK] UMN: {src.name} → {dst.name}  ({W}x{H} → {out_size[0]}x{out_size[1]})")

def process_video_med(src, dst):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[SKIP] Non apro {src}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, f0 = cap.read()
    if not ret:
        cap.release(); print(f"[SKIP] Vuoto: {src}"); return

    roi = crop_to_aspect_center(f0, TARGET_AR_W, TARGET_AR_H)
    outW, outH = roi.shape[1], roi.shape[0]

    vw = write_video(dst, fps, (outW, outH))
    vw.write(roi)

    while True:
        ret, f = cap.read()
        if not ret: break
        f2 = crop_to_aspect_center(f, TARGET_AR_W, TARGET_AR_H)
        vw.write(f2)
    cap.release(); vw.release()
    print(f"[OK] MED: {src.name} → {dst.name}  ({W}x{H} → {outW}x{outH})")

def main():
    ap = argparse.ArgumentParser(description="Prep video AVI: UMN = rimozione top overlay, MED = crop centrale 4:3.")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--dataset", choices=["UMN","MED"], required=True)
    ap.add_argument("--cut-top", type=int, default=80, help="Pixel da tagliare in alto (solo UMN).")
    args = ap.parse_args()

    inp = Path(args.input_dir); out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for p in sorted(inp.glob("*.avi")):
        dst = out / p.name
        if args.dataset == "UMN":
            process_video_umn(p, dst, args.cut_top)
        else:
            process_video_med(p, dst)

if __name__ == "__main__":
    main()
