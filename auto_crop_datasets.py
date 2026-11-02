"""
auto_crop_dataset.py

Crop otomatis semua video dataset SIBI supaya framing konsisten
(setengah badan ke atas), berdasarkan deteksi MediaPipe per video
— bukan koordinat crop tetap, karena jarak/posisi kamera tiap video
bisa berbeda.

Cara kerja:
  1. Scan tiap video: deteksi kepala/bahu/pinggul + jangkauan kedua
     tangan di SEPANJANG video (supaya gerakan tangan saat isyarat
     tidak ikut terpotong).
  2. Hitung kotak crop per video (dengan margin aman).
  3. Panggil ffmpeg untuk crop + scale ke resolusi seragam.

Requirement:
  pip install mediapipe opencv-python numpy tqdm
  ffmpeg harus terinstall dan ada di PATH (cek: `ffmpeg -version`)
"""

import os
import shutil
import subprocess
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm


def resolve_ffmpeg_binary():
    """
    Cari ffmpeg yang bisa dipakai, urutannya:
    1. ffmpeg dari PATH sistem (kalau ada & sudah benar ter-install)
    2. binary bawaan package imageio-ffmpeg (tidak butuh PATH sama sekali)
    Kalau keduanya gagal, program berhenti dengan pesan jelas.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        print(f"Pakai ffmpeg dari PATH sistem: {system_ffmpeg}")
        return system_ffmpeg

    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"ffmpeg tidak ditemukan di PATH, pakai binary bawaan imageio-ffmpeg: {bundled}")
        return bundled
    except ImportError:
        raise RuntimeError(
            "ffmpeg tidak ditemukan di PATH sistem, dan package 'imageio-ffmpeg' "
            "belum terinstall. Jalankan: pip install imageio-ffmpeg"
        )


FFMPEG_BINARY = resolve_ffmpeg_binary()

# ====== KONFIGURASI — SESUAIKAN ======
RAW_VIDEO_ROOT = r"D:\Projects\SIBI Recognition Lecturers\Datasets"
OUTPUT_VIDEO_ROOT = r"D:\Projects\SIBI Recognition Lecturers\DatasetsCropped"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi")
OVERWRITE_EXISTING = False

# Margin tambahan di luar bounding box yang terdeteksi (fraksi dari lebar/tinggi frame)
MARGIN_TOP = 0.08      # ruang di atas kepala
MARGIN_SIDE = 0.10     # ruang kiri-kanan di luar jangkauan tangan
MARGIN_BOTTOM = 0.06   # ruang di bawah batas pinggul/tangan terendah

# Batas bawah crop: seberapa jauh dari kepala ke bawah badan yang diambil.
# 1.0 = sampai garis pinggul (hip) persis, >1.0 = melebihi pinggul dikit
# supaya tidak terlalu ketat kalau tangan biasa turun sedikit di bawah pinggang.
HIP_EXTENSION_FACTOR = 1.15

# Resolusi output seragam untuk semua video (lebar x tinggi). Harus genap.
OUTPUT_WIDTH = 480
OUTPUT_HEIGHT = 640

# Setiap berapa frame dilakukan sampling deteksi saat pass 1 (1 = tiap frame,
# lebih presisi tapi lebih lambat; 2-3 = lebih cepat, cukup akurat untuk video pendek)
SAMPLE_STRIDE = 2
# ======================================

mp_holistic = mp.solutions.holistic


def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    return results


def scan_video_bounds(video_path, holistic):
    """
    Pass 1: scan video, kumpulkan pixel coords dari kepala, bahu, pinggul,
    dan kedua tangan di semua frame yang di-sample. Return None kalau
    tidak ada landmark yang cukup terdeteksi (video bermasalah -> skip).
    """
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    xs, ys = [], []
    hip_ys = []
    head_ys = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_STRIDE == 0:
            results = mediapipe_detection(frame, holistic)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # nose (0) untuk batas atas kepala
                head_ys.append(lm[0].y * frame_h)
                # rata-rata hip kiri(23)-kanan(24) untuk batas bawah "setengah badan"
                hip_y = (lm[23].y + lm[24].y) / 2 * frame_h
                hip_ys.append(hip_y)
                # bahu kiri(11)-kanan(12) ikut jadi acuan horizontal juga
                for idx in (11, 12):
                    xs.append(lm[idx].x * frame_w)
                    ys.append(lm[idx].y * frame_h)

            for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
                if hand:
                    for p in hand.landmark:
                        xs.append(p.x * frame_w)
                        ys.append(p.y * frame_h)

        frame_idx += 1

    cap.release()

    if not head_ys or not hip_ys or not xs:
        return None

    return {
        "frame_w": frame_w,
        "frame_h": frame_h,
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y_hand_pose": min(ys),
        "top_head": min(head_ys),
        "hip_line": max(hip_ys),  # ambil yang paling bawah selama video (jaga-jaga gerak)
    }


def compute_crop_box(bounds):
    fw, fh = bounds["frame_w"], bounds["frame_h"]

    top = bounds["top_head"] - MARGIN_TOP * fh
    bottom = bounds["hip_line"] * HIP_EXTENSION_FACTOR
    # kalau tangan turun lebih rendah dari batas pinggul, ikutkan (jangan kepotong)
    bottom = max(bottom, bounds["min_y_hand_pose"])
    bottom += MARGIN_BOTTOM * fh

    left = bounds["min_x"] - MARGIN_SIDE * fw
    right = bounds["max_x"] + MARGIN_SIDE * fw

    # clamp ke batas frame asli
    top = max(0, top)
    bottom = min(fh, bottom)
    left = max(0, left)
    right = min(fw, right)

    crop_w = right - left
    crop_h = bottom - top

    if crop_w <= 10 or crop_h <= 10:
        return None

    # genapkan ke integer, dan pastikan lebar/tinggi genap (syarat encoder video)
    x = int(left)
    y = int(top)
    w = int(crop_w) - (int(crop_w) % 2)
    h = int(crop_h) - (int(crop_h) % 2)

    return x, y, w, h


def crop_with_ffmpeg(input_path, output_path, crop_box):
    x, y, w, h = crop_box
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # crop lalu scale+pad ke resolusi seragam, aspect ratio dijaga (tidak distorsi)
    vf = (
        f"crop={w}:{h}:{x}:{y},"
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black"
    )

    cmd = [
        FFMPEG_BINARY, "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",  # audio tidak diperlukan untuk dataset keypoints
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ffmpeg error] {os.path.basename(input_path)}: {result.stderr[-500:]}")
        return False
    return True


def main():
    if not os.path.isdir(RAW_VIDEO_ROOT):
        print(f"RAW_VIDEO_ROOT tidak ditemukan: {RAW_VIDEO_ROOT}")
        return

    actions = sorted([
        d for d in os.listdir(RAW_VIDEO_ROOT)
        if os.path.isdir(os.path.join(RAW_VIDEO_ROOT, d))
    ])
    print(f"Ditemukan {len(actions)} kelas kata: {actions}\n")

    skipped = []

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=False,
            enable_segmentation=False,
            model_complexity=1) as holistic:

        for action in actions:
            action_dir = os.path.join(RAW_VIDEO_ROOT, action)
            out_action_dir = os.path.join(OUTPUT_VIDEO_ROOT, action)
            videos = sorted([
                f for f in os.listdir(action_dir)
                if f.lower().endswith(VIDEO_EXTENSIONS)
            ])

            for video_name in tqdm(videos, desc=action):
                input_path = os.path.join(action_dir, video_name)
                output_path = os.path.join(out_action_dir, video_name)

                if not OVERWRITE_EXISTING and os.path.exists(output_path):
                    continue

                bounds = scan_video_bounds(input_path, holistic)
                if bounds is None:
                    skipped.append(input_path)
                    continue

                crop_box = compute_crop_box(bounds)
                if crop_box is None:
                    skipped.append(input_path)
                    continue

                crop_with_ffmpeg(input_path, output_path, crop_box)

    print(f"\nSelesai. Hasil crop tersimpan di: {OUTPUT_VIDEO_ROOT}")
    if skipped:
        print(f"\n{len(skipped)} video dilewati (landmark tidak cukup terdeteksi), perlu dicek manual:")
        for p in skipped:
            print(f"  - {p}")


if __name__ == "__main__":
    main()