"""
auto_crop_dataset.py

Crop otomatis semua video dataset SIBI, dengan 2 jenis pemotongan:

  1. SPASIAL: crop area frame (kiri-kanan-atas-bawah) supaya framing
     konsisten (setengah badan ke atas), margin atas kepala dihitung
     ADAPTIF berdasarkan ukuran kepala orangnya di video (bukan
     persentase tetap dari tinggi frame) -> tidak lagi kepotong di
     kepala walau jarak kamera beda-beda tiap video.

  2. TEMPORAL: memotong bagian AWAL/AKHIR video yang tangannya diam
     (idle, misal tangan terlipat di depan badan sebelum/sesudah
     gesture), berdasarkan skor gerakan (motion score) tangan per
     frame. Cuma bagian video dimana tangan benar-benar bergerak
     (sedang melakukan gesture) yang dipertahankan, plus sedikit
     padding di kiri-kanan supaya awal gerakan tidak terlalu mepet.

Requirement:
  pip install mediapipe opencv-python numpy tqdm imageio-ffmpeg
  (ffmpeg dari PATH sistem dipakai kalau ada, kalau tidak otomatis
  pakai binary bawaan package imageio-ffmpeg)
"""

import os
import shutil
import subprocess
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm


def resolve_ffmpeg_binary():
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
OUTPUT_VIDEO_ROOT = r"D:\Projects\SIBI Recognition Lecturers\DatasetsCropped2"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi")
OVERWRITE_EXISTING = False

# --- Margin crop SPASIAL ---
MARGIN_SIDE = 0.10       # ruang kiri-kanan di luar jangkauan tangan (fraksi lebar frame)
MARGIN_BOTTOM = 0.06     # ruang di bawah batas pinggul/tangan terendah (fraksi tinggi frame)
HIP_EXTENSION_FACTOR = 1.15  # 1.0 = pas garis pinggul, >1.0 = sedikit di bawah pinggul

# Margin atas kepala sekarang ADAPTIF: dihitung dari lebar antar-telinga (head_width)
# sebagai proxy ukuran kepala, dikali faktor ini, ditambahkan di atas titik hidung.
# Naikkan angka ini kalau hasil crop MASIH kepotong di kepala.
HEAD_TOP_FACTOR = 1.6
# Fallback kalau telinga tidak terdeteksi sama sekali di satu video: pakai
# persentase tetap dari tinggi frame (jaga-jaga, dinaikkan dari versi sebelumnya).
FALLBACK_MARGIN_TOP = 0.18

# --- Deteksi TEMPORAL (potong bagian diam) ---
ENABLE_TEMPORAL_TRIM = True
# Skor gerakan = perpindahan centroid tangan antar frame sample (pixel).
# Di bawah ini dianggap "diam". Sesuaikan kalau terlalu sensitif/kurang sensitif.
MOTION_THRESHOLD_PX = 12.0
# Berapa frame sample berturut-turut yang harus "aktif" sebelum dianggap
# benar-benar mulai gesture (menghindari noise/getaran kamera dianggap gerakan).
MIN_ACTIVE_RUN = 2
# Padding tambahan di awal & akhir segmen aktif (dalam frame ASLI video,
# bukan frame sample), supaya awal gerakan tidak terlalu mepet terpotong.
ACTION_PADDING_FRAMES = 6
# Kalau video ternyata TIDAK ADA gerakan terdeteksi sama sekali (semua diam),
# video ini di-skip & dicatat untuk dicek manual, bukan diproses asal-asalan.

# Resolusi output seragam untuk semua video. Harus genap.
OUTPUT_WIDTH = 480
OUTPUT_HEIGHT = 640

# Setiap berapa frame dilakukan sampling deteksi saat pass 1
SAMPLE_STRIDE = 2
# ======================================

mp_holistic = mp.solutions.holistic

# Index landmark MediaPipe Pose yang dipakai
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24


def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    return model.process(image_rgb)


def hand_centroid(hand_landmarks, frame_w, frame_h):
    if not hand_landmarks:
        return None
    xs = [p.x * frame_w for p in hand_landmarks.landmark]
    ys = [p.y * frame_h for p in hand_landmarks.landmark]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def scan_video(video_path, holistic):
    """
    Pass 1: scan video, kumpulkan data per-frame-sample yang dipakai untuk
    (a) menghitung kotak crop spasial, dan (b) menentukan rentang waktu aktif.
    """
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    xs, ys = [], []           # untuk batas horizontal (bahu + tangan)
    hip_ys = []
    nose_ys = []
    head_widths = []
    sample_records = []       # (frame_idx, hand_centroid_or_None)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_STRIDE == 0:
            results = mediapipe_detection(frame, holistic)

            centroid = None
            hc_list = []
            for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
                c = hand_centroid(hand, frame_w, frame_h)
                if c:
                    hc_list.append(c)
            if hc_list:
                centroid = (
                    sum(c[0] for c in hc_list) / len(hc_list),
                    sum(c[1] for c in hc_list) / len(hc_list),
                )
            sample_records.append((frame_idx, centroid))

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                nose_ys.append(lm[NOSE].y * frame_h)
                hip_ys.append((lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2 * frame_h)

                le, re = lm[LEFT_EAR], lm[RIGHT_EAR]
                if le.visibility > 0.3 and re.visibility > 0.3:
                    head_widths.append(abs(le.x - re.x) * frame_w)

                for idx in (LEFT_SHOULDER, RIGHT_SHOULDER):
                    xs.append(lm[idx].x * frame_w)
                    ys.append(lm[idx].y * frame_h)

            for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
                if hand:
                    for p in hand.landmark:
                        xs.append(p.x * frame_w)
                        ys.append(p.y * frame_h)

        frame_idx += 1

    cap.release()
    total_frames = frame_idx

    if not nose_ys or not hip_ys or not xs:
        return None

    return {
        "frame_w": frame_w, "frame_h": frame_h, "fps": fps,
        "total_frames": total_frames,
        "min_x": min(xs), "max_x": max(xs),
        "min_y_hand_pose": min(ys),
        "nose_top": min(nose_ys),
        "hip_line": max(hip_ys),
        "head_width": (sum(head_widths) / len(head_widths)) if head_widths else None,
        "sample_records": sample_records,
    }


def compute_spatial_crop(bounds):
    fw, fh = bounds["frame_w"], bounds["frame_h"]

    if bounds["head_width"]:
        top = bounds["nose_top"] - HEAD_TOP_FACTOR * bounds["head_width"]
    else:
        top = bounds["nose_top"] - FALLBACK_MARGIN_TOP * fh

    bottom = bounds["hip_line"] * HIP_EXTENSION_FACTOR
    bottom = max(bottom, bounds["min_y_hand_pose"])
    bottom += MARGIN_BOTTOM * fh

    left = bounds["min_x"] - MARGIN_SIDE * fw
    right = bounds["max_x"] + MARGIN_SIDE * fw

    top = max(0, top)
    bottom = min(fh, bottom)
    left = max(0, left)
    right = min(fw, right)

    crop_w = right - left
    crop_h = bottom - top
    if crop_w <= 10 or crop_h <= 10:
        return None

    x = int(left)
    y = int(top)
    w = int(crop_w) - (int(crop_w) % 2)
    h = int(crop_h) - (int(crop_h) % 2)
    return x, y, w, h


def compute_temporal_range(bounds):
    """
    Return (start_frame, end_frame) dalam frame ASLI video (bukan sample),
    atau None kalau tidak ada gerakan terdeteksi / fitur ini dimatikan.
    """
    if not ENABLE_TEMPORAL_TRIM:
        return 0, bounds["total_frames"] - 1

    records = bounds["sample_records"]
    motions = []  # (frame_idx, motion_score)
    prev_centroid = None
    prev_idx = None

    for idx, centroid in records:
        if centroid is not None and prev_centroid is not None:
            dist = ((centroid[0] - prev_centroid[0]) ** 2 +
                    (centroid[1] - prev_centroid[1]) ** 2) ** 0.5
            # normalisasi terhadap jarak antar-sample frame supaya adil
            # walau SAMPLE_STRIDE berubah
            step = max(1, idx - prev_idx)
            motions.append((idx, dist / step * SAMPLE_STRIDE))
        prev_centroid = centroid
        prev_idx = idx

    if not motions:
        return None

    active_flags = [m[1] > MOTION_THRESHOLD_PX for m in motions]

    # cari run aktif terpanjang yang memenuhi MIN_ACTIVE_RUN, ambil dari
    # active pertama sampai active terakhir dalam window itu (union sederhana:
    # ambil rentang dari kemunculan aktif pertama sampai terakhir yang valid)
    active_indices = [motions[i][0] for i in range(len(motions)) if active_flags[i]]

    if not active_indices:
        return None  # video ini dianggap diam sepenuhnya -> skip, cek manual

    # buang blip noise: hanya pertahankan titik aktif yang punya minimal
    # MIN_ACTIVE_RUN tetangga aktif berturut-turut di sekitarnya
    filtered = []
    for i in range(len(motions)):
        if not active_flags[i]:
            continue
        lo = max(0, i - MIN_ACTIVE_RUN + 1)
        hi = min(len(motions), i + MIN_ACTIVE_RUN)
        if sum(active_flags[lo:hi]) >= MIN_ACTIVE_RUN:
            filtered.append(motions[i][0])

    if not filtered:
        filtered = active_indices  # fallback kalau filter terlalu ketat

    start_frame = max(0, min(filtered) - ACTION_PADDING_FRAMES)
    end_frame = min(bounds["total_frames"] - 1, max(filtered) + ACTION_PADDING_FRAMES)

    if end_frame - start_frame < 5:
        return None  # rentang aktif terlalu pendek, mencurigakan -> skip

    return start_frame, end_frame


def crop_with_ffmpeg(input_path, output_path, crop_box, time_range, fps):
    x, y, w, h = crop_box
    start_frame, end_frame = time_range
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame) / fps

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vf = (
        f"crop={w}:{h}:{x}:{y},"
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black"
    )

    cmd = [
        FFMPEG_BINARY, "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", input_path,
        "-t", f"{duration_sec:.3f}",
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
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

    skipped_no_landmark = []
    skipped_no_motion = []

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

                bounds = scan_video(input_path, holistic)
                if bounds is None:
                    skipped_no_landmark.append(input_path)
                    continue

                crop_box = compute_spatial_crop(bounds)
                if crop_box is None:
                    skipped_no_landmark.append(input_path)
                    continue

                time_range = compute_temporal_range(bounds)
                if time_range is None:
                    skipped_no_motion.append(input_path)
                    continue

                crop_with_ffmpeg(input_path, output_path, crop_box, time_range, bounds["fps"])

    print(f"\nSelesai. Hasil crop tersimpan di: {OUTPUT_VIDEO_ROOT}")
    if skipped_no_landmark:
        print(f"\n{len(skipped_no_landmark)} video dilewati (landmark tidak cukup terdeteksi):")
        for p in skipped_no_landmark:
            print(f"  - {p}")
    if skipped_no_motion:
        print(f"\n{len(skipped_no_motion)} video dilewati (tidak ada gerakan tangan terdeteksi "
              f"sama sekali / rentang aktif terlalu pendek -> cek manual, mungkin MOTION_THRESHOLD_PX "
              f"terlalu tinggi atau video memang bermasalah):")
        for p in skipped_no_motion:
            print(f"  - {p}")


if __name__ == "__main__":
    main()