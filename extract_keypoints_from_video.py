"""
extract_keypoints_from_video.py

Script ekstraksi keypoints MediaPipe Holistic dari video dataset SIBI:
- Sumber Video  : D:/Projects/SIBI Recognition Lecturers/For_Extract (gabungan Raw_Datasets + Raw_Datasets_Addition)
- Output Folder : D:/Projects/SIBI Recognition Lecturers/KeypointsUpdate
- Dimensi Fitur : 258 fitur per frame (33 Pose x 4 + 21 LH x 3 + 21 RH x 3), TANPA face mesh.
- Sequence Len  : Tepat 30 frame per sequence (smooth temporal resampling).
- Quality Check : Memvalidasi apakah tangan kanan (dominant hand) terdeteksi (mencegah data kosong/rusak).
- Multiplatform : Kompatibel dengan Windows native (D:/...) dan WSL (/mnt/d/...).
"""

import os
import json
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ==================== KONFIGURASI ====================
def resolve_path(wsl_path, win_path):
    if os.path.exists(wsl_path):
        return wsl_path
    elif os.path.exists(win_path):
        return win_path
    else:
        # Fallback relative
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        folder_name = os.path.basename(win_path)
        return os.path.join(base, folder_name)

RAW_VIDEO_ROOT = resolve_path(
    "/mnt/d/Projects/SIBI Recognition Lecturers/For_Extract",
    r"D:\Projects\SIBI Recognition Lecturers\For_Extract"
)
OUTPUT_KEYPOINTS_ROOT = resolve_path(
    "/mnt/d/Projects/SIBI Recognition Lecturers/KeypointsUpdate",
    r"D:\Projects\SIBI Recognition Lecturers\KeypointsUpdate"
)

SEQUENCE_LENGTH = 30
FITUR_PER_FRAME = 258
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")

# Bila True, sequence yang sudah pernah diekstrak lengkap (30 file .npy) dilewati
OVERWRITE_EXISTING = False

# Filter Kualitas: Abaikan video jika tangan kanan terdeteksi kurang dari rasio minimum (misal < 20% frame)
FILTER_LOW_QUALITY = True
MIN_HAND_DETECTION_RATIO = 0.20  # Minimal 6 dari 30 frame harus ada tangan kanan terdeteksi
# =====================================================

mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    """
    Ekstrak koordinat:
    - Pose: 33 landmark x 4 (x, y, z, visibility) = 132
    - Left Hand: 21 landmark x 3 (x, y, z) = 63
    - Right Hand: 21 landmark x 3 (x, y, z) = 63
    Total: 258 fitur per frame (TANPA face mesh).
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 4, dtype=np.float32)

    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)

    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32)

    return np.concatenate([pose, lh, rh]).astype(np.float32)


def mediapipe_detection(image, model):
    """Proses gambar dengan MediaPipe Holistic dalam format RGB."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return results


def extract_video_to_keypoints(video_path, holistic):
    """
    Membaca video, melakukan temporal resampling tepat 30 frame,
    mengekstrak keypoints, dan memeriksa kualitas deteksi tangan kanan.
    
    Returns:
        tuple (list_of_30_keypoints, rh_detection_ratio, is_valid)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0, False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, 0.0, False

    # Smooth temporal sampling tepat 30 frame merata sepanjang video
    target_indices = np.round(np.linspace(0, total_frames - 1, SEQUENCE_LENGTH)).astype(int)
    target_set = set(target_indices)

    # Dictionary frame_index -> frame image
    extracted_frames = {}
    frame_idx = 0

    while cap.isOpened() and len(extracted_frames) < len(target_set):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_set:
            extracted_frames[frame_idx] = frame
        frame_idx += 1

    cap.release()

    if len(extracted_frames) == 0:
        return None, 0.0, False

    # Proses tepat 30 frame yang telah disample
    keypoints_sequence = []
    rh_detected_count = 0

    # Fallback frame terakhir bila ada frame yang gagal dibaca
    available_indices = sorted(extracted_frames.keys())
    last_valid_frame = extracted_frames[available_indices[-1]]

    for idx in target_indices:
        frame = extracted_frames.get(idx, last_valid_frame)
        results = mediapipe_detection(frame, holistic)
        kp = extract_keypoints(results)
        keypoints_sequence.append(kp)

        if results.right_hand_landmarks:
            rh_detected_count += 1

    rh_ratio = rh_detected_count / float(SEQUENCE_LENGTH)
    is_valid = (rh_ratio >= MIN_HAND_DETECTION_RATIO) if FILTER_LOW_QUALITY else True

    return keypoints_sequence, rh_ratio, is_valid


def main():
    print("=" * 60)
    print("    EKSTRAKSI KEYPOINTS SIBI — PIPELINE TEROPTIMASI")
    print("=" * 60)
    print(f"Sumber Video   : {RAW_VIDEO_ROOT}")
    print(f"Output Tujuan  : {OUTPUT_KEYPOINTS_ROOT}")
    print(f"Sequence Len   : {SEQUENCE_LENGTH} frame per isyarat")
    print(f"Dimensi Fitur  : {FITUR_PER_FRAME} fitur/frame (Pose 132 + LH 63 + RH 63)")
    print(f"Quality Filter : {'Aktif (Min ' + str(int(MIN_HAND_DETECTION_RATIO*100)) + '% frame tangan kanan)' if FILTER_LOW_QUALITY else 'Nonaktif'}")
    print("=" * 60 + "\n")

    if not os.path.isdir(RAW_VIDEO_ROOT):
        print(f"❌ Error: RAW_VIDEO_ROOT tidak ditemukan: {RAW_VIDEO_ROOT}")
        return

    os.makedirs(OUTPUT_KEYPOINTS_ROOT, exist_ok=True)

    actions = sorted([
        d for d in os.listdir(RAW_VIDEO_ROOT)
        if os.path.isdir(os.path.join(RAW_VIDEO_ROOT, d))
    ])
    print(f"Ditemukan {len(actions)} kelas kata untuk diekstrak.\n")

    stats = {
        "total_classes": len(actions),
        "total_extracted": 0,
        "total_skipped_existing": 0,
        "total_skipped_low_quality": 0,
        "classes": {}
    }

    # Inisialisasi MediaPipe Holistic
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,           # Konsisten dengan complexity pada MediaPipe Web TFJS
            refine_face_landmarks=False,  # Face mesh dimatikan (tidak diperlukan untuk SIBI 258 fitur)
            enable_segmentation=False) as holistic:

        for action in actions:
            action_dir = os.path.join(RAW_VIDEO_ROOT, action)
            dest_action_dir = os.path.join(OUTPUT_KEYPOINTS_ROOT, action)
            os.makedirs(dest_action_dir, exist_ok=True)

            videos = sorted([
                f for f in os.listdir(action_dir)
                if f.lower().endswith(VIDEO_EXTENSIONS)
            ])

            # Cari nomor sequence folder awal agar penomoran urut: 0, 1, 2...
            existing_folders = [
                int(f) for f in os.listdir(dest_action_dir)
                if f.isdigit() and len(os.listdir(os.path.join(dest_action_dir, f))) >= SEQUENCE_LENGTH
            ]
            current_seq_idx = max(existing_folders) + 1 if existing_folders else 0

            class_extracted = 0
            class_low_quality = 0
            class_skipped_existing = len(existing_folders)

            # Jika overwrite False dan jumlah folder sudah sama/lebih dengan jumlah video, lewati
            if not OVERWRITE_EXISTING and len(existing_folders) >= len(videos) and len(videos) > 0:
                print(f"[{action:14s}] Sudah selesai ({len(existing_folders)} sequence) -> Dilewati")
                stats["classes"][action] = {
                    "videos": len(videos),
                    "extracted": len(existing_folders),
                    "low_quality_skipped": 0
                }
                stats["total_skipped_existing"] += len(existing_folders)
                continue

            for video_name in tqdm(videos, desc=f"{action:<14s}"):
                save_dir = os.path.join(dest_action_dir, str(current_seq_idx))

                # Bila folder sudah ada dan lengkap
                if not OVERWRITE_EXISTING and os.path.isdir(save_dir) and \
                        len(os.listdir(save_dir)) >= SEQUENCE_LENGTH:
                    current_seq_idx += 1
                    continue

                video_path = os.path.join(action_dir, video_name)
                keypoints_seq, rh_ratio, is_valid = extract_video_to_keypoints(video_path, holistic)

                if keypoints_seq is None:
                    continue

                if not is_valid:
                    class_low_quality += 1
                    continue

                # Simpan sequence (30 file .npy)
                os.makedirs(save_dir, exist_ok=True)
                for frame_i, kp in enumerate(keypoints_seq):
                    np.save(os.path.join(save_dir, f"{frame_i}.npy"), kp)

                current_seq_idx += 1
                class_extracted += 1

            stats["total_extracted"] += class_extracted
            stats["total_skipped_low_quality"] += class_low_quality
            stats["classes"][action] = {
                "videos": len(videos),
                "extracted": class_extracted + class_skipped_existing,
                "low_quality_skipped": class_low_quality
            }

    # Simpan ringkasan log ekstraksi
    summary_path = os.path.join(OUTPUT_KEYPOINTS_ROOT, "extraction_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("          EKSTRAKSI KEYPOINTS SELESAI!")
    print("=" * 60)
    print(f"Total kelas diproses         : {stats['total_classes']}")
    print(f"Total sequence baru tersimpan: {stats['total_extracted']}")
    print(f"Total video kualitas rendah  : {stats['total_skipped_low_quality']} (tangan tidak terdeteksi)")
    print(f"Lokasi Output                : {OUTPUT_KEYPOINTS_ROOT}")
    print(f"File Ringkasan               : {summary_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
