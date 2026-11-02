"""
extract_sample_frames_with_landmarks.py

Script ekstraksi 1 sample video menjadi 30 frame JPG dengan visualisasi
garis-garis keypoint MediaPipe Holistic (Pose, Tangan Kiri, Tangan Kanan),
TANPA garis mesh wajah (sesuai konfigurasi pada extract_keypoints_from_video.py).

- Sumber Video : D:/Projects/SIBI Recognition Lecturers/For_Extract/Adik/Adik_2026-10-02 10-06-56-00.00.00.331-00.00.03.708-seg01.mp4
- Output Frame : D:/Projects/SIBI Recognition Lecturers/Sample_Extracted_Frames/Adik_2026-10-02_seg01
- Target Frame : 30 frame terdistribusi merata (smooth temporal resampling)
- Landmark     : Pose (33), Tangan Kiri (21), Tangan Kanan (21), Face Mesh dimatikan
- Multiplatform: Kompatibel Windows native (D:/...) dan WSL (/mnt/d/...)
"""

import os
import argparse
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


# ==================== KONFIGURASI DEFAULT ====================
def resolve_path(wsl_path, win_path):
    if os.path.exists(wsl_path):
        return wsl_path
    elif os.path.exists(win_path):
        return win_path
    else:
        # Fallback relative terhadap root workspace
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        subpath = win_path.replace(r"D:\Projects\SIBI Recognition Lecturers", "").lstrip("\\/")
        return os.path.join(base, subpath)


# Path Video Target Default
DEFAULT_SAMPLE_VIDEO = resolve_path(
    "/mnt/d/Projects/SIBI Recognition Lecturers/For_Extract/Adik/Adik_2026-10-02 10-06-56-00.00.00.331-00.00.03.708-seg01.mp4",
    r"D:\Projects\SIBI Recognition Lecturers\For_Extract\Adik\Adik_2026-10-02 10-06-56-00.00.00.331-00.00.03.708-seg01.mp4"
)

# Folder Penyimpanan Output Default
DEFAULT_OUTPUT_DIR = resolve_path(
    "/mnt/d/Projects/SIBI Recognition Lecturers/Sample_Extracted_Frames/Adik_2026-10-02_seg01",
    r"D:\Projects\SIBI Recognition Lecturers\Sample_Extracted_Frames\Adik_2026-10-02_seg01"
)

DEFAULT_SEQUENCE_LENGTH = 30
# ==============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Definisi koneksi pose tubuh saja (tanpa koneksi wajah landmark 0..10)
BODY_POSE_CONNECTIONS = frozenset([
    c for c in mp_holistic.POSE_CONNECTIONS
    if c[0] > 10 and c[1] > 10
])


def mediapipe_detection(image, model):
    """Proses gambar dengan MediaPipe Holistic dalam format RGB."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results


def extract_keypoints(results):
    """
    Ekstrak koordinat numerik 258 dimensi (persis dengan extract_keypoints_from_video.py):
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


def draw_styled_landmarks(image, results, draw_pose_face_connections=False):
    """
    Menggambar garis-garis keypoint MediaPipe Holistic:
    - Pose: Garis tubuh dan titik persendian
    - Tangan Kiri: Garis dan titik 21 landmark tangan kiri
    - Tangan Kanan: Garis dan titik 21 landmark tangan kanan
    - Face Mesh: TIDAK digambar sama sekali (sesuai spesifikasi)
    """
    annotated = image.copy()

    # 1. Gambar Pose Landmarks & Connections
    if results.pose_landmarks:
        pose_connections = (
            mp_holistic.POSE_CONNECTIONS
            if draw_pose_face_connections
            else BODY_POSE_CONNECTIONS
        )

        # Style titik & garis Pose (Teal / Biru Kehijauan)
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            pose_connections,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(200, 110, 31), thickness=2, circle_radius=3  # BGR
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(150, 200, 50), thickness=2, circle_radius=2  # BGR
            )
        )

    # 2. Gambar Tangan Kiri (Ungu / Magenta cerah)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(121, 22, 76), thickness=2, circle_radius=3  # BGR
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(250, 44, 121), thickness=2, circle_radius=2  # BGR
            )
        )

    # 3. Gambar Tangan Kanan (Kuning / Oranye cerah)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(245, 117, 66), thickness=2, circle_radius=3  # BGR
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(66, 230, 245), thickness=2, circle_radius=2  # BGR
            )
        )

    # CATATAN PENTING:
    # Face mesh (results.face_landmarks) TIDAK digambar di sini,
    # menjaga area wajah tetap bersih dan fokus pada gestur SIBI.

    return annotated


def extract_sample_video(
    video_path=DEFAULT_SAMPLE_VIDEO,
    output_dir=DEFAULT_OUTPUT_DIR,
    sequence_length=DEFAULT_SEQUENCE_LENGTH,
    draw_pose_face_connections=False,
    draw_overlay=True,
    save_npy=True
):
    print("=" * 70)
    print("     EKSTRAKSI 30 FRAME SAMPLE VIDEO DENGAN KEYPOINT MEDIAPIPE")
    print("=" * 70)
    print(f"Video Sumber   : {video_path}")
    print(f"Folder Output  : {output_dir}")
    print(f"Target Frame   : {sequence_length} frame (JPG)")
    print(f"Garis Wajah    : {'Nonaktif (Wajah Bersih)' if not draw_pose_face_connections else 'Standar Pose'}")
    print(f"Overlay Info   : {'Aktif' if draw_overlay else 'Nonaktif'}")
    print(f"Simpan .npy    : {'Ya (keypoints_30x258.npy)' if save_npy else 'Tidak'}")
    print("=" * 70 + "\n")

    if not os.path.isfile(video_path):
        print(f"❌ Error: File video tidak ditemukan di:\n   {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Buka video dengan OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Gagal membuka video: {video_path}")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Metadata Video:")
    print(f"       - Resolusi     : {video_w} x {video_h}")
    print(f"       - Total Frame  : {total_video_frames} frame")
    print(f"       - Framerate    : {video_fps:.2f} fps\n")

    if total_video_frames <= 0:
        print("❌ Error: Total frame video bernilai 0.")
        cap.release()
        return

    # Hitung 30 target index secara temporal linier (sama seperti di extract_keypoints_from_video.py)
    target_indices = np.round(np.linspace(0, total_video_frames - 1, sequence_length)).astype(int)
    target_set = set(target_indices)

    print(f"[INFO] Membaca {sequence_length} frame target...")
    extracted_raw_frames = {}
    frame_idx = 0

    with tqdm(total=total_video_frames, desc="Membaca Video") as pbar:
        while cap.isOpened() and len(extracted_raw_frames) < len(target_set):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in target_set:
                extracted_raw_frames[frame_idx] = frame
            frame_idx += 1
            pbar.update(1)

    cap.release()

    if len(extracted_raw_frames) == 0:
        print("❌ Error: Tidak ada frame yang berhasil dibaca dari video.")
        return

    available_indices = sorted(extracted_raw_frames.keys())
    last_valid_frame = extracted_raw_frames[available_indices[-1]]

    # Inisialisasi MediaPipe Holistic
    # (refine_face_landmarks=False persis seperti di extract_keypoints_from_video.py)
    print("\n[INFO] Memproses deteksi MediaPipe dan menyimpan frame JPG...")
    keypoints_sequence = []
    saved_files = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        refine_face_landmarks=False,
        enable_segmentation=False
    ) as holistic:

        for seq_i, target_idx in enumerate(tqdm(target_indices, desc="Ekstraksi Frame")):
            raw_frame = extracted_raw_frames.get(target_idx, last_valid_frame)

            # Deteksi MediaPipe
            results = mediapipe_detection(raw_frame, holistic)

            # Ekstrak data numerik 258 fitur (Pose 132 + LH 63 + RH 63)
            kp = extract_keypoints(results)
            keypoints_sequence.append(kp)

            # Gambar visualisasi garis keypoints TANPA face mesh
            annotated_frame = draw_styled_landmarks(
                raw_frame, results, draw_pose_face_connections=draw_pose_face_connections
            )

            # Tambahkan overlay label (opsional)
            if draw_overlay:
                pose_ok = "OK" if results.pose_landmarks else "--"
                lh_ok = "OK" if results.left_hand_landmarks else "--"
                rh_ok = "OK" if results.right_hand_landmarks else "--"
                label = f"Frame {seq_i+1:02d}/{sequence_length} (Src #{target_idx:03d}) | Pose:{pose_ok} LH:{lh_ok} RH:{rh_ok}"

                # Background badge untuk teks agar terbaca jelas
                cv2.rectangle(annotated_frame, (10, 10), (450, 42), (0, 0, 0), -1)
                cv2.putText(
                    annotated_frame,
                    label,
                    (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            # Simpan file JPG (format penamaan: frame_00.jpg ... frame_29.jpg)
            filename = f"frame_{seq_i:02d}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_files.append(save_path)

    # Simpan array numerik keypoints (30, 258) bila diaktifkan
    if save_npy:
        npy_path = os.path.join(output_dir, f"keypoints_{sequence_length}x258.npy")
        np.save(npy_path, np.array(keypoints_sequence, dtype=np.float32))
        print(f"\n[INFO] Data keypoints numerik ({sequence_length}x258) tersimpan di:\n       {npy_path}")

    print("\n" + "=" * 70)
    print("                      EKSTRAKSI BERHASIL!")
    print("=" * 70)
    print(f"Total frame JPG disimpan : {len(saved_files)} file (frame_00.jpg s/d frame_{len(saved_files)-1:02d}.jpg)")
    print(f"Direktori Output         : {output_dir}")
    print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ekstraksi 30 frame JPG dari video SIBI dengan visualisasi keypoints MediaPipe Holistic (tanpa face mesh)."
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=DEFAULT_SAMPLE_VIDEO,
        help="Path file video sumber (default: video Adik terbaru)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Direktori tujuan penyimpanan frame JPG"
    )
    parser.add_argument(
        "--seq-len", "-n",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Jumlah frame yang diekstrak (default: 30)"
    )
    parser.add_argument(
        "--include-pose-face",
        action="store_true",
        help="Aktifkan koneksi garis pose di wajah (mata/hidung/telinga). Default: False (wajah bersih)"
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Hilangkan badge info frame teks di pojok kiri atas"
    )
    parser.add_argument(
        "--no-npy",
        action="store_true",
        help="Jangan simpan file koordinat numerik .npy"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_sample_video(
        video_path=args.video,
        output_dir=args.output,
        sequence_length=args.seq_len,
        draw_pose_face_connections=args.include_pose_face,
        draw_overlay=not args.no_overlay,
        save_npy=not args.no_npy
    )
