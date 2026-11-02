# Dokumentasi Model SIBI Recognition — 15 Kelas

> Dokumen ini dibuat otomatis pada akhir proses training untuk mencatat seluruh kelas kata yang dilatih pada sesi ini.

---

### 📌 Ringkasan Informasi Sesi Training
* **Waktu Selesai Training** : `2026-10-27 16:51:02`
* **Jumlah Kelas**          : **15 kata**
* **Arsitektur Model**       : BiLSTM (96 $\rightarrow$ 64) + Attention Layer + Focal Loss + Label Smoothing
* **Dimensi Input**          : `(None, 30, 258)` (30 frame @ 30fps $\times$ 258 keypoints)
* **File Output Model**      : `model15class.keras` / `model15class.h5`
* **Folder Output**          : `Epoch500_20261027_162855`

---

### 📊 Performa Evaluasi Model
* **Test Accuracy** : **100.00%**
* **Test Loss**     : **0.0014**
* **Macro F1-Score**: **1.0000**
* **Weighted F1**   : **1.0000**
* **Data Train (Asli / Setelah Aug)** : `1050` / `9450` sequence
* **Data Val (Murni)**                : `225` sequence
* **Data Test (Murni)**               : `226` sequence

---

### 📋 Daftar Lengkap Kelas Kata yang Dilatih (15 Kelas)

| No | Label Kelas (Action) | Data Train (Asli) | Train (Augmentasi) | Val (Asli) | Test (Asli) | Precision | Recall | F1-Score | Support |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Adik** | 71 | 639 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 2 | **Apa** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 3 | **Ayah** | 68 | 612 | 15 | 14 | 1.000 | 1.000 | 1.000 | 14 |
| 4 | **Baik** | 69 | 621 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 5 | **Berapa** | 71 | 639 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 6 | **Bertemu** | 71 | 639 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 7 | **Cantik** | 69 | 621 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 8 | **Dari** | 71 | 639 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 9 | **Dia** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 10 | **Dimana** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 11 | **Ganteng** | 69 | 621 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 12 | **Gemuk** | 69 | 621 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 13 | **Hallo** | 70 | 630 | 15 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 14 | **Hobi** | 71 | 639 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 15 | **Ibu** | 71 | 639 | 15 | 16 | 1.000 | 1.000 | 1.000 | 16 |

---

### 💡 Catatan Pipeline Augmentasi:
Model ini dilatih dengan 8 augmentasi khusus *right-hand dominant* & multi-jarak:
1. `aug_zoom_half_body` (1.40x - 2.05x zoom setengah badan khas webcam)
2. `aug_centered_scale` (0.85x - 1.35x variasi jarak natural)
3. `aug_translation` (+-8% pergeseran posisi duduk)
4. `aug_time_warp` (0.80x - 1.20x variasi kecepatan isyarat)
5. `aug_hand_dropout` (1-3 frame toleransi kedipan tracking)
6. `aug_visibility_variation` (tahan variasi visibility browser)
7. `aug_slight_rotation` (+-6 derajat kemiringan sudut webcam)
8. `aug_gaussian_noise` (noise sensor halus)
