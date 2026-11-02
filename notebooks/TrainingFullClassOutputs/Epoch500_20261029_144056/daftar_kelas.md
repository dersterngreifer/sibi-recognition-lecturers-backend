# Dokumentasi Model SIBI Recognition — 63 Kelas

> Dokumen ini dibuat otomatis pada akhir proses training untuk mencatat seluruh kelas kata yang dilatih pada sesi ini.

---

### 📌 Ringkasan Informasi Sesi Training
* **Waktu Selesai Training** : `2026-10-29 15:58:46`
* **Jumlah Kelas**          : **63 kata**
* **Arsitektur Model**       : BiLSTM (96 $\rightarrow$ 64) + Attention Layer + Focal Loss + Label Smoothing
* **Dimensi Input**          : `(None, 30, 258)` (30 frame @ 30fps $\times$ 258 keypoints)
* **File Output Model**      : `modelFullClass.keras` / `modelFullClass.h5`
* **Folder Output**          : `Epoch500_20261029_144056`

---

### 📊 Performa Evaluasi Model
* **Test Accuracy** : **99.68%**
* **Test Loss**     : **0.0004**
* **Macro F1-Score**: **0.9968**
* **Weighted F1**   : **0.9968**
* **Data Train (Asli / Setelah Aug)** : `4409` / `39681` sequence
* **Data Val (Murni)**                : `945` sequence
* **Data Test (Murni)**               : `945` sequence

---

### 📋 Daftar Lengkap Kelas Kata yang Dilatih (63 Kelas)

| No | Label Kelas (Action) | Data Train (Asli) | Train (Augmentasi) | Val (Asli) | Test (Asli) | Precision | Recall | F1-Score | Support |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Adik** | 71 | 639 | 15 | 15 | 0.938 | 1.000 | 0.968 | 15 |
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
| 13 | **Hallo** | 71 | 639 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 14 | **Hobi** | 71 | 639 | 15 | 15 | 1.000 | 0.933 | 0.966 | 15 |
| 15 | **Ibu** | 71 | 639 | 16 | 15 | 1.000 | 0.933 | 0.966 | 15 |
| 16 | **Jumat** | 73 | 657 | 15 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 17 | **Jurusan** | 71 | 639 | 15 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 18 | **Kabar** | 71 | 639 | 15 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 19 | **Kakak** | 34 | 306 | 8 | 7 | 1.000 | 1.000 | 1.000 | 7 |
| 20 | **Kalian** | 70 | 630 | 15 | 15 | 1.000 | 0.933 | 0.966 | 15 |
| 21 | **Kami** | 71 | 639 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 22 | **Kamis** | 69 | 621 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 23 | **Kampus** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 24 | **Kamu** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 25 | **Kelas** | 73 | 657 | 15 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 26 | **Keluarga** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 27 | **Kemana** | 69 | 621 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 28 | **Kenapa** | 74 | 666 | 16 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 29 | **Kita** | 65 | 585 | 14 | 13 | 1.000 | 1.000 | 1.000 | 13 |
| 30 | **Kuliah** | 69 | 621 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 31 | **Kurus** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 32 | **Lucu** | 69 | 621 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 33 | **Malam** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 34 | **Mau** | 73 | 657 | 15 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 35 | **Mereka** | 71 | 639 | 15 | 15 | 0.938 | 1.000 | 0.968 | 15 |
| 36 | **Minggu** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 37 | **Nama** | 71 | 639 | 15 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 38 | **Pagi** | 69 | 621 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 39 | **Pelit** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 40 | **Pendidikan** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 41 | **Pintar** | 69 | 621 | 15 | 14 | 1.000 | 1.000 | 1.000 | 14 |
| 42 | **Pulang** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 43 | **Rabu** | 69 | 621 | 15 | 14 | 1.000 | 1.000 | 1.000 | 14 |
| 44 | **Ramah** | 90 | 810 | 20 | 19 | 1.000 | 1.000 | 1.000 | 19 |
| 45 | **Rendah Hati** | 71 | 639 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 46 | **Sabar** | 74 | 666 | 16 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 47 | **Sabtu** | 71 | 639 | 15 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 48 | **Sakit** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 49 | **Sampai Jumpa** | 74 | 666 | 16 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 50 | **Saya** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 51 | **Sekolah** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 52 | **Selamat** | 67 | 603 | 15 | 14 | 1.000 | 1.000 | 1.000 | 14 |
| 53 | **Selasa** | 71 | 639 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 54 | **Semester** | 65 | 585 | 14 | 13 | 1.000 | 1.000 | 1.000 | 13 |
| 55 | **Senang** | 69 | 621 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 56 | **Senin** | 74 | 666 | 15 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 57 | **Siang** | 69 | 621 | 15 | 14 | 1.000 | 1.000 | 1.000 | 14 |
| 58 | **Siapa** | 71 | 639 | 15 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 59 | **Sore** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 60 | **Suka** | 73 | 657 | 15 | 16 | 1.000 | 1.000 | 1.000 | 16 |
| 61 | **Terimakasih** | 70 | 630 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 62 | **Tinggal** | 69 | 621 | 15 | 15 | 1.000 | 1.000 | 1.000 | 15 |
| 63 | **Umur** | 70 | 630 | 15 | 15 | 0.938 | 1.000 | 0.968 | 15 |

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
