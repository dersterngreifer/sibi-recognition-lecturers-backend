"""
Script untuk memindahkan sub-folder kedua (yang berisi video) dari setiap
folder dataset ke satu folder tujuan bernama "Lists".

Struktur sebelum:
    Datasets/
        Ayah-20260910T124004Z-1-001/
            Ayah/                <-- ini yang mau dipindahkan
                Ayah-00.00.00...-seg01.mp4
                ... (50 video)
        Baik-20260910T081127Z-1-001/
            Baik/
                ...

Struktur sesudah:
    Datasets/
        Lists/
            Ayah/
                ...50 video...
            Baik/
                ...50 video...
        Ayah-20260910T124004Z-1-001/   <-- folder lama jadi kosong (bisa dihapus)
        Baik-20260910T081127Z-1-001/

Cara pakai:
    1. Sesuaikan variabel ROOT_DIR di bawah.
    2. Jalankan dulu dengan DRY_RUN = True untuk melihat preview aksi
       tanpa benar-benar memindahkan apa pun.
    3. Kalau sudah yakin, ubah DRY_RUN = False lalu jalankan lagi.
"""

import shutil
from pathlib import Path

# ====== KONFIGURASI ======
ROOT_DIR = Path(r"D:\Projects\SIBI Recognition Lecturers\Datasets")
LISTS_DIR_NAME = "Lists"
DRY_RUN = False  # set False kalau sudah yakin untuk benar-benar memindahkan
DELETE_EMPTY_PARENT = True  # hapus folder "Ayah-xxxxx-1-001" setelah isinya dipindah
# ==========================


def get_single_subfolder(folder: Path) -> Path | None:
    """Ambil satu-satunya subfolder di dalam `folder`. None kalau tidak ada / lebih dari 1."""
    subdirs = [p for p in folder.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        print(f"  [SKIP] '{folder.name}' punya {len(subdirs)} subfolder, bukan 1 -> dilewati")
        return None
    return subdirs[0]


def merge_move(src: Path, dst: Path):
    """Pindahkan src ke dst. Kalau dst sudah ada, gabungkan isi filenya satu per satu."""
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return

    print(f"  [MERGE] '{dst}' sudah ada, menggabungkan isi file satu per satu...")
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            print(f"    [WARNING] File '{target.name}' sudah ada di tujuan, dilewati: {item}")
            continue
        shutil.move(str(item), str(target))
    # folder src sekarang seharusnya kosong (atau sisa file yang bentrok)
    try:
        src.rmdir()
    except OSError:
        print(f"    [INFO] '{src}' tidak kosong sepenuhnya, tidak dihapus")


def main():
    if not ROOT_DIR.exists():
        print(f"ROOT_DIR tidak ditemukan: {ROOT_DIR}")
        return

    lists_dir = ROOT_DIR / LISTS_DIR_NAME
    if not DRY_RUN:
        lists_dir.mkdir(exist_ok=True)

    # ambil semua folder level-1 di root, kecuali folder Lists itu sendiri
    parent_folders = [
        p for p in ROOT_DIR.iterdir()
        if p.is_dir() and p.name != LISTS_DIR_NAME
    ]

    print(f"Ditemukan {len(parent_folders)} folder di root.\n")

    for parent in sorted(parent_folders):
        print(f"Memproses: {parent.name}")
        inner = get_single_subfolder(parent)
        if inner is None:
            continue

        destination = lists_dir / inner.name

        if DRY_RUN:
            print(f"  [DRY-RUN] Akan memindahkan:\n    dari: {inner}\n    ke  : {destination}")
        else:
            merge_move(inner, destination)
            print(f"  [OK] Dipindahkan ke {destination}")

            if DELETE_EMPTY_PARENT:
                try:
                    # hapus parent kalau sudah kosong (tidak ada isi lain selain subfolder tadi)
                    if not any(parent.iterdir()):
                        parent.rmdir()
                        print(f"  [OK] Folder kosong '{parent.name}' dihapus")
                    else:
                        print(f"  [INFO] '{parent.name}' masih ada isi lain, tidak dihapus")
                except OSError as e:
                    print(f"  [WARNING] Gagal menghapus '{parent.name}': {e}")

        print()

    print("Selesai.")
    if DRY_RUN:
        print("\n>> Ini masih DRY RUN. Tidak ada file yang benar-benar dipindahkan.")
        print(">> Set DRY_RUN = False di bagian atas script untuk menjalankan sungguhan.")


if __name__ == "__main__":
    main()
