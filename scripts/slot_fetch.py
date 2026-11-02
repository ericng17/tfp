"""
slot_fetch.py
Fetch traffic camera images for all 28 cameras from link_camera.py

- Method: Based directly on slot_fetch.py (requests / urllib fallback)
- Source: Reads directly from link_camera.py
- Supports single run (for Cron / Windows Task Scheduler) or continuous loop
"""

import os
import sys
import time
import re
from datetime import datetime
from pathlib import Path

# Thử dùng requests nếu đã cài, nếu chưa có thì dùng urllib có sẵn trong Python
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    HAS_REQUESTS = False

# =========================
# CONFIG
# =========================

# Thư mục lưu ảnh
BASE_DIR = Path("data/raw/snapshots")
BASE_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT = 10  # seconds
LINK_FILE = Path("link_camera.py")

# =========================
# LOAD CAMERAS TỪ LINK_CAMERA.PY
# =========================

def remove_vietnamese_accents(text: str) -> str:
    patterns = {
        '[àáảãạăằắẳẵặâầấẩẫậ]': 'a', '[ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ]': 'A',
        '[đ]': 'd', '[Đ]': 'D',
        '[èéẻẽẹêềếểễệ]': 'e', '[ÈÉẺẼẸÊỀẾỂỄỆ]': 'E',
        '[ìíỉĩị]': 'i', '[ÌÍỈĨỊ]': 'I',
        '[òóỏõọôồốổỗộơờớởỡợ]': 'o', '[ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ]': 'O',
        '[ùúủũụưừứửữự]': 'u', '[ÙÚỦŨỤƯỪỨỬỮỰ]': 'U',
        '[ỳýỷỹỵ]': 'y', '[ỲÝỶỸỴ]': 'Y'
    }
    for p, r in patterns.items():
        text = re.sub(p, r, text)
    return text

def load_cameras():
    cameras = {}
    if not LINK_FILE.exists():
        print(f"[ERROR] Không tìm thấy file {LINK_FILE}")
        return cameras

    with open(LINK_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        if ": " not in line:
            continue
        raw_name, raw_url = line.split(": ", 1)
        url = raw_url.strip()
        if not url.startswith("http"):
            url = "https://" + url

        # Bỏ dấu tiếng Việt và chuẩn hóa tên camera sang ASCII
        name_no_accents = remove_vietnamese_accents(raw_name)
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", name_no_accents.replace(" ", "_").replace("-", "_"))
        clean_name = re.sub(r"_+", "_", clean_name).strip("_")

        # Trích xuất camId nếu có để gắn vào tên thư mục
        cam_id_match = re.search(r"camId=([a-f0-9]+)", url)
        suffix = f"_{cam_id_match.group(1)[-4:]}" if cam_id_match else ""
        folder_key = f"{clean_name}{suffix}"

        cameras[folder_key] = url

    return cameras

# =========================
# UTILS (THEO ĐÚNG SLOT_FETCH.PY)
# =========================

def fetch_image(url: str) -> bytes | None:
    try:
        if HAS_REQUESTS:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200 and r.content:
                return r.content
        else:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
                if response.status == 200:
                    content = response.read()
                    if content:
                        return content
    except Exception as e:
        print(f"[ERROR] Fetch failed: {url[:60]}... | {e}")
    return None


def save_image(cam_id: str, content: bytes, ts: datetime):
    cam_dir = BASE_DIR / cam_id
    cam_dir.mkdir(parents=True, exist_ok=True)

    fname = ts.strftime("%Y%m%d_%H%M%S") + ".jpg"
    out_path = cam_dir / fname

    with open(out_path, "wb") as f:
        f.write(content)

    print(f"[OK] Saved {cam_id} → {out_path.name} ({len(content)/1024:.1f} KB)")


# =========================
# MAIN
# =========================

def run_once(cameras: dict):
    print(f"🚦 Traffic slot fetch started lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ts = datetime.now()
    saved = 0

    for cam_id, url in cameras.items():
        print(f"📡 Fetching {cam_id} ...")
        img = fetch_image(url)
        if img is None:
            print(f"[WARN] No image for {cam_id}")
            continue

        save_image(cam_id, img, ts)
        saved += 1
        time.sleep(0.3)  # tránh spam server

    print(f"✅ Hoàn thành: Đã lưu {saved}/{len(cameras)} camera")
    return saved


def main():
    cameras = load_cameras()
    print(f"📋 Đã nạp {len(cameras)} camera từ {LINK_FILE}")

    # Nếu truyền cờ --loop (ví dụ: python slot_fetch.py --loop 300)
    # thì script sẽ tự động lặp lại sau mỗi N giây (mặc định 300s = 5 phút)
    if "--loop" in sys.argv:
        try:
            idx = sys.argv.index("--loop")
            interval = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            interval = 300  # 5 phút

        print(f"🔄 Chế độ chạy liên tục: Tự động quét mỗi {interval} giây (Nhấn Ctrl+C để dừng)")
        cycle = 1
        while True:
            print(f"\n--- [Lần chạy #{cycle}] ---")
            start = time.time()
            run_once(cameras)
            cycle += 1
            elapsed = time.time() - start
            wait = max(0.0, interval - elapsed)
            time.sleep(wait)
    else:
        # Mặc định chạy 1 lần (dành cho Cron job / Task Scheduler)
        run_once(cameras)


if __name__ == "__main__":
    main()
