import io
import os
import time
import hashlib
import logging
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, CommitOperationAdd

# Load các biến môi trường từ file .env
load_dotenv()

# ==========================================
# CẤU HÌNH HỆ THỐNG TỪ .ENV
# ==========================================

# 1. TÙY CHỌN LƯU LOCAL
SAVE_LOCAL = os.getenv("SAVE_LOCAL", "True").lower() in ("true", "1", "t")
BASE_DIR = Path(os.getenv("BASE_DIR", "data/raw/snapshots"))

# 2. DISCORD WEBHOOK
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# 3. NGƯỠNG CẢNH BÁO (THRESHOLDS)
MAX_FAILED_PER_CYCLE = int(os.getenv("MAX_FAILED_PER_CYCLE", 10))
MAX_CONSECUTIVE_FAILS = int(os.getenv("MAX_CONSECUTIVE_FAILS", 3))

# 4. HUGGING FACE CONFIG
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_ID = os.getenv("HF_REPO_ID", "")

# Cấu hình tần suất commit lên HF (12 chu kỳ x 5 phút = 60 phút / 1 commit)
UPLOAD_EVERY_N_CYCLES = int(os.getenv("UPLOAD_EVERY_N_CYCLES", 12))

# 5. CẤU HÌNH MẠNG
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 10))
INTERVAL_SECONDS = int(os.getenv("INTERVAL_SECONDS", 300))
IMAGE_HANDLER_URL = os.getenv(
    "IMAGE_HANDLER_URL", 
    "http://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id="
)

# ==========================================
# LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("camera_crawler.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

CAMERAS = [
    {"id": "5deb576d1dc17d7c5515ad08", "name": "Cong_Hoa_Tan_Ky_Tan_Quy"},
    {"id": "662b57471afb9c00172d9095", "name": "Cong_Hoa_Ap_Bac"},
    {"id": "59414c883dcfc400106f260b", "name": "Cong_Hoa_Binh_Gia"},
    {"id": "58ad6214bd82540010390be2", "name": "Cong_Hoa_Hoang_Hoa_Tham"},
    {"id": "662b57711afb9c00172d90a7", "name": "Cong_Hoa_Duong_C4"},
    {"id": "586e1d4ef9fab7001111b0a2", "name": "Cong_Hoa_Duong_C18"},
    {"id": "587476e3b807da0011e33cee", "name": "Cong_Hoa_Ut_Tich1"},
    {"id": "5874769eb807da0011e33cec", "name": "Cong_Hoa_Ut_Tich2"},
    {"id": "5d8cdbdc766c88001718896a", "name": "Hoang_Van_Thu_Cong_Hoa"},
    {"id": "587c8661b807da0011e33d43", "name": "Le_Van_Si_Hoang_Van_Thu"},
    {"id": "5d8cdb4b766c880017188966", "name": "Hoang_Van_Thu_Ut_Tich"},
    {"id": "5deb576d1dc17d7c5515ad0a", "name": "Hoang_Van_Thu_Xuan_Dieu"},
    {"id": "58747923b807da0011e33cf5", "name": "Nut_Giao_Bay_Hien2"},
    {"id": "5d8cdb0a766c880017188964", "name": "Truong_Chinh_Xuan_Hong"},
    {"id": "662b5a9c1afb9c00172d9240", "name": "Truong_Chinh_Nguyen_Thai_Binh"},
    {"id": "5deb576d1dc17d7c5515ad09", "name": "Truong_Chinh_Hoang_Hoa_Tham"},
    {"id": "662b5a401afb9c00172d91fc", "name": "Truong_Chinh_Nguyen_Hong_Dao"},
    {"id": "662b5b481afb9c00172d92a8", "name": "Truong_Chinh_Truong_Cong_Dinh"},
    {"id": "662b5b271afb9c00172d9296", "name": "Truong_Chinh_Tan_Hai"},
    {"id": "662b58401afb9c00172d90f5", "name": "Hoang_Van_Thu_Bui_Thi_Xuan"},
    {"id": "662b58791afb9c00172d9107", "name": "225_Hoang_Van_Thu"},
    {"id": "66b1c4e7779f7400186741e4", "name": "Hoang_Van_Thu_Pham_Van_Hai"},
    {"id": "587c8183b807da0011e33d3f", "name": "Hoang_Van_Thu_Tran_Quoc_Hoan"},
    {"id": "586e25e1f9fab7001111b0ae", "name": "Truong_Chinh_Tan_Ky_Tan_Quy"},
    {"id": "56df807bc062921100c143da", "name": "Truong_Chinh_Au_Co"},
    {"id": "586e1f18f9fab7001111b0a5", "name": "Cong_Hoa_Truong_Chinh"},
    {"id": "5d8cdc57766c88001718896e", "name": "Ly_Thuong_Kiet_Lac_Long_Quan"},
    {"id": "66b1c4a2779f74001867418c", "name": "Le_Van_Si_Pham_Van_Hai"},
]

last_image_hashes = {}
hf_api = HfApi()

def send_alert(message: str):
    """Gửi cảnh báo trực tiếp qua Discord Webhook"""
    if not DISCORD_WEBHOOK_URL:
        logger.warning("Chưa cấu hình DISCORD_WEBHOOK_URL trong .env!")
        return

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=5)
        if response.status_code not in (200, 204):
            logger.error(f"Gửi Discord alert thất bại: Status {response.status_code}")
    except Exception as e:
        logger.error(f"Lỗi kết nối Discord Webhook: {e}")

def save_image_to_local(cam_id: str, content: bytes, now_str: str):
    """Lưu ảnh xuống thư mục Local nếu SAVE_LOCAL = True"""
    cam_dir = BASE_DIR / cam_id
    cam_dir.mkdir(parents=True, exist_ok=True)
    out_path = cam_dir / f"{now_str}.jpg"
    with open(out_path, "wb") as f:
        f.write(content)

def fetch_image(cam_id: str) -> bytes | None:
    url = f"{IMAGE_HANDLER_URL}{cam_id}&t={int(time.time() * 1000)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200 and r.content:
            content = r.content
            if len(content) < 5000 or not content.startswith(b"\xff\xd8"):
                return None

            img_hash = hashlib.md5(content).hexdigest()
            if last_image_hashes.get(cam_id) == img_hash:
                return None  # Bỏ qua nếu ảnh bị trùng / đơ
            
            last_image_hashes[cam_id] = img_hash
            return content
    except Exception as e:
        logger.debug(f"Fetch failed [{cam_id}]: {e}")
    return None

def upload_batch_to_hf(operations: list, commit_message: str) -> bool:
    """Commit toàn bộ danh sách file tích lũy lên Hugging Face trong 1 commit duy nhất"""
    if not HF_TOKEN or not HF_REPO_ID:
        logger.error("Thiếu HF_TOKEN hoặc HF_REPO_ID trong .env!")
        return False

    if not operations:
        return True

    try:
        hf_api.create_commit(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            operations=operations,
            commit_message=commit_message
        )
        return True
    except Exception as e:
        logger.error(f"Lỗi commit batch lên HF: {e}")
        return False

def main():
    if not HF_TOKEN or not HF_REPO_ID:
        logger.error("❌ Thiếu biến HF_TOKEN hoặc HF_REPO_ID trong file .env! Dừng chương trình.")
        return

    if SAVE_LOCAL:
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Chế độ lưu local: BẬT (Thư mục: {BASE_DIR})")
    else:
        logger.info("⚡ Chế độ lưu local: TẮT (Chỉ đẩy dữ liệu từ RAM lên Hugging Face)")

    logger.info(f"🚦 Khởi chạy crawler cho {len(CAMERAS)} camera (Upload lên HF mỗi {UPLOAD_EVERY_N_CYCLES} chu kỳ)...")
    send_alert(f"🚀 **Traffic Crawler Started** | Save Local: `{SAVE_LOCAL}` | Commit frequency: Every {UPLOAD_EVERY_N_CYCLES} cycles.")
    
    cycle = 1
    consecutive_failed_cycles = 0
    pending_operations = []  # Danh sách tích lũy các thao tác add file chờ commit

    while True:
        ts = datetime.now()
        start_time = time.time()
        failed_count = 0
        cycle_staged_count = 0

        now_str = ts.strftime("%Y%m%d_%H%M%S")
        logger.info(f"=== [Chu kỳ #{cycle} lúc {ts.strftime('%Y-%m-%d %H:%M:%S')}] ===")
        
        for cam in CAMERAS:
            cam_id = cam["id"]
            cam_name = cam["name"]

            img_bytes = fetch_image(cam_id)
            if img_bytes is None:
                failed_count += 1
                continue

            # 1. Lưu local nếu cấu hình BẬT
            if SAVE_LOCAL:
                save_image_to_local(cam_id, img_bytes, now_str)

            # 2. Tích lũy vào bộ nhớ đệm pending_operations
            path_in_repo = f"snapshots/{cam_id}/{now_str}.jpg"
            pending_operations.append(
                CommitOperationAdd(
                    path_in_repo=path_in_repo,
                    path_or_fileobj=img_bytes
                )
            )
            cycle_staged_count += 1
            logger.info(f"  [STAGED] [{cam_name}] -> {path_in_repo}")
            time.sleep(0.1)

        elapsed = time.time() - start_time
        logger.info(
            f"✅ Kết thúc quét chu kỳ #{cycle}: Thêm {cycle_staged_count} ảnh vào hàng chờ | "
            f"Hàng chờ HF hiện có: {len(pending_operations)} ảnh ({elapsed:.1f}s)"
        )

        # 3. Đạt số chu kỳ quy định (UPLOAD_EVERY_N_CYCLES) -> Thực hiện 1 commit duy nhất
        if cycle % UPLOAD_EVERY_N_CYCLES == 0 and pending_operations:
            commit_msg = f"Add batch of {len(pending_operations)} snapshots (Cycle #{cycle - UPLOAD_EVERY_N_CYCLES + 1} to #{cycle})"
            logger.info(f"🚀 [HF COMMIT] Đang đẩy {len(pending_operations)} ảnh tích lũy lên Hugging Face...")
            
            if upload_batch_to_hf(pending_operations, commit_msg):
                logger.info(f"🎉 [HF COMMIT SUCCESS] Đã commit thành công {len(pending_operations)} ảnh!")
                pending_operations.clear()  # Xóa hàng chờ sau khi commit thành công
            else:
                logger.error("❌ Commit thất bại! Các ảnh sẽ tiếp tục được giữ lại ở chu kỳ tiếp theo.")

        # Xử lý Cảnh Báo Discord
        if failed_count >= MAX_FAILED_PER_CYCLE:
            send_alert(
                f"⚠️ **CẢNH BÁO CÀO CAMERA FAILED**\n"
                f"• Chu kỳ: #{cycle}\n"
                f"• Thất bại: **{failed_count}/{len(CAMERAS)}** camera\n"
                f"• Thời gian: {ts.strftime('%H:%M:%S %d/%m/%Y')}"
            )

        if cycle_staged_count == 0:
            consecutive_failed_cycles += 1
            if consecutive_failed_cycles >= MAX_CONSECUTIVE_FAILS:
                send_alert(
                    f"🚨 **CRITICAL ALERT**\n"
                    f"Hệ thống mất kết nối **LIÊN TỤC {consecutive_failed_cycles} CHU KỲ**!\n"
                    f"Vui lòng kiểm tra lại mạng hoặc server camera."
                )
        else:
            consecutive_failed_cycles = 0

        cycle += 1
        sleep_wait = max(0.0, INTERVAL_SECONDS - elapsed)
        time.sleep(sleep_wait)

if __name__ == "__main__":
    main()