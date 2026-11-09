import zipfile
from pathlib import Path

def emergency_dump_to_zips(pending_ops: list, chunk_size: int = 5000):
    """Xả 90,000 ảnh từ RAM ra các file ZIP nhỏ (5,000 ảnh/file) để giải phóng RAM tức thì."""
    total = len(pending_ops)
    if total == 0:
        logger.info("Hàng chờ trống.")
        return

    logger.warning(f"🚨 ĐANG XẢ KHẨN CẤP {total} ẢNH TỪ RAM RA Ổ ĐĨA...")
    
    # Chia nhỏ 90k ảnh thành các file zip 5,000 ảnh
    part = 1
    for i in range(0, total, chunk_size):
        chunk = pending_ops[i:i + chunk_size]
        zip_name = f"snapshot_dump_part_{part}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        logger.info(f"📦 Đang nén Part {part}: {len(chunk)} ảnh vào {zip_name}...")
        with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for op in chunk:
                zf.writestr(op.path_in_repo, op.path_or_fileobj)
        
        logger.info(f"✅ Đã lưu xong {zip_name} ({i + len(chunk)}/{total})")
        part += 1

    # Xóa hàng chờ giải phóng RAM ngay lập tức
    pending_ops.clear()
    logger.info("🎉 ĐÃ GIẢI PHỎNG RAM THÀNH CÔNG! Hàng chờ hiện tại đã về 0.")