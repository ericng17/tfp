import os
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login

# =========================================================
# 0. KHAI BÁO HUGGING FACE TOKEN
# =========================================================
# Cách 1: Dán trực tiếp Token của bạn vào đây (Tạo tại: https://huggingface.co/settings/tokens - chọn quyền WRITE)
HF_TOKEN = "xxx"

# Cách 2 (Nếu chạy trên Kaggle Secrets để bảo mật):
# from kaggle_secrets import UserSecretsClient
# HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")

# Cách 3 (Nếu dùng biến môi trường):
# HF_TOKEN = os.getenv("HF_TOKEN")

# Thực hiện đăng nhập hệ thống
login(token=HF_TOKEN)

# =========================================================
# 1. KHAI BÁO REPOSITORY & NẠP DỮ LIỆU
# =========================================================
REPO_PART1 = "ericng17/hcm_traffic_snapshots"
REPO_PART2 = "ericng17/hcm_traffic_snapshots2"
MERGED_REPO = "ericng17/hcm_traffic"

print("Đang nạp dữ liệu từ Hugging Face...")
# Truyền token vào load_dataset phòng trường hợp repo ở chế độ Private
ds1 = load_dataset(REPO_PART1, split="train", token=HF_TOKEN)
ds2 = load_dataset(REPO_PART2, split="train", token=HF_TOKEN)

print(f"Dung lượng tập 1: {len(ds1)} mẫu")
print(f"Dung lượng tập 2: {len(ds2)} mẫu")

# =========================================================
# 2. GỘP DỮ LIỆU
# =========================================================
merged_ds = concatenate_datasets([ds1, ds2])
print(f"Tổng số mẫu sau khi gộp: {len(merged_ds)} mẫu")

# =========================================================
# 3. ĐẨY DATASET LÊN HUGGING FACE HUB
# =========================================================
print("Đang đẩy dataset đã gộp lên Hugging Face Hub...")
# Truyền token trực tiếp vào push_to_hub để ghi dữ liệu thành công
merged_ds.push_to_hub(MERGED_REPO, private=True, token=HF_TOKEN)

print("Gộp và đẩy dữ liệu lên Hugging Face thành công!")