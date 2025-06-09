import os
import gdown

MODEL_DIR = os.path.join(os.path.dirname(__file__), "inference", "pytorch_states")
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    "audio_cnn.pt": "1NlXs5XlZRKN2UZsNx44C4l5mu7bTh6g0",
    "fusion_nlp_final.pt": "1ewRWwaif20abeBYpiM-tzyVwpjVE59VN",
    "xlm_roberta_emotion_augmented.pt": "1iuDH9d8aZbIrBIktgVFAk3t9-rd4MlVz"
}

def download_models():
    for filename, file_id in FILES.items():
        dest_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest_path):
            print(f"Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, dest_path, quiet=False)
        else:
            print(f"{filename} already exists. Skipping...")

if __name__ == "__main__":
    download_models()
