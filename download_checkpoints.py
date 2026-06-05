import os
import sys
import urllib.request
from tqdm import tqdm

CHECKPOINT_MAPPING = {
    "SuperCool-4x-Small.safetensors": "https://huggingface.co/andrewdalpino/SuperCool-4x-Small/resolve/main/model.safetensors",
    "SuperCool-4x-Medium.safetensors": "https://huggingface.co/andrewdalpino/SuperCool-4x-Medium/resolve/main/model.safetensors",
    "SuperCool-4x-Large.safetensors": "https://huggingface.co/andrewdalpino/SuperCool-4x-Large/resolve/main/model.safetensors"
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
    os.makedirs(target_dir, exist_ok=True)
    
    print("\n----------------------------------------------------")
    print("Initiating Asset Sync: Remote Model Matrix Download")
    print("----------------------------------------------------")
    
    for filename, download_url_target in CHECKPOINT_MAPPING.items():
        destination = os.path.join(target_dir, filename)
        
        if os.path.exists(destination):
            print(f" -> Found asset [Present]: {filename} (Skipping)")
            continue
            
        print(f" -> Fetching asset [Missing]: {filename}")
        try:
            download_url(download_url_target, destination)
        except Exception as e:
            print(f"ERROR: Failed downloading {filename} via {download_url_target}. Details: {e}")
            
    print("\nAsset synchronization complete.")

if __name__ == "__main__":
    main()