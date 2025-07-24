import sys
import os
import requests

output_dir = "/scratch/acisse/unzipped"
error_log = "/home/tau/acisse/mixing_secrets/mixing_secrets/error_downloading.txt"

def download_file(url):
    try:
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.basename(url)
        if not filename.endswith('.zip'):
            filename += ".zip"
        file_path = os.path.join(output_dir, filename)

        if os.path.exists(file_path):
            print(f"Skipping {file_path}, already exists.")
            return

        print(f"Downloading {url} -> {file_path}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded {file_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        with open(error_log, "a") as err_file:
            err_file.write(url + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download_one.py <url>")
        sys.exit(1)

    download_file(sys.argv[1])
