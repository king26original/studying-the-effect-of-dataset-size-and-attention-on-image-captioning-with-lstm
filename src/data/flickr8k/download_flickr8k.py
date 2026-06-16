#this cell is ai generated
#its use is to download the required dataset
import os
import urllib.request
import zipfile

def download_and_extract(url, download_path, extract_path):
    os.makedirs(download_path, exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)

    filename = url.split("/")[-1]
    zip_path = os.path.join(download_path, filename)

    if not os.path.exists(zip_path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print(f"{filename} already exists.")

    print(f"Extracting {filename}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.\n")


# URLs
images_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
captions_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

# Paths
download_dir = "downloads"
dataset_dir = "flickr8k"

# Download & extract
def download():
    download_and_extract(images_url, download_dir, dataset_dir)
    download_and_extract(captions_url, download_dir, dataset_dir)