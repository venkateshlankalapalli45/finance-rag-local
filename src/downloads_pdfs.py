import os
import json
import requests
from tqdm import tqdm

# Define directories based on your exact workspace setup
JSONL_PATH = "data/financebench_document_information.jsonl"
OUTPUT_DIR = "data/raw_pdfs/"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_pdf(url, save_path):
    # SEC EDGAR and financial servers block scripts without a proper browser header
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=15)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            print(f"\n[!] Failed to download {url} - Status Code: {response.status_code}")
            return False
    except Exception as e:
        print(f"\n[!] Connection timed out or error for {url}: {e}")
        return False

def main():
    if not os.path.exists(JSONL_PATH):
        print(f"Error: Could not find catalog file at {JSONL_PATH}. Make sure you are running the script from your main 'nlp project' folder.")
        return

    print("Reading FinanceBench metadata catalog...")
    unique_docs = {}

    # Read the JSONL file line-by-line to extract names and links
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            doc_name = item.get("doc_name")
            doc_link = item.get("doc_link")
            if doc_name and doc_link:
                unique_docs[doc_name] = doc_link

    print(f"Found {len(unique_docs)} unique documents to download.\n")

    # Iterate through documents and display a progress bar
    for doc_name, url in tqdm(unique_docs.items(), desc="Downloading PDFs"):
        filename = doc_name if doc_name.lower().endswith(".pdf") else f"{doc_name}.pdf"
        save_path = os.path.join(OUTPUT_DIR, filename)

        # Skip if the file has already been downloaded
        if os.path.exists(save_path):
            continue

        download_pdf(url, save_path)

if __name__ == "__main__":
    main()