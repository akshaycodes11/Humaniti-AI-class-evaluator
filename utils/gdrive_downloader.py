import os
import re
import urllib.parse
import requests
from typing import Optional, Tuple


def extract_gdrive_file_id(url: str) -> Optional[str]:
    """Extract Google Drive file ID from various URL formats."""
    # Pattern 1: /file/d/<id>/
    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    
    # Pattern 2: id=<id>
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    
    # Pattern 3: /folders/<id>
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)

    return None


def download_media_from_url(url: str, output_dir: str) -> Tuple[bool, str, str]:
    """
    Download media from either a Google Drive link or direct HTTP/HTTPS link.
    Returns: (success: bool, local_file_path: str, message: str)
    """
    os.makedirs(output_dir, exist_ok=True)
    clean_url = url.strip()

    if not clean_url.startswith("http://") and not clean_url.startswith("https://"):
        return False, "", "Invalid URL scheme. Must start with http:// or https://"

    file_id = extract_gdrive_file_id(clean_url)
    
    # Handle Google Drive Link
    if "drive.google.com" in clean_url and file_id:
        try:
            import gdown
            out_path = os.path.join(output_dir, f"gdrive_{file_id}.mp4")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                return True, out_path, "Loaded from cache"

            # gdown download
            download_url = f"https://drive.google.com/uc?id={file_id}"
            output = gdown.download(download_url, out_path, quiet=False)
            if output and os.path.exists(output) and os.path.getsize(output) > 1024:
                return True, output, "Downloaded successfully from Google Drive"
            else:
                return False, "", "Could not download file from Google Drive. Ensure the file sharing is set to 'Anyone with the link'."
        except Exception as e:
            return False, "", f"Error downloading Google Drive file: {str(e)}"

    # Handle standard direct HTTP/HTTPS URL
    try:
        parsed = urllib.parse.urlparse(clean_url)
        filename = os.path.basename(parsed.path) or "downloaded_media.mp4"
        if not re.search(r"\.(mp4|mkv|webm|avi|mp3|wav|m4a)$", filename, re.IGNORECASE):
            filename += ".mp4"
            
        out_path = os.path.join(output_dir, filename)
        
        with requests.get(clean_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        f.write(chunk)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True, out_path, "Downloaded successfully"
        else:
            return False, "", "Downloaded file is empty"
    except Exception as e:
        return False, "", f"Direct download failed: {str(e)}"
