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


from pipeline.security import validate_safe_url, sanitize_filename, MAX_FILE_SIZE_BYTES


def download_media_from_url(url: str, output_dir: str) -> Tuple[bool, str, str]:
    """
    Download media from either a Google Drive link or direct HTTP/HTTPS link.
    Guarded by:
    - SSRF protection against loopback/private/metadata addresses.
    - Path traversal sanitization on output filenames.
    - Max streaming file size limits (500 MB).
    Returns: (success: bool, local_file_path: str, message: str)
    """
    os.makedirs(output_dir, exist_ok=True)
    clean_url = url.strip()

    # SSRF Guardrail check
    is_safe, reason = validate_safe_url(clean_url)
    if not is_safe:
        return False, "", f"Security Guardrail: {reason}"

    file_id = extract_gdrive_file_id(clean_url)
    
    # Handle Google Drive Link
    if "drive.google.com" in clean_url and file_id:
        try:
            import gdown
            safe_name = sanitize_filename(f"gdrive_{file_id}.mp4")
            out_path = os.path.join(output_dir, safe_name)
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
        raw_filename = os.path.basename(parsed.path) or "downloaded_media.mp4"
        if not re.search(r"\.(mp4|mkv|webm|avi|mp3|wav|m4a)$", raw_filename, re.IGNORECASE):
            raw_filename += ".mp4"
            
        safe_filename = sanitize_filename(raw_filename)
        out_path = os.path.join(output_dir, safe_filename)
        
        downloaded_bytes = 0
        with requests.get(clean_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            
            # Check Content-Length header if provided
            content_len = r.headers.get("Content-Length")
            if content_len and int(content_len) > MAX_FILE_SIZE_BYTES:
                return False, "", f"File size ({round(int(content_len)/(1024*1024), 1)} MB) exceeds guardrail limit of 500 MB."
                
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > MAX_FILE_SIZE_BYTES:
                            f.close()
                            if os.path.exists(out_path):
                                os.remove(out_path)
                            return False, "", "Download aborted: Stream exceeded maximum allowed limit of 500 MB."
                        f.write(chunk)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True, out_path, "Downloaded successfully"
        else:
            return False, "", "Downloaded file is empty"
    except Exception as e:
        return False, "", f"Direct download failed: {str(e)}"

