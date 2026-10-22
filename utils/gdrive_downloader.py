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


def normalize_downloaded_file_extension(file_path: str) -> str:
    """Inspect downloaded file header and correct the extension (e.g. rename .mp4 to .vtt if text transcript)."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return file_path
    
    try:
        with open(file_path, "rb") as f:
            header = f.read(512)
        
        base, ext = os.path.splitext(file_path)
        correct_path = file_path
        
        if b"WEBVTT" in header or (b"-->" in header and b"00:" in header):
            if ext.lower() != ".vtt":
                correct_path = f"{base}.vtt"
        elif header.startswith(b"RIFF") and b"WAVE" in header[:16]:
            if ext.lower() != ".wav":
                correct_path = f"{base}.wav"
        elif header.startswith(b"ID3") or (len(header) > 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
            if ext.lower() != ".mp3":
                correct_path = f"{base}.mp3"
                
        if correct_path != file_path:
            if os.path.exists(correct_path):
                try:
                    os.remove(correct_path)
                except Exception:
                    pass
            os.rename(file_path, correct_path)
            return correct_path
    except Exception:
        pass
        
    return file_path


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
            # Check if already cached as .vtt or .mp4
            cached_vtt = os.path.join(output_dir, sanitize_filename(f"gdrive_{file_id}.vtt"))
            if os.path.exists(cached_vtt) and os.path.getsize(cached_vtt) > 100:
                return True, cached_vtt, "Loaded from cache"

            cached_mp4 = os.path.join(output_dir, sanitize_filename(f"gdrive_{file_id}.mp4"))
            if os.path.exists(cached_mp4) and os.path.getsize(cached_mp4) > 100:
                normalized = normalize_downloaded_file_extension(cached_mp4)
                return True, normalized, "Loaded from cache"

            safe_name = sanitize_filename(f"gdrive_{file_id}.mp4")
            out_path = os.path.join(output_dir, safe_name)

            # gdown download
            download_url = f"https://drive.google.com/uc?id={file_id}"
            output = gdown.download(download_url, out_path, quiet=False)
            if output and os.path.exists(output) and os.path.getsize(output) > 100:
                final_path = normalize_downloaded_file_extension(output)
                return True, final_path, "Downloaded successfully from Google Drive"
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
            final_path = normalize_downloaded_file_extension(out_path)
            return True, final_path, "Downloaded successfully"
        else:
            return False, "", "Downloaded file is empty"
    except Exception as e:
        return False, "", f"Direct download failed: {str(e)}"

