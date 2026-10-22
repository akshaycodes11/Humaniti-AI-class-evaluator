import os
import re
import socket
import urllib.parse
import ipaddress
import html
import time
from typing import Tuple, Dict, Any, List


# Whitelist of permissible media extensions
ALLOWED_MEDIA_EXTENSIONS = {"mp4", "mkv", "webm", "avi", "mov", "wav", "mp3", "m4a", "vtt", "txt"}
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB max to prevent denial-of-service / disk flooding


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent Path Traversal attacks (e.g. ../../etc/passwd or absolute paths).
    Strips directory separators and non-standard characters.
    """
    clean = os.path.basename(filename)
    clean = re.sub(r"[^a-zA-Z0-9_.\-\s]", "_", clean)
    clean = clean.strip()
    return clean if clean else "safe_media_upload.mp4"


def validate_safe_url(url: str) -> Tuple[bool, str]:
    """
    Server-Side Request Forgery (SSRF) Guardrail.
    Validates that a URL is safe to fetch:
    - Scheme must be strictly http or https.
    - Resolves DNS to check that destination IP is not private, loopback, link-local, or cloud metadata.
    """
    clean_url = url.strip()
    if not clean_url:
        return False, "URL cannot be empty."

    try:
        parsed = urllib.parse.urlparse(clean_url)
    except Exception as e:
        return False, f"Malformed URL syntax: {e}"

    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Forbidden URL scheme '{parsed.scheme}'. Only 'http://' and 'https://' are allowed."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL is missing a valid hostname."

    # Block explicit localhost and cloud metadata domain names
    blocked_hostnames = {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "metadata.google.internal",
        "metadata.internal",
        "instance-data"
    }
    if hostname.lower() in blocked_hostnames:
        return False, f"Access to internal endpoint '{hostname}' is blocked by SSRF guardrail."

    # Resolve IP and verify against private/reserved ranges
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip_str)

            if ip_obj.is_private:
                return False, f"Blocked connection to private IP range ({ip_str}) - SSRF Guardrail."
            if ip_obj.is_loopback:
                return False, f"Blocked connection to loopback IP ({ip_str}) - SSRF Guardrail."
            if ip_obj.is_link_local:
                return False, f"Blocked connection to link-local IP ({ip_str}) - SSRF Guardrail."
            if ip_obj.is_reserved:
                return False, f"Blocked connection to reserved IP ({ip_str}) - SSRF Guardrail."
            if ip_str == "169.254.169.254":
                return False, "Blocked connection to cloud instance metadata endpoint - SSRF Guardrail."
    except socket.gaierror:
        # If hostname cannot be resolved, allow gdown to handle drive links or flag error
        if "drive.google.com" not in clean_url:
            return False, f"Could not resolve hostname '{hostname}'."
    except Exception as e:
        return False, f"DNS resolution check error: {e}"

    return True, "URL verified safe."


def scrub_pii(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    COPPA / GDPR-K Child Privacy & PII Guardrail.
    Detects and redacts sensitive Personally Identifiable Information (PII) from transcripts
    and user input before sending to LLM APIs or embedding in exportable reports.
    """
    if not text:
        return "", {"redacted_count": 0, "types_found": []}

    redacted_count = 0
    types_found = set()
    cleaned = text

    # 1. Email addresses
    email_regex = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    emails = email_regex.findall(cleaned)
    if emails:
        redacted_count += len(emails)
        types_found.add("Email Address")
        cleaned = email_regex.sub("[EMAIL_REDACTED]", cleaned)

    # 2. Phone numbers (international, US, Indian 10-digit formats)
    phone_regex = re.compile(
        r"(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4,5})"
    )
    # Match candidate phone numbers with 9+ digits total
    def replace_phone(match):
        raw = match.group(0)
        digits = re.sub(r"\D", "", raw)
        if 9 <= len(digits) <= 15:
            nonlocal redacted_count
            redacted_count += 1
            types_found.add("Phone Number")
            return "[PHONE_REDACTED]"
        return raw

    cleaned = phone_regex.sub(replace_phone, cleaned)

    # 3. IPv4 addresses
    ip_regex = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
    ips = ip_regex.findall(cleaned)
    if ips:
        # Avoid redacting chess notation or timestamps like 1.2
        real_ips = [ip for ip in ips if all(0 <= int(octet) <= 255 for octet in ip.split("."))]
        if real_ips:
            redacted_count += len(real_ips)
            types_found.add("IP Address")
            cleaned = ip_regex.sub("[IP_REDACTED]", cleaned)

    return cleaned, {
        "redacted_count": redacted_count,
        "types_found": list(types_found),
        "is_scrubbed": redacted_count > 0
    }


def sanitize_prompt_injection(user_text: str) -> str:
    """
    LLM Security Guardrail:
    Neutralizes prompt injection patterns attempting to manipulate the grading rubric or hijack system instructions.
    """
    if not user_text:
        return ""

    injection_patterns = [
        r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
        r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\b",
        r"(?i)\byou\s+are\s+now\b",
        r"(?i)\bsystem\s+prompt\b",
        r"(?i)\balways\s+rate\s+(as\s+)?(100|excellent)\b",
        r"(?i)\b(bypass|override)\s+the\s+rubric\b",
        r"(?i)\bdo\s+not\s+penalize\b"
    ]


    sanitized = user_text
    for pattern in injection_patterns:
        sanitized = re.sub(pattern, "[PROMPT_INJECTION_FLAGGED_AND_REMOVED]", sanitized)

    return sanitized


def purge_ephemeral_storage(directory_path: str, max_age_seconds: float = 3600.0) -> int:
    """
    Zero-Retention Privacy Guardrail.
    Purges temporary cached videos, audio WAVs, and keyframes older than max_age_seconds
    to prevent retention of identifiable child recordings.
    Returns count of files removed.
    """
    if not os.path.exists(directory_path):
        return 0

    removed = 0
    now = time.time()

    for root, _, files in os.walk(directory_path):
        for f in files:
            file_path = os.path.join(root, f)
            try:
                # If older than max age, or if an on-demand flush
                if now - os.path.getmtime(file_path) > max_age_seconds:
                    os.remove(file_path)
                    removed += 1
            except Exception:
                pass

    return removed
