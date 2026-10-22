import os
import subprocess
import cv2
from typing import List, Tuple, Optional


def get_ffmpeg_binary() -> str:
    """Return path to ffmpeg binary via imageio_ffmpeg or system path."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def format_seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def extract_audio(video_or_audio_path: str, output_wav_path: str) -> bool:
    """
    Extract 16kHz mono WAV audio from any input video or audio file.
    Returns True if successful, False otherwise.
    """
    ffmpeg_bin = get_ffmpeg_binary()
    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
    
    cmd = [
        ffmpeg_bin,
        "-y",               # Overwrite output
        "-i", video_or_audio_path,
        "-vn",              # Disable video recording
        "-acodec", "pcm_s16le",
        "-ar", "16000",     # 16kHz
        "-ac", "1",         # Mono channel
        output_wav_path
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0
    except (subprocess.SubprocessError, FileNotFoundError, Exception) as e:
        print(f"Notice: ffmpeg audio extraction skipped or failed ({e}). Proceeding with visual and container analysis.")
        return False


def extract_audio_mp3(video_or_audio_path: str, output_mp3_path: str, bitrate: str = "64k") -> bool:
    """
    Extract lightweight MP3 audio (64kbps mono) suitable for low-latency API uploads.
    """
    ffmpeg_bin = get_ffmpeg_binary()
    os.makedirs(os.path.dirname(os.path.abspath(output_mp3_path)), exist_ok=True)
    
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_or_audio_path,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", bitrate,
        output_mp3_path
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return os.path.exists(output_mp3_path) and os.path.getsize(output_mp3_path) > 0
    except Exception as e:
        print(f"Error extracting MP3 audio: {e}")
        return False


def extract_representative_keyframes(
    video_path: str,
    output_dir: str,
    max_frames: int = 10,
    target_width: int = 720
) -> List[Tuple[str, str]]:
    """
    Extract up to max_frames keyframes spaced across the recording.
    Returns list of tuples: [(timestamp_str, image_file_path), ...]
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[Tuple[str, str]] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return results

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    if total_frames <= 0 or duration_sec <= 0:
        cap.release()
        return results

    # Choose evenly spaced sampling points, skipping the first and last 2% to avoid splash screens
    start_sec = max(5.0, duration_sec * 0.02)
    end_sec = max(start_sec + 5.0, duration_sec * 0.98)
    
    if max_frames == 1:
        sample_times = [(start_sec + end_sec) / 2.0]
    else:
        step = (end_sec - start_sec) / float(max_frames - 1)
        sample_times = [start_sec + i * step for i in range(max_frames)]

    for idx, t_sec in enumerate(sample_times):
        frame_idx = int(t_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total_frames - 1))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Resize preserving aspect ratio
        h, w = frame.shape[:2]
        if w > target_width:
            target_height = int((target_width / float(w)) * h)
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        timestamp_str = format_seconds_to_timestamp(t_sec)
        clean_time_name = timestamp_str.replace(":", "_")
        out_path = os.path.join(output_dir, f"frame_{idx:02d}_{clean_time_name}.jpg")
        
        # Save as optimized JPEG
        cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        results.append((timestamp_str, out_path))

    cap.release()
    return results


import re


def probe_media_duration_ffmpeg(file_path: str) -> float:
    """Use FFmpeg container probe to determine duration for any video or audio file."""
    if not os.path.exists(file_path):
        return 0.0
    try:
        ffmpeg_bin = get_ffmpeg_binary()
        cmd = [ffmpeg_bin, "-i", file_path]
        res = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, timeout=10)
        # Look for Duration: 00:37:53.69
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", res.stderr)
        if m:
            h, mn, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
            return round(h * 3600.0 + mn * 60.0 + s, 1)
    except Exception:
        pass
    return 0.0


def check_is_vtt_file(file_path: str) -> bool:
    """Check if file starts with WEBVTT or contains WebVTT cue header."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) > 10 * 1024 * 1024:
        return False
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(512)
        return b"WEBVTT" in chunk or b"-->" in chunk
    except Exception:
        return False


def get_media_metadata(file_path: str) -> dict:
    """Return video/audio basic properties: duration, has_video, resolution."""
    metadata = {
        "file_path": file_path,
        "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2) if os.path.exists(file_path) else 0.0,
        "has_video": False,
        "is_transcript": False,
        "duration_seconds": 0.0,
        "width": 0,
        "height": 0
    }
    
    if not os.path.exists(file_path):
        return metadata

    # 1. Check if the file is actually a WebVTT transcript (even if misnamed as .mp4)
    if check_is_vtt_file(file_path):
        metadata["is_transcript"] = True
        try:
            from pipeline.vtt_parser import parse_vtt_file
            vtt_info = parse_vtt_file(file_path)
            metadata["duration_seconds"] = vtt_info.get("duration_seconds", 0.0)
        except Exception:
            pass
        return metadata

    # 2. Try OpenCV for video files
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total_frames > 0 and fps > 0 and width > 0:
            metadata["has_video"] = True
            metadata["duration_seconds"] = round(total_frames / fps, 1)
            metadata["width"] = width
            metadata["height"] = height
        elif width > 0 and height > 0:
            metadata["has_video"] = True
            metadata["width"] = width
            metadata["height"] = height
        cap.release()

    # 3. If OpenCV returned 0.0 duration, probe with FFmpeg (supports pure audio, streaming WebM/MP4, VFR)
    if metadata["duration_seconds"] <= 0.0:
        ff_dur = probe_media_duration_ffmpeg(file_path)
        if ff_dur > 0.0:
            metadata["duration_seconds"] = ff_dur
            
    # 4. If still 0 and file is a WAV audio file, probe sample rate / length directly
    if metadata["duration_seconds"] <= 0.0 and file_path.lower().endswith(".wav"):
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(file_path)
            if sr > 0:
                metadata["duration_seconds"] = round(len(data) / float(sr), 1)
        except Exception:
            pass

    return metadata
