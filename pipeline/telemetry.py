import os
import math
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.io import wavfile
import cv2


def analyze_audio_acoustics(wav_path: str) -> Dict[str, Any]:
    """
    Deterministically analyze audio for:
    - Total duration
    - Silence ratio (percentage of dead air / pauses)
    - Dead air events (pauses > 7 seconds)
    - Speech energy distribution
    """
    if not os.path.exists(wav_path):
        return {
            "duration_seconds": 0.0,
            "silence_ratio_pct": 0.0,
            "dead_air_incidents": 0,
            "longest_silence_seconds": 0.0,
            "active_speech_seconds": 0.0
        }
    
    try:
        sample_rate, data = wavfile.read(wav_path)
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        # Normalize to float between -1.0 and 1.0
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            audio = (data.astype(np.float32) - 128.0) / 128.0
        else:
            audio = data.astype(np.float32)

        total_samples = len(audio)
        duration_sec = total_samples / float(sample_rate) if sample_rate > 0 else 0.0
        if duration_sec <= 0:
            return {"duration_seconds": 0.0, "silence_ratio_pct": 0.0, "dead_air_incidents": 0}

        # Frame-level RMS energy calculation (window = 200ms)
        frame_size = int(sample_rate * 0.20)  # 200ms
        if frame_size <= 0:
            frame_size = 1000

        num_frames = total_samples // frame_size
        if num_frames == 0:
            return {"duration_seconds": duration_sec, "silence_ratio_pct": 0.0, "dead_air_incidents": 0}

        frames = audio[:num_frames * frame_size].reshape((num_frames, frame_size))
        rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

        # Dynamic silence threshold based on noise floor
        # Use 20th percentile + offset or absolute floor
        p20 = np.percentile(rms_per_frame, 20)
        p80 = np.percentile(rms_per_frame, 80)
        silence_threshold = max(p20 * 1.5, 0.008)

        is_silent = rms_per_frame < silence_threshold
        silent_frames_count = int(np.sum(is_silent))
        silence_ratio_pct = round((silent_frames_count / float(num_frames)) * 100.0, 1)

        # Detect consecutive silent intervals (> 7 seconds)
        consecutive_silent_frames = 0
        min_dead_air_frames = int(7.0 / 0.20)  # 7 seconds / 0.2s = 35 frames
        dead_air_count = 0
        longest_silence_frames = 0
        current_silence_run = 0

        dead_air_intervals: List[Tuple[float, float]] = []
        silence_start_frame = None

        for idx, silent in enumerate(is_silent):
            if silent:
                if current_silence_run == 0:
                    silence_start_frame = idx
                current_silence_run += 1
            else:
                if current_silence_run > longest_silence_frames:
                    longest_silence_frames = current_silence_run
                if current_silence_run >= min_dead_air_frames and silence_start_frame is not None:
                    dead_air_count += 1
                    start_s = round(silence_start_frame * 0.20, 1)
                    end_s = round(idx * 0.20, 1)
                    dead_air_intervals.append((start_s, end_s))
                current_silence_run = 0

        # Check end of file
        if current_silence_run >= min_dead_air_frames and silence_start_frame is not None:
            dead_air_count += 1
            dead_air_intervals.append((round(silence_start_frame * 0.20, 1), round(duration_sec, 1)))

        longest_silence_sec = round(longest_silence_frames * 0.20, 1)
        active_speech_sec = round(duration_sec * (1.0 - silence_ratio_pct / 100.0), 1)

        return {
            "duration_seconds": round(duration_sec, 1),
            "silence_ratio_pct": silence_ratio_pct,
            "dead_air_incidents": dead_air_count,
            "longest_silence_seconds": longest_silence_sec,
            "active_speech_seconds": active_speech_sec,
            "dead_air_intervals": dead_air_intervals[:10]  # First 10 intervals
        }
    except Exception as e:
        return {
            "duration_seconds": 0.0,
            "silence_ratio_pct": 0.0,
            "dead_air_incidents": 0,
            "error": str(e)
        }


def analyze_visual_pacing(video_path: str, sample_interval_sec: float = 30.0) -> Dict[str, Any]:
    """
    Extract visual dynamics from video using OpenCV:
    - Counts visual screen/slide changes per minute
    - Detects static/frozen screen periods
    """
    if not os.path.exists(video_path):
        return {"visual_change_rate": 0.0, "screen_state": "unknown"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"visual_change_rate": 0.0, "screen_state": "cannot_open"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    if duration_sec <= 0:
        cap.release()
        return {"visual_change_rate": 0.0, "duration_sec": 0.0}

    step_frames = int(fps * sample_interval_sec)
    prev_gray = None
    slide_changes = 0
    sampled_count = 0

    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Resize for fast computation
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Absolute difference
            diff = cv2.absdiff(prev_gray, gray)
            mean_diff = np.mean(diff)
            # A mean difference above 25 in 0-255 indicates significant slide/screen change
            if mean_diff > 25.0:
                slide_changes += 1

        prev_gray = gray
        sampled_count += 1
        frame_idx += step_frames

    cap.release()

    duration_minutes = duration_sec / 60.0
    change_rate = round(slide_changes / duration_minutes, 2) if duration_minutes > 0 else 0.0

    return {
        "visual_change_rate": change_rate,
        "total_slide_transitions_detected": slide_changes,
        "duration_sec": round(duration_sec, 1)
    }
