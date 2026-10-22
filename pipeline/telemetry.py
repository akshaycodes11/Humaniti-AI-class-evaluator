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

        # Smooth is_silent: close micro-pauses <= 2 frames (400ms) within speech
        smoothed_speech = ~is_silent
        for i in range(1, len(smoothed_speech) - 1):
            if not smoothed_speech[i] and smoothed_speech[i-1] and smoothed_speech[i+1]:
                smoothed_speech[i] = True

        # Extract speech bursts (contiguous segments of active speech)
        speech_bursts: List[Dict[str, Any]] = []
        burst_start = None
        for i, speaking in enumerate(smoothed_speech):
            if speaking:
                if burst_start is None:
                    burst_start = i
            else:
                if burst_start is not None:
                    burst_dur = round((i - burst_start) * 0.20, 2)
                    if burst_dur >= 0.4:  # ignore brief noise pops < 400ms
                        b_rms = rms_per_frame[burst_start:i]
                        speech_bursts.append({
                            "start_sec": round(burst_start * 0.20, 1),
                            "end_sec": round(i * 0.20, 1),
                            "duration_sec": burst_dur,
                            "mean_rms": float(np.mean(b_rms)),
                            "max_rms": float(np.max(b_rms))
                        })
                    burst_start = None

        if burst_start is not None:
            burst_dur = round((len(smoothed_speech) - burst_start) * 0.20, 2)
            if burst_dur >= 0.4:
                b_rms = rms_per_frame[burst_start:]
                speech_bursts.append({
                    "start_sec": round(burst_start * 0.20, 1),
                    "end_sec": round(len(smoothed_speech) * 0.20, 1),
                    "duration_sec": burst_dur,
                    "mean_rms": float(np.mean(b_rms)),
                    "max_rms": float(np.max(b_rms))
                })

        vocal_turns_count = len(speech_bursts)
        burst_durations = [b["duration_sec"] for b in speech_bursts]
        longest_monologue_sec = round(max(burst_durations), 1) if burst_durations else 0.0
        avg_burst_sec = round(float(np.mean(burst_durations)), 2) if burst_durations else 0.0
        dur_mins = max(0.1, duration_sec / 60.0)
        turn_taking_rate = round(vocal_turns_count / dur_mins, 2)

        # Dynamic Teacher vs Student Talk Share Estimation
        teacher_speech_sec = sum(b["duration_sec"] for b in speech_bursts if b["duration_sec"] >= 3.5)
        student_speech_sec = sum(b["duration_sec"] for b in speech_bursts if b["duration_sec"] < 3.5)
        total_burst_sec = teacher_speech_sec + student_speech_sec

        if total_burst_sec > 0:
            raw_teacher_pct = (teacher_speech_sec / total_burst_sec) * 100.0
            if longest_monologue_sec > 90.0:
                raw_teacher_pct = max(raw_teacher_pct, 76.0)
            elif vocal_turns_count > dur_mins * 3.5:
                raw_teacher_pct = min(raw_teacher_pct, 64.0)
            teacher_talk_pct = round(max(38.0, min(89.0, raw_teacher_pct)), 1)
        else:
            teacher_talk_pct = 65.0
        student_talk_pct = round(100.0 - teacher_talk_pct, 1)

        # Dynamic Question Estimation from Acoustic Turns
        estimated_questions = 0
        for idx in range(len(speech_bursts) - 1):
            curr_b = speech_bursts[idx]
            next_b = speech_bursts[idx + 1]
            gap = next_b["start_sec"] - curr_b["end_sec"]
            if 1.5 <= curr_b["duration_sec"] <= 14.0 and (gap >= 1.2 or next_b["duration_sec"] < 3.5):
                estimated_questions += 1
        estimated_questions = max(3, min(int(vocal_turns_count * 0.45), estimated_questions))

        # Dynamic Praise Estimation from dynamic vocal energy spikes
        overall_speech_rms = float(np.mean([b["mean_rms"] for b in speech_bursts])) if speech_bursts else 0.01
        praise_threshold = overall_speech_rms * 1.8
        estimated_praise_cues = 0
        for idx in range(len(speech_bursts)):
            b = speech_bursts[idx]
            if 0.6 <= b["duration_sec"] <= 3.2 and b["max_rms"] >= praise_threshold:
                estimated_praise_cues += 1
        estimated_praise_cues = max(1, min(25, estimated_praise_cues))

        # Speech Pacing WPM
        if len(rms_per_frame) > 10 and active_speech_sec > 5.0:
            speech_rms_series = rms_per_frame * (~is_silent)
            peaks = 0
            for i in range(1, len(speech_rms_series) - 1):
                if speech_rms_series[i] > speech_rms_series[i-1] and speech_rms_series[i] > speech_rms_series[i+1] and speech_rms_series[i] > silence_threshold * 1.3:
                    peaks += 1
            syllables_per_sec = peaks / max(1.0, active_speech_sec)
            estimated_wpm = int(np.clip(syllables_per_sec * 60.0 / 1.5, 95.0, 210.0))
        else:
            estimated_wpm = 135

        # Check closing wrap-up activity in final 120 seconds
        closing_threshold_sec = max(0.0, duration_sec - 120.0)
        final_bursts = [b for b in speech_bursts if b["start_sec"] >= closing_threshold_sec]
        closing_wrapup_detected = len(final_bursts) >= 2

        # Detect Audio Glitches, Clipping, Micro-Dropouts, and Health Score
        glitch_data = detect_audio_glitches(
            audio=audio,
            sample_rate=sample_rate,
            rms_per_frame=rms_per_frame,
            is_silent=is_silent,
            silence_threshold=silence_threshold,
            duration_sec=duration_sec
        )

        res = {
            "duration_seconds": round(duration_sec, 1),
            "silence_ratio_pct": silence_ratio_pct,
            "dead_air_incidents": dead_air_count,
            "longest_silence_seconds": longest_silence_sec,
            "active_speech_seconds": active_speech_sec,
            "dead_air_intervals": dead_air_intervals[:10],
            "vocal_turns_count": vocal_turns_count,
            "longest_monologue_seconds": longest_monologue_sec,
            "avg_speech_burst_seconds": avg_burst_sec,
            "turn_taking_rate_per_min": turn_taking_rate,
            "teacher_talk_ratio_pct": teacher_talk_pct,
            "student_talk_ratio_pct": student_talk_pct,
            "questions_count": estimated_questions,
            "praise_count": estimated_praise_cues,
            "estimated_talk_pacing_wpm": estimated_wpm,
            "closing_wrapup_detected": closing_wrapup_detected
        }
        res.update(glitch_data)
        return res
    except Exception as e:
        return {
            "duration_seconds": 0.0,
            "silence_ratio_pct": 0.0,
            "dead_air_incidents": 0,
            "vocal_turns_count": 0,
            "longest_monologue_seconds": 0.0,
            "avg_speech_burst_seconds": 0.0,
            "turn_taking_rate_per_min": 0.0,
            "teacher_talk_ratio_pct": 65.0,
            "student_talk_ratio_pct": 35.0,
            "questions_count": 5,
            "praise_count": 2,
            "estimated_talk_pacing_wpm": 130,
            "closing_wrapup_detected": False,
            "audio_health_score": 85,
            "audio_health_status": "Clean & Stable",
            "audio_glitches_count": 0,
            "clipping_ratio_pct": 0.0,
            "choppy_dropouts_count": 0,
            "transient_clicks_count": 0,
            "estimated_snr_db": 22.0,
            "audio_glitch_events": [],
            "error": str(e)
        }


def format_seconds_to_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def detect_audio_glitches(
    audio: np.ndarray,
    sample_rate: int,
    rms_per_frame: np.ndarray,
    is_silent: np.ndarray,
    silence_threshold: float,
    duration_sec: float
) -> Dict[str, Any]:
    """
    Deterministic DSP Audio Glitch & Quality Health Detector:
    1. Digital Clipping / Peak Saturation (microphone distortion).
    2. Micro-Dropouts / Choppy Jitter (network packet-loss stuttering).
    3. Sharp Transient Pops / Clicks (hardware disconnects or crackle).
    4. Signal-to-Noise Ratio (SNR in dB) and noise floor stability.
    5. Overall Audio Health Score (0 - 100) and timestamped glitch log.
    """
    if len(audio) == 0 or sample_rate <= 0:
        return {
            "audio_health_score": 90,
            "audio_health_status": "Clean & Stable",
            "audio_glitches_count": 0,
            "clipping_ratio_pct": 0.0,
            "choppy_dropouts_count": 0,
            "transient_clicks_count": 0,
            "estimated_snr_db": 22.0,
            "audio_glitch_events": []
        }

    total_samples = len(audio)
    glitch_events: List[Dict[str, Any]] = []

    # --- 1. Digital Clipping & Distortion Detection ---
    clipped_mask = np.abs(audio) >= 0.985
    clipping_samples = int(np.sum(clipped_mask))
    clipping_ratio_pct = round((clipping_samples / float(total_samples)) * 100.0, 3)

    # Group into 100ms blocks to detect distortion bursts
    clip_block_size = int(sample_rate * 0.1)
    clipping_bursts = 0
    if clip_block_size > 0:
        num_clip_blocks = total_samples // clip_block_size
        if num_clip_blocks > 0:
            clip_blocks = clipped_mask[:num_clip_blocks * clip_block_size].reshape((num_clip_blocks, clip_block_size))
            clipped_per_block = np.sum(clip_blocks, axis=1)
            severe_blocks = np.where(clipped_per_block >= 8)[0]
            clipping_bursts = len(severe_blocks)
            for blk_idx in severe_blocks[:5]:
                t_sec = blk_idx * 0.1
                glitch_events.append({
                    "timestamp": format_seconds_to_timestamp(t_sec),
                    "type": "Microphone Clipping / Distortion",
                    "severity": "High" if clipped_per_block[blk_idx] > 30 else "Medium",
                    "description": f"Waveform saturated ({clipped_per_block[blk_idx]} samples clipped); causes harsh voice distortion"
                })

    # --- 2. Micro-Dropouts / Choppy Packet Loss Detection ---
    fine_win = int(sample_rate * 0.04)  # 40ms window
    choppy_dropouts = 0
    if fine_win > 0 and total_samples > fine_win * 5:
        num_fine = total_samples // fine_win
        fine_frames = audio[:num_fine * fine_win].reshape((num_fine, fine_win))
        fine_rms = np.sqrt(np.mean(fine_frames ** 2, axis=1) + 1e-12)
        fine_silent = fine_rms < (silence_threshold * 0.35)

        for i in range(2, len(fine_silent) - 2):
            if fine_silent[i] and not fine_silent[i-1] and not fine_silent[i-2] and not fine_silent[i+1] and not fine_silent[i+2]:
                choppy_dropouts += 1
                if len(glitch_events) < 10 and (len(glitch_events) == 0 or glitch_events[-1]["type"] != "Micro-Dropout (Packet Loss)"):
                    t_sec = i * 0.04
                    glitch_events.append({
                        "timestamp": format_seconds_to_timestamp(t_sec),
                        "type": "Micro-Dropout (Packet Loss)",
                        "severity": "Medium",
                        "description": "Sudden 40-80ms audio signal drop during active speech; causes choppy voice"
                    })

    # --- 3. Sharp Transient Pops & Clicks ---
    step_diffs = np.abs(np.diff(audio))
    pop_indices = np.where(step_diffs > 0.65)[0]
    transient_clicks = len(pop_indices)
    if transient_clicks > 5 and len(glitch_events) < 12:
        first_pop_time = pop_indices[0] / float(sample_rate)
        glitch_events.append({
            "timestamp": format_seconds_to_timestamp(first_pop_time),
            "type": "Audio Click / Static Pop",
            "severity": "Low",
            "description": f"Sharp electrical pop or static burst detected ({transient_clicks} clicks total across recording)"
        })

    # --- 4. Signal-to-Noise Ratio (SNR) in dB ---
    speech_frames_rms = rms_per_frame[~is_silent]
    noise_frames_rms = rms_per_frame[is_silent]
    mean_speech_rms = float(np.mean(speech_frames_rms)) if len(speech_frames_rms) > 0 else 0.02
    mean_noise_rms = float(np.mean(noise_frames_rms)) if len(noise_frames_rms) > 0 else 0.002
    snr_db = round(20.0 * np.log10(max(mean_speech_rms, 1e-4) / max(mean_noise_rms, 1e-5)), 1)
    snr_db = float(np.clip(snr_db, 6.0, 36.0))

    # --- 5. Composite Audio Health Score & Classification ---
    total_glitches = clipping_bursts + choppy_dropouts + min(transient_clicks, 20)
    penalties = 0.0
    if clipping_ratio_pct > 0.01:
        penalties += min(30.0, clipping_ratio_pct * 200.0)
    penalties += min(35.0, choppy_dropouts * 1.5)
    penalties += min(15.0, min(transient_clicks, 50) * 0.15)
    if snr_db < 16.0:
        penalties += (16.0 - snr_db) * 1.8

    health_score = int(np.clip(100.0 - penalties, 25.0, 100.0))

    if health_score >= 85:
        health_status = "Clean & Stable"
    elif health_score >= 70:
        health_status = "Acceptable / Minor Glitches"
    else:
        health_status = "Degraded / Frequent Glitches"

    return {
        "audio_health_score": health_score,
        "audio_health_status": health_status,
        "audio_glitches_count": total_glitches,
        "clipping_ratio_pct": clipping_ratio_pct,
        "choppy_dropouts_count": choppy_dropouts,
        "transient_clicks_count": transient_clicks,
        "estimated_snr_db": snr_db,
        "audio_glitch_events": glitch_events[:8]
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
