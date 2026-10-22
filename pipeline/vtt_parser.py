import re
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pipeline.security import scrub_pii


def parse_vtt_file(vtt_path: str) -> Dict[str, Any]:
    """
    Parses a WebVTT transcript file from an 8x8 class recording.
    Guarded by:
    - COPPA / Minor PII Redaction (strips emails, phone numbers, IP addresses).
    Extracts:
    - Utterances with timestamps and speakers
    - Speaker talk distribution (teacher vs students)
    - Question frequency
    - Names called (one-to-one interaction)
    - Positive reinforcement keywords
    - Homework / conclusion cues
    """
    try:
        with open(vtt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return {"error": str(e), "utterances": []}

    utterances = []
    total_pii_redacted = 0

    # Regex to match timestamps: 00:01:41.190 --> 00:01:42.590
    time_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})")
    speaker_pattern = re.compile(r"^([^:\n]+):\s*(.*)$")

    def parse_ts_to_sec(ts: str) -> float:
        parts = ts.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return 0.0

    raw_cues: List[Tuple[float, float, str, str]] = []
    current_start = ""
    current_end = ""
    curr_start_s = 0.0
    curr_end_s = 0.0

    for line in lines:
        line_str = line.strip()
        time_match = time_pattern.search(line_str)
        if time_match:
            current_start = time_match.group(1)[:5]  # MM:SS
            current_end = time_match.group(2)[:5]
            curr_start_s = parse_ts_to_sec(time_match.group(1))
            curr_end_s = parse_ts_to_sec(time_match.group(2))
            continue

        spk_match = speaker_pattern.match(line_str)
        if spk_match and current_start:
            speaker = spk_match.group(1).strip()
            text = spk_match.group(2).strip()
            if text:
                clean_text, pii_meta = scrub_pii(text)
                if pii_meta.get("is_scrubbed"):
                    total_pii_redacted += pii_meta.get("redacted_count", 0)
                utterances.append({
                    "start": current_start,
                    "end": current_end,
                    "speaker": speaker,
                    "text": clean_text
                })
                raw_cues.append((curr_start_s, curr_end_s, speaker, clean_text))


    if not utterances:
        return {"utterances": [], "speakers": {}, "summary": {}}

    # Analyze timing & pauses between utterances
    class_duration_s = max(60.0, raw_cues[-1][1] - raw_cues[0][0]) if raw_cues else 1800.0
    dead_air_intervals: List[Tuple[float, float]] = []
    total_silence_s = 0.0
    longest_silence_s = 0.0

    for i in range(len(raw_cues) - 1):
        gap = raw_cues[i+1][0] - raw_cues[i][1]
        if gap > 0:
            total_silence_s += gap
            if gap > longest_silence_s:
                longest_silence_s = gap
            if gap >= 7.0:
                dead_air_intervals.append((round(raw_cues[i][1], 1), round(raw_cues[i+1][0], 1)))

    silence_ratio_pct = round((total_silence_s / class_duration_s) * 100.0, 1)
    turn_taking_rate = round(len(utterances) / (class_duration_s / 60.0), 2)

    # Analyze speakers
    speakers_word_count: Dict[str, int] = {}
    speakers_turns: Dict[str, int] = {}
    for u in utterances:
        spk = u["speaker"]
        words = len(u["text"].split())
        speakers_word_count[spk] = speakers_word_count.get(spk, 0) + words
        speakers_turns[spk] = speakers_turns.get(spk, 0) + 1

    # Identify primary teacher (speaker with highest word count)
    primary_teacher = max(speakers_word_count.items(), key=lambda x: x[1])[0] if speakers_word_count else "Teacher"
    total_words = sum(speakers_word_count.values()) or 1
    teacher_words = speakers_word_count.get(primary_teacher, 0)
    teacher_talk_pct = round((teacher_words / float(total_words)) * 100.0, 1)

    # Keyword signals based on 8x8 evaluator criteria
    full_text = " ".join([u["text"].lower() for u in utterances])
    
    # Positive reinforcement cues
    praise_words = ["great", "excellent", "good job", "well done", "awesome", "super", "wonderful", "fantastic", "brilliant"]
    praise_count = sum(full_text.count(p) for p in praise_words)

    # Questions asked by teacher
    teacher_questions = sum(1 for u in utterances if u["speaker"] == primary_teacher and "?" in u["text"])

    # Homework / HW mention
    hw_mentioned = bool(re.search(r"\b(homework|hw|assignment|practice sheet|puzzles)\b", full_text))

    # Greeting / Rapport in beginning (first 10 utterances)
    first_text = " ".join([u["text"].lower() for u in utterances[:10]])
    has_greeting = bool(re.search(r"\b(good evening|good afternoon|good morning|hello|hi|welcome|how are you|introduce|introduction)\b", first_text))

    # Technical audio / connection issues detected in transcript
    technical_glitches = []
    tech_patterns = [
        r"\bcan you hear me\b",
        r"\bam i audible\b",
        r"\byou are breaking up\b",
        r"\baudio is lagging\b",
        r"\bcan't hear\b",
        r"\bcannot hear\b",
        r"\bvoice is breaking\b",
        r"\bvoice is cracking\b",
        r"\bscreen is frozen\b",
        r"\bcan't see the screen\b",
        r"\breconnect\b"
    ]
    tech_regex = re.compile("|".join(tech_patterns), re.IGNORECASE)
    for u in utterances:
        if tech_regex.search(u["text"]):
            technical_glitches.append({
                "timestamp": u.get("start", "00:00")[:5],
                "type": "Technical / Audio Glitch Mention",
                "severity": "Medium",
                "description": f"{u['speaker']}: \"{u['text']}\""
            })

    # Group consecutive cues by the same speaker into coherent conversational turns
    turns: List[Dict[str, Any]] = []
    curr_turn = None
    for start_s, end_s, spk, txt in raw_cues:
        if curr_turn is None:
            curr_turn = {"speaker": spk, "start_s": start_s, "end_s": end_s, "text": txt}
        elif curr_turn["speaker"] == spk:
            curr_turn["end_s"] = end_s
            curr_turn["text"] += " " + txt
        else:
            turns.append(curr_turn)
            curr_turn = {"speaker": spk, "start_s": start_s, "end_s": end_s, "text": txt}
    if curr_turn:
        turns.append(curr_turn)

    # Bloom's Taxonomy Question Classification
    hot_pat = re.compile(
        r"\b(why|how come|what if|how did you|how would you|explain|what happens when|what happens if|plan|strategy|idea|think about|reason|justify|compare|how do you know)\b",
        re.IGNORECASE
    )
    proc_pat = re.compile(
        r"\b(which move|where should|where can|how do we|how can we|what should|what do we play|next step|can you find|can you show|how to solve|which square)\b",
        re.IGNORECASE
    )

    hot_questions: List[Dict[str, str]] = []
    proc_questions: List[Dict[str, str]] = []
    recall_questions: List[Dict[str, str]] = []
    wait_times: List[float] = []
    self_interrupted_count = 0

    for i in range(len(turns)):
        turn = turns[i]
        if turn["speaker"] == primary_teacher and "?" in turn["text"]:
            q_text = turn["text"].strip()
            turn_min = int(turn["start_s"] // 60)
            turn_sec = int(turn["start_s"] % 60)
            ts_str = f"{turn_min:02d}:{turn_sec:02d}"

            if hot_pat.search(q_text):
                hot_questions.append({"timestamp": ts_str, "text": q_text, "type": "Higher-Order / Inquiry"})
            elif proc_pat.search(q_text):
                proc_questions.append({"timestamp": ts_str, "text": q_text, "type": "Procedural / Application"})
            else:
                recall_questions.append({"timestamp": ts_str, "text": q_text, "type": "Recall / Closed"})

            if i < len(turns) - 1:
                next_turn = turns[i + 1]
                gap = max(0.1, next_turn["start_s"] - turn["end_s"])
                if next_turn["speaker"] != primary_teacher:
                    wait_times.append(round(gap, 2))
                else:
                    if gap < 2.5:
                        self_interrupted_count += 1

    total_classified_q = len(hot_questions) + len(proc_questions) + len(recall_questions)
    hot_ratio_pct = round((len(hot_questions) / float(max(1, total_classified_q))) * 100.0, 1)

    if wait_times:
        avg_wait_s = round(float(np.mean(wait_times)), 2)
        median_wait_s = round(float(np.median(wait_times)), 2)
        deep_think_count = int(np.sum(np.array(wait_times) >= 2.5))
        if 2.0 <= median_wait_s <= 5.0:
            wait_time_score = min(100, int(85 + deep_think_count * 1.5))
        elif median_wait_s < 2.0:
            wait_time_score = max(35, int(60 + median_wait_s * 12))
        else:
            wait_time_score = max(40, int(90 - (median_wait_s - 5.0) * 8))
    else:
        avg_wait_s = 2.5
        median_wait_s = 2.0
        deep_think_count = 0
        wait_time_score = 75

    return {
        "utterances_count": len(utterances),
        "primary_teacher": primary_teacher,
        "speakers_list": list(speakers_word_count.keys()),
        "teacher_talk_ratio_pct": teacher_talk_pct,
        "student_talk_ratio_pct": round(100.0 - teacher_talk_pct, 1),
        "praise_count": praise_count,
        "teacher_questions_count": teacher_questions,
        "homework_mentioned": hw_mentioned,
        "rapport_greeting_detected": has_greeting,
        "audio_glitches_count": len(technical_glitches),
        "audio_glitch_events": technical_glitches[:8],
        "avg_wait_time_seconds": avg_wait_s,
        "median_wait_time_seconds": median_wait_s,
        "deep_think_time_count": deep_think_count,
        "self_interrupted_questions_count": self_interrupted_count,
        "wait_time_effectiveness_score": wait_time_score,
        "open_ended_questions_count": len(hot_questions),
        "closed_questions_count": len(recall_questions) + len(proc_questions),
        "higher_order_ratio_pct": hot_ratio_pct,
        "bloom_categories": {
            "Higher_Order_Analysis": len(hot_questions),
            "Procedural_Application": len(proc_questions),
            "Recall_Closed": len(recall_questions)
        },
        "sample_inquiry_questions": hot_questions[:5],
        "total_words": total_words,
        "duration_seconds": round(class_duration_s, 1),
        "silence_ratio_pct": silence_ratio_pct,
        "dead_air_incidents": len(dead_air_intervals),
        "longest_silence_seconds": round(longest_silence_s, 1),
        "dead_air_intervals": dead_air_intervals[:10],
        "vocal_turns_count": len(utterances),
        "turn_taking_rate_per_min": turn_taking_rate,
        "pii_redacted_count": total_pii_redacted,
        "sample_utterances": utterances[:15]
    }


