import re
from typing import Dict, List, Any, Optional


def parse_vtt_file(vtt_path: str) -> Dict[str, Any]:
    """
    Parses a WebVTT transcript file from an 8x8 class recording.
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
                utterances.append({
                    "start": current_start,
                    "end": current_end,
                    "speaker": speaker,
                    "text": text
                })
                raw_cues.append((curr_start_s, curr_end_s, speaker, text))

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
        "total_words": total_words,
        "duration_seconds": round(class_duration_s, 1),
        "silence_ratio_pct": silence_ratio_pct,
        "dead_air_incidents": len(dead_air_intervals),
        "longest_silence_seconds": round(longest_silence_s, 1),
        "dead_air_intervals": dead_air_intervals[:10],
        "vocal_turns_count": len(utterances),
        "turn_taking_rate_per_min": turn_taking_rate,
        "sample_utterances": utterances[:15]
    }

