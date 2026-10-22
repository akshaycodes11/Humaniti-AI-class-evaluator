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

    current_start = ""
    current_end = ""

    for line in lines:
        line_str = line.strip()
        time_match = time_pattern.search(line_str)
        if time_match:
            current_start = time_match.group(1)[:5]  # MM:SS
            current_end = time_match.group(2)[:5]
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

    if not utterances:
        return {"utterances": [], "speakers": {}, "summary": {}}

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
    student_words = total_words - teacher_words
    teacher_talk_pct = round((teacher_words / float(total_words)) * 100.0, 1)

    # Keyword signals based on 8x8 evaluator criteria
    full_text = " ".join([u["text"].lower() for u in utterances])
    
    # Positive reinforcement cues
    praise_words = ["great", "excellent", "good job", "well done", "awesome", "super", "wonderful", "fantastic", "brilliant"]
    praise_count = sum(full_text.count(p) for p in praise_words)

    # Questions asked by teacher
    teacher_questions = sum(1 for u in utterances if u["speaker"] == primary_teacher and "?" in u["text"])

    # Homework / HW mention
    hw_mentioned = bool(re.search(r"\b(homework|hw|assignment|practice sheet)\b", full_text))

    # Greeting / Rapport in beginning (first 10 utterances)
    first_text = " ".join([u["text"].lower() for u in utterances[:10]])
    has_greeting = bool(re.search(r"\b(good evening|good afternoon|hello|welcome|how are you|introduce|introduction)\b", first_text))

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
        "sample_utterances": utterances[:15]
    }
