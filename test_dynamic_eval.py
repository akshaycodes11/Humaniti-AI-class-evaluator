import os, glob
from pipeline.vtt_parser import parse_vtt_file
from pipeline.evaluator import evaluate_class_local_fallback
from pipeline.graph import build_evaluation_graph, ClassEvaluationState

print("=== 1. TESTING DYNAMIC EVALUATION ON ALL TRANSCRIPTS ===")
vtt_files = sorted(glob.glob("dataset_samples/transcripts/*.vtt"))

for f in vtt_files:
    vtt_analysis = parse_vtt_file(f)
    telemetry = {
        "duration_seconds": vtt_analysis.get("duration_seconds", 2400.0),
        "silence_ratio_pct": vtt_analysis.get("silence_ratio_pct", 25.0),
        "dead_air_incidents": vtt_analysis.get("dead_air_incidents", 3),
        "longest_silence_seconds": vtt_analysis.get("longest_silence_seconds", 8.0),
        "dead_air_intervals": vtt_analysis.get("dead_air_intervals", []),
        "vocal_turns_count": vtt_analysis.get("vocal_turns_count", 80),
        "turn_taking_rate_per_min": vtt_analysis.get("turn_taking_rate_per_min", 2.0),
        "teacher_talk_ratio_pct": vtt_analysis.get("teacher_talk_ratio_pct", 60.0),
        "student_talk_ratio_pct": vtt_analysis.get("student_talk_ratio_pct", 40.0),
        "questions_count": vtt_analysis.get("teacher_questions_count", 20),
        "praise_count": vtt_analysis.get("praise_count", 4),
        "closing_wrapup_detected": vtt_analysis.get("homework_mentioned", False),
        "visual_change_rate": 1.2,
        "total_slide_transitions_detected": 10,
        "estimated_talk_pacing_wpm": 138
    }
    meta = {"has_video": True, "width": 1280, "height": 720, "duration_seconds": telemetry["duration_seconds"]}
    rep = evaluate_class_local_fallback(telemetry, meta, [], vtt_analysis=vtt_analysis)
    
    print(f"{os.path.basename(f):35s} -> {rep.overall_rating.upper():9s} | Score: {rep.overall_score:2d}/100 | Eng: {rep.student_engagement.score}/10 | Clar: {rep.instructional_clarity.score}/10 | Clm: {rep.emotional_climate.score}/10 | Vis: {rep.visual_and_screen_utility.score}/10 | Pace: {rep.pacing_and_flow.score}/10")
    print(f"   Summary: {rep.executive_summary[:120]}...")

print("\n=== 2. TESTING FULL LANGGRAPH STATEGRAPH EXECUTION ===")
graph = build_evaluation_graph()
sample_mp4 = os.path.join("dataset_samples", "class recordings", "average", "B435 Intermediate Part  - 2026-08-28.mp4")

if os.path.exists(sample_mp4):
    state: ClassEvaluationState = {
        "media_path": sample_mp4,
        "vtt_path": None,  # Test auto-resolution of transcript!
        "api_key": None,
        "use_local_only": True,
        "working_dir": os.path.join(os.environ.get("TEMP", "."), "humaniti_test"),
        "metadata": {},
        "keyframes": [],
        "audio_path": None,
        "acoustic_telemetry": {},
        "visual_telemetry": {},
        "vtt_analysis": None,
        "final_report": None,
        "execution_logs": [],
        "current_step": "init"
    }
    final_state = graph.invoke(state)
    rep = final_state.get("final_report")
    print(f"LangGraph execution succeeded!")
    print(f"Auto-resolved VTT: {os.path.basename(final_state.get('vtt_analysis', {}).get('matched_path', 'N/A') or 'Acoustics')}")
    print(f"Report: {rep.overall_rating} ({rep.overall_score}/100)")
    print(f"Executive Summary:\n{rep.executive_summary}")
    print(f"Telemetry summary turns: {rep.telemetry.vocal_turns_count if rep.telemetry else 'N/A'}, questions: {rep.telemetry.questions_count if rep.telemetry else 'N/A'}")
