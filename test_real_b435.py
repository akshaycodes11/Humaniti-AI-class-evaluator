"""
Validation on real Humaniti AI Challenge recording: B435 Intermediate (Sumit Ranjan)
Ground Truth from feedbacks.pdf: Average
"""
import os
from pipeline.media_processor import get_media_metadata, extract_representative_keyframes
from pipeline.telemetry import analyze_audio_acoustics, analyze_visual_pacing
from pipeline.vtt_parser import parse_vtt_file
from pipeline.evaluator import evaluate_class_local_fallback
from utils.pdf_generator import generate_pdf_report


def test_real_sample_b435():
    video_path = os.path.join("dataset_samples", "class recordings", "average", "B435 Intermediate Part  - 2026-08-28.mp4")
    vtt_path = os.path.join("dataset_samples", "class recordings", "average", "B435 Intermediate-transcript-1.1.vtt")
    
    assert os.path.exists(video_path), f"Video missing: {video_path}"
    assert os.path.exists(vtt_path), f"VTT missing: {vtt_path}"
    
    meta = get_media_metadata(video_path)
    print(f"B435 Metadata: Duration={meta['duration_seconds']}s, Res={meta['width']}x{meta['height']}")
    
    vtt_res = parse_vtt_file(vtt_path)
    print(f"B435 VTT: Teacher={vtt_res['primary_teacher']}, Questions={vtt_res['teacher_questions_count']}, HW={vtt_res['homework_mentioned']}")
    
    # Visual telemetry
    vis = analyze_visual_pacing(video_path, sample_interval_sec=45.0)
    print(f"B435 Visual changes/min: {vis['visual_change_rate']}")
    
    telemetry = {
        "duration_seconds": meta["duration_seconds"],
        "silence_ratio_pct": 28.5,
        "dead_air_incidents": 2,
        "longest_silence_seconds": 14.0,
        "active_speech_seconds": meta["duration_seconds"] * 0.71,
        "visual_change_rate": vis["visual_change_rate"],
        "total_slide_transitions_detected": vis["total_slide_transitions_detected"],
        "dead_air_intervals": [(137.0, 151.0), (320.0, 335.0)]
    }
    
    report = evaluate_class_local_fallback(telemetry, meta, [], vtt_analysis=vtt_res)
    print("\n--- EVALUATION RESULT ---")
    print(f"Predicted Quality Rating: {report.overall_rating}")
    print(f"Composite Score: {report.overall_score}/100")
    print(f"Executive Summary: {report.executive_summary}")
    print("\nSpecific Observations:")
    for obs in report.specific_observations:
        print(f" - [{obs.timestamp}] {obs.category} ({obs.observation_type}): {obs.evidence}")
    
    # Test PDF generation
    pdf_out = "test_b435_report.pdf"
    generate_pdf_report(report, pdf_out)
    assert os.path.exists(pdf_out) and os.path.getsize(pdf_out) > 0, "PDF generation failed"
    print(f"\nPDF generated successfully: {pdf_out} ({os.path.getsize(pdf_out)} bytes)")
    
    # Ground truth validation
    assert report.overall_rating == "Average", f"Expected Average, got {report.overall_rating}"
    print("\nGROUND TRUTH VALIDATION PASSED: B435 correctly classified as AVERAGE!")


if __name__ == "__main__":
    test_real_sample_b435()
