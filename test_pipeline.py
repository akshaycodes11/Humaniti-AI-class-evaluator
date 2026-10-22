"""
Automated End-to-End Verification Test for EduGrade AI Pipeline
"""
from pipeline.models import EvaluationReport
from pipeline.evaluator import evaluate_class_local_fallback


def test_poor_scenario():
    telemetry = {
        "duration_seconds": 1800.0,
        "silence_ratio_pct": 52.0,
        "dead_air_incidents": 7,
        "longest_silence_seconds": 28.0,
        "active_speech_seconds": 860.0,
        "visual_change_rate": 0.1,
        "total_slide_transitions_detected": 1,
        "dead_air_intervals": [(120.0, 148.0), (350.0, 378.0)]
    }
    meta = {"has_video": True, "width": 1280, "height": 720, "duration_seconds": 1800.0}
    vtt = {
        "teacher_talk_ratio_pct": 92.0,
        "teacher_questions_count": 3,
        "praise_count": 0,
        "homework_mentioned": False,
        "rapport_greeting_detected": False
    }
    report: EvaluationReport = evaluate_class_local_fallback(telemetry, meta, [], vtt_analysis=vtt)
    
    assert report.overall_rating == "Poor", f"Expected Poor, got {report.overall_rating}"
    assert report.overall_score < 48, f"Expected score < 48, got {report.overall_score}"
    assert len(report.specific_observations) > 0, "Expected specific observations"
    print(f"Poor Scenario Test Passed: Score={report.overall_score}/100, Rating={report.overall_rating}")


def test_excellent_scenario():
    telemetry = {
        "duration_seconds": 1800.0,
        "silence_ratio_pct": 21.0,
        "dead_air_incidents": 0,
        "longest_silence_seconds": 4.0,
        "active_speech_seconds": 1422.0,
        "visual_change_rate": 2.5,
        "total_slide_transitions_detected": 25,
        "dead_air_intervals": []
    }
    meta = {"has_video": True, "width": 1280, "height": 720, "duration_seconds": 1800.0}
    vtt = {
        "teacher_talk_ratio_pct": 55.0,
        "teacher_questions_count": 65,
        "praise_count": 14,
        "homework_mentioned": True,
        "rapport_greeting_detected": True
    }
    report: EvaluationReport = evaluate_class_local_fallback(telemetry, meta, [], vtt_analysis=vtt)
    
    assert report.overall_rating == "Excellent", f"Expected Excellent, got {report.overall_rating}"
    assert report.overall_score >= 78, f"Expected score >= 78, got {report.overall_score}"
    print(f"Excellent Scenario Test Passed: Score={report.overall_score}/100, Rating={report.overall_rating}")


if __name__ == "__main__":
    test_poor_scenario()
    test_excellent_scenario()
    print("ALL TESTS PASSED SUCCESSFULLY!")
