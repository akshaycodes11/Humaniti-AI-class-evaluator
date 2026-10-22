"""
Test LangGraph StateGraph Execution on Real Challenge Sample B435
"""
import os
from pipeline.graph import build_evaluation_graph, ClassEvaluationState

def test_langgraph_pipeline():
    video_path = os.path.join("dataset_samples", "class recordings", "average", "B435 Intermediate Part  - 2026-08-28.mp4")
    vtt_path = os.path.join("dataset_samples", "class recordings", "average", "B435 Intermediate-transcript-1.1.vtt")
    
    assert os.path.exists(video_path), "Sample video not found"
    
    # 1. Compile LangGraph workflow
    graph = build_evaluation_graph()
    print("LangGraph StateGraph compiled successfully!")
    
    # 2. Prepare initial state
    initial_state: ClassEvaluationState = {
        "media_path": video_path,
        "vtt_path": vtt_path,
        "api_key": None,
        "use_local_only": True,
        "working_dir": "dataset_samples/temp_eval",
        "metadata": {},
        "keyframes": [],
        "audio_path": None,
        "acoustic_telemetry": {},
        "visual_telemetry": {},
        "vtt_analysis": None,
        "final_report": None,
        "execution_logs": [],
        "current_step": "Initialized"
    }
    
    # 3. Execute LangGraph workflow
    print("\nExecuting LangGraph workflow nodes:")
    final_state = graph.invoke(initial_state)
    
    for log in final_state["execution_logs"]:
        print(f" -> {log}")
        
    report = final_state["final_report"]
    assert report is not None, "LangGraph failed to produce final_report"
    print(f"\nFinal State Rating: {report.overall_rating} (Score: {report.overall_score}/100)")
    print(f"Step Reached: {final_state['current_step']}")
    assert report.overall_rating == "Average", f"Expected Average, got {report.overall_rating}"
    print("LANGGRAPH WORKFLOW VERIFICATION PASSED!")

if __name__ == "__main__":
    test_langgraph_pipeline()
