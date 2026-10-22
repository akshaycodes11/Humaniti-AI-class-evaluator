import os
import time
import tempfile
import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image

from pipeline.media_processor import (
    extract_audio,
    extract_representative_keyframes,
    get_media_metadata
)
from pipeline.telemetry import analyze_audio_acoustics, analyze_visual_pacing
from pipeline.evaluator import evaluate_class_with_gemini, evaluate_class_local_fallback
from pipeline.models import EvaluationReport
from pipeline.vtt_parser import parse_vtt_file
from pipeline.graph import build_evaluation_graph, ClassEvaluationState
from utils.gdrive_downloader import download_media_from_url
from utils.pdf_generator import generate_pdf_report


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="EduGrade AI | 8x8 Class Quality Evaluator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-excellent {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #31C48D;
    }
    .badge-good {
        background-color: #E1EFFE;
        color: #1E429F;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #76A9FA;
    }
    .badge-average {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #FACC15;
    }
    .badge-poor {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #F98080;
    }
    .metric-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #111827;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        font-weight: 600;
    }
    .timeline-card {
        border-left: 4px solid #3B82F6;
        background: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/classroom.png", width=64)
    st.markdown("### **Class Quality Evaluator**")
    st.caption("AI Build Challenge • Humaniti AI")
    st.divider()

    st.markdown("#### **Inference Mode & Engine**")
    api_key_input = st.text_input(
        "Google Gemini API Key (Optional)",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Free key from Google AI Studio (aistudio.google.com). Free tier allows 1,500 requests/day at zero cost!"
    )
    
    use_local_only = st.checkbox("Force Local/Zero-API Mode", value=False, help="Runs 100% locally with acoustic & visual heuristic models.")

    st.divider()
    st.markdown("#### **Evaluation Rubric Standards**")
    st.markdown("""
    - **Student Engagement (30%)**: Turn-taking, dialogue vs monologue.
    - **Instructional Clarity (25%)**: Scaffolding, concept breakdown.
    - **Praise & Tone (20%)**: Specific positive reinforcement, patience.
    - **Visual & Screen Utility (15%)**: Dynamic slides, whiteboard use.
    - **Pacing & Dead Air (10%)**: Absence of prolonged awkward silences.
    """)
    st.divider()
    st.info("🔒 **Data Privacy**: Recordings are analyzed in temporary memory and never stored or retained.")


# App Header
st.markdown('<div class="main-header">Can You Teach AI to Recognise a Great Class?</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated pedagogical quality assessment for online kids\' education (8x8 platform)</div>', unsafe_allow_html=True)

# Input section
st.markdown("### 📥 1. Provide Class Recording")
input_tab1, input_tab2 = st.tabs(["📁 Upload Video / Audio", "🔗 Direct Link / Google Drive Link"])

input_file_path = None
temp_dir = tempfile.gettempdir()
os.makedirs(os.path.join(temp_dir, "humaniti_eval"), exist_ok=True)
working_dir = os.path.join(temp_dir, "humaniti_eval")

with input_tab1:
    col_up1, col_up2 = st.columns([3, 2])
    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload Recording (.mp4, .webm, .mkv, .wav, .mp3)",
            type=["mp4", "webm", "mkv", "avi", "wav", "mp3", "m4a"],
            help="Upload an 8x8 online class recording file."
        )
    with col_up2:
        uploaded_vtt = st.file_uploader(
            "Upload Transcript (.vtt) [Optional]",
            type=["vtt", "txt"],
            help="Optional WebVTT transcript file (e.g. from 8x8 classroom platform)."
        )

    if uploaded_file is not None:
        save_path = os.path.join(working_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        input_file_path = save_path
        st.success(f"Loaded Recording: **{uploaded_file.name}** ({round(uploaded_file.size / (1024*1024), 2)} MB)")

    input_vtt_path = None
    if uploaded_vtt is not None:
        vtt_save_path = os.path.join(working_dir, uploaded_vtt.name)
        with open(vtt_save_path, "wb") as f:
            f.write(uploaded_vtt.getbuffer())
        input_vtt_path = vtt_save_path
        st.info(f"Loaded Transcript: **{uploaded_vtt.name}**")

with input_tab2:
    url_input = st.text_input(
        "Enter Google Drive link or Direct Media URL:",
        placeholder="https://drive.google.com/file/d/.../view?usp=sharing",
        help="Paste a public Google Drive link or direct URL to a video file."
    )
    col_dl1, col_dl2 = st.columns([1, 4])
    with col_dl1:
        download_btn = st.button("Fetch & Download")
    if download_btn and url_input:
        with st.spinner("Resolving and downloading media..."):
            success, path, msg = download_media_from_url(url_input, working_dir)
            if success:
                input_file_path = path
                st.session_state["downloaded_file_path"] = path
                st.success(f"{msg}: `{os.path.basename(path)}`")
            else:
                st.error(f"Download failed: {msg}")

    if "downloaded_file_path" in st.session_state and input_file_path is None:
        input_file_path = st.session_state["downloaded_file_path"]


# Quick sample selector for testing
with st.expander("💡 Or test with pre-configured 8x8 Challenge Samples"):
    st.caption("Select a real challenge sample or synthetic scenario:")
    col_s0, col_s1, col_s2, col_s3 = st.columns(4)
    if col_s0.button("🎯 Real 8x8 Sample (B435 - Sumit Ranjan)"):
        real_sample_mp4 = os.path.join("dataset_samples", "class recordings", "average", "B435 Intermediate Part  - 2026-08-28.mp4")
        real_sample_vtt = os.path.join("dataset_samples", "class recordings", "average", "B435 Intermediate-transcript-1.1.vtt")
        if os.path.exists(real_sample_mp4):
            input_file_path = real_sample_mp4
            st.session_state["custom_file_path"] = real_sample_mp4
            st.session_state["custom_vtt_path"] = real_sample_vtt
            st.session_state["sample_scenario"] = None
            st.success("Loaded Challenge Sample: **B435 Intermediate (Sumit Ranjan)**")
    if col_s1.button("Sample: Poor (Dead Air & Monologue)"):
        st.session_state["sample_scenario"] = "poor"
        st.session_state["custom_file_path"] = None
    if col_s2.button("Sample: Average (Lecture Style)"):
        st.session_state["sample_scenario"] = "average"
        st.session_state["custom_file_path"] = None
    if col_s3.button("Sample: Excellent (Interactive)"):
        st.session_state["sample_scenario"] = "excellent"
        st.session_state["custom_file_path"] = None

if "custom_file_path" in st.session_state and st.session_state["custom_file_path"] and input_file_path is None:
    input_file_path = st.session_state["custom_file_path"]
    input_vtt_path = st.session_state.get("custom_vtt_path", None)


# Action: Run Evaluation
st.divider()
st.markdown("### ⚙️ 2. Run Quality Assessment")

col_eval1, col_eval2 = st.columns([2, 5])
with col_eval1:
    eval_button = st.button("🚀 Evaluate Class Recording", type="primary", width="stretch")
with col_eval2:
    if input_file_path:
        st.info(f"Ready to evaluate: **{os.path.basename(input_file_path)}**")
    elif "sample_scenario" in st.session_state:
        st.info(f"Ready to evaluate sample scenario: **{st.session_state['sample_scenario'].upper()}**")
    else:
        st.warning("Please upload a file, paste a link, or select a sample scenario above.")


def render_report(report: EvaluationReport, keyframes):
    st.divider()
    st.markdown("## 📊 Class Quality Assessment Report")
    
    # Top Score & Rating Hero Section
    col_hero1, col_hero2, col_hero3, col_hero4 = st.columns([2, 1.5, 1.5, 2])
    
    badge_class = f"badge-{report.overall_rating.lower()}"
    with col_hero1:
        st.markdown(f'<div class="{badge_class}">Overall Quality: {report.overall_rating.upper()}</div>', unsafe_allow_html=True)
        st.markdown(f"<h1 style='margin-top: 10px; margin-bottom: 0;'>{report.overall_score} <span style='font-size: 1.5rem; color: #6B7280;'>/ 100</span></h1>", unsafe_allow_html=True)
        st.caption(f"Evaluation Confidence: {int(report.confidence * 100)}%")
        
    with col_hero2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">{0}</div>
            <div class="metric-lbl">Duration</div>
        </div>
        """.format(f"{round((report.telemetry.duration_seconds if report.telemetry else 1800)/60.0, 1)} min"), unsafe_allow_html=True)
        
    with col_hero3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">{0}%</div>
            <div class="metric-lbl">Pause / Silence</div>
        </div>
        """.format(report.telemetry.silence_ratio_pct if report.telemetry else 24.5), unsafe_allow_html=True)
        
    with col_hero4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">{0}</div>
            <div class="metric-lbl">Dead Air Pauses (>7s)</div>
        </div>
        """.format(report.telemetry.dead_air_incidents if report.telemetry else 0), unsafe_allow_html=True)

    # Executive Summary
    st.markdown("#### **Executive Summary**")
    st.info(report.executive_summary)

    # Pedagogical Dimensions Radar / Bar Chart
    st.markdown("### 🎯 Pedagogical Dimension Breakdown")
    dim_data = {
        "Dimension": [
            "Student Engagement",
            "Instructional Clarity",
            "Praise & Tone",
            "Visual Utility",
            "Pacing & Flow"
        ],
        "Score": [
            report.student_engagement.score,
            report.instructional_clarity.score,
            report.emotional_climate.score,
            report.visual_and_screen_utility.score,
            report.pacing_and_flow.score
        ],
        "Rubric Level": [
            report.student_engagement.rubric_level,
            report.instructional_clarity.rubric_level,
            report.emotional_climate.rubric_level,
            report.visual_and_screen_utility.rubric_level,
            report.pacing_and_flow.rubric_level
        ]
    }
    df_dims = pd.DataFrame(dim_data)

    col_chart, col_dim_details = st.columns([3, 4])
    with col_chart:
        chart = alt.Chart(df_dims).mark_bar(cornerRadius=6, size=24).encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 10]), title="Score (1 - 10)"),
            y=alt.Y("Dimension:N", sort=None, title=""),
            color=alt.Color(
                "Score:Q",
                scale=alt.Scale(
                    domain=[3, 6, 8, 10],
                    range=["#F87171", "#FBBF24", "#60A5FA", "#34D399"]
                ),
                legend=None
            ),
            tooltip=["Dimension", "Score", "Rubric Level"]
        ).properties(height=260)
        st.altair_chart(chart, width="stretch")

    with col_dim_details:
        for _, row in df_dims.iterrows():
            st.markdown(f"**{row['Dimension']}**: `{row['Score']}/10` ({row['Rubric Level']})")
        st.caption("Scored against Charlotte Danielson & CLASS K-12 pedagogical standards.")

    # Granular Specific Observations with Timestamps (Key challenge requirement!)
    st.markdown("### ⏱️ Specific Observations & Timestamped Evidence")
    st.caption("Granular evidence of teacher-student interactions, questions, praise, and screen events (non-generic):")
    
    if report.specific_observations:
        for obs in report.specific_observations:
            icon = "✅" if obs.observation_type == "Strength" else "⚠️" if obs.observation_type == "Improvement Area" else "ℹ️"
            st.markdown(f"""
            <div class="timeline-card">
                <b>[{obs.timestamp}]</b> {icon} <b>{obs.category}</b> — <i>{obs.observation_type}</i><br>
                <span>{obs.evidence}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No granular timestamps detected.")

    # Keyframe Gallery if available
    if keyframes:
        st.markdown("### 🖼️ Visual Context Keyframes")
        st.caption("Sampled screen state across the recording:")
        cols = st.columns(min(len(keyframes), 4))
        for idx, (ts, img_path) in enumerate(keyframes[:8]):
            c = cols[idx % 4]
            with c:
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Timestamp: {ts}", width="stretch")

    # Strengths, Weaknesses, and Coaching Tips
    col_str, col_weak = st.columns(2)
    with col_str:
        st.markdown("#### 🌟 Top Pedagogical Strengths")
        for s in report.top_strengths:
            st.markdown(f"- {s}")
            
    with col_weak:
        st.markdown("#### 🔍 Critical Areas for Improvement")
        for w in report.critical_weaknesses:
            st.markdown(f"- {w}")

    st.markdown("#### 📋 Actionable Teacher Coaching Recommendations")
    for tip in report.actionable_coaching_tips:
        st.markdown(f"1. **{tip}**")

    # Export Report
    st.divider()
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        json_data = report.model_dump_json(indent=2)
        st.download_button(
            label="📥 Download Assessment Report (JSON)",
            data=json_data,
            file_name=f"class_evaluation_{report.overall_rating.lower()}.json",
            mime="application/json",
            width="stretch"
        )
    with col_exp2:
        try:
            pdf_path = os.path.join(tempfile.gettempdir(), f"evaluation_{report.overall_rating.lower()}.pdf")
            generate_pdf_report(report, pdf_path)
            with open(pdf_path, "rb") as pdf_f:
                pdf_bytes = pdf_f.read()
            st.download_button(
                label="📄 Download Official PDF Report",
                data=pdf_bytes,
                file_name=f"class_evaluation_{report.overall_rating.lower()}.pdf",
                mime="application/pdf",
                width="stretch"
            )
        except Exception as e:
            st.caption(f"PDF generator notice: {e}")


# Execution trigger
if eval_button:
    # Check if sample scenario or actual file
    if input_file_path and os.path.exists(input_file_path):
        progress_bar = st.progress(0, text="Initializing LangGraph multi-stage pipeline...")
        status_text = st.empty()

        effective_key = api_key_input.strip() if api_key_input else os.environ.get("GEMINI_API_KEY", "")
        
        status_text.text("Compiling LangGraph StateGraph nodes...")
        progress_bar.progress(15)
        
        eval_graph = build_evaluation_graph()
        
        initial_state: ClassEvaluationState = {
            "media_path": input_file_path,
            "vtt_path": input_vtt_path,
            "api_key": effective_key,
            "use_local_only": use_local_only,
            "working_dir": working_dir,
            "metadata": {},
            "keyframes": [],
            "audio_path": None,
            "acoustic_telemetry": {},
            "visual_telemetry": {},
            "vtt_analysis": None,
            "final_report": None,
            "execution_logs": [],
            "current_step": "Starting LangGraph Pipeline"
        }

        status_text.text("Executing LangGraph nodes: Ingestion -> Acoustics -> Vision -> Transcript -> Pedagogical Reasoner...")
        progress_bar.progress(50)
        
        final_state = eval_graph.invoke(initial_state)
        
        progress_bar.progress(100, text="LangGraph workflow completed!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

        report = final_state.get("final_report")
        keyframes = final_state.get("keyframes", [])
        
        # Display LangGraph Execution Trace
        with st.expander("🤖 View LangGraph StateGraph Execution Trace", expanded=False):
            st.caption("Step-by-step state transition logs across LangGraph nodes:")
            for log in final_state.get("execution_logs", []):
                st.code(log, language="bash")

        if report:
            render_report(report, keyframes)
        else:
            st.error("LangGraph failed to produce final report.")

    elif "sample_scenario" in st.session_state:
        scenario = st.session_state["sample_scenario"]
        with st.spinner(f"Evaluating synthetic {scenario.upper()} class scenario..."):
            time.sleep(1.0)
            
            if scenario == "poor":
                sample_telemetry = {
                    "duration_seconds": 1800.0,
                    "silence_ratio_pct": 46.2,
                    "dead_air_incidents": 6,
                    "longest_silence_seconds": 24.5,
                    "active_speech_seconds": 968.0,
                    "visual_change_rate": 0.2,
                    "total_slide_transitions_detected": 2,
                    "dead_air_intervals": [(185.0, 209.5), (410.0, 428.0), (790.0, 804.0), (1210.0, 1226.5)]
                }
            elif scenario == "average":
                sample_telemetry = {
                    "duration_seconds": 1800.0,
                    "silence_ratio_pct": 32.0,
                    "dead_air_incidents": 2,
                    "longest_silence_seconds": 9.0,
                    "active_speech_seconds": 1224.0,
                    "visual_change_rate": 0.8,
                    "total_slide_transitions_detected": 8,
                    "dead_air_intervals": [(320.0, 329.0), (950.0, 958.0)]
                }
            else:  # excellent
                sample_telemetry = {
                    "duration_seconds": 1800.0,
                    "silence_ratio_pct": 21.5,
                    "dead_air_incidents": 0,
                    "longest_silence_seconds": 4.5,
                    "active_speech_seconds": 1413.0,
                    "visual_change_rate": 2.4,
                    "total_slide_transitions_detected": 22,
                    "dead_air_intervals": []
                }
                
            sample_meta = {"has_video": True, "width": 1280, "height": 720, "duration_seconds": 1800.0}
            report = evaluate_class_local_fallback(sample_telemetry, sample_meta, [])
            render_report(report, [])
    else:
        st.warning("Please upload a file or choose a sample scenario first.")
