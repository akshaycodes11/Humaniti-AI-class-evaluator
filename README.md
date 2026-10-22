# EduGrade AI: K-12 Online Class Quality Evaluator 🎓
*Automated Pedagogical Assessment Engine for 8x8 Virtual Classrooms*

Developed for the **Humaniti AI AI Build Challenge** — *"Can You Teach AI to Recognise a Great Class?"*

---

## 🌟 Executive Summary & Approach (< 300 Words Submission Summary)

Online classes for children cannot be judged solely by transcription or slide count. A high-quality K-12 class hinges on four critical dimensions: **conversational turn-taking** (avoiding one-sided monologues), **scaffolding & clarity**, **emotional climate** (specific, immediate praise like *"super job carrying the 1!"*), and **classroom pacing** (minimizing dead air and disengagement).

To achieve high accuracy with near-zero cost ($0.00) and ultra-low latency (< 15s), we designed a **Hierarchical Hybrid Architecture**:
1. **Deterministic Edge Telemetry (Zero-Cost)**: Rather than brute-forcing 45 minutes of heavy video tokens into costly APIs, the system extracts acoustic energy envelopes (silence %, turn-taking ratio, and >7s dead-air pauses) and visual screen transition dynamics using OpenCV.
2. **Keyframe & Acoustic Fusion**: Spaced keyframes capturing screen/whiteboard changes are coupled with acoustic intervals.
3. **Pedagogical Evaluation (Gemini 1.5/2.0 Flash Free Tier)**: The multimodal context is evaluated against Charlotte Danielson and CLASS pedagogical frameworks, generating strict Pydantic JSON outputs with timestamped citations.
4. **Resilient Dual-Engine**: If cloud APIs are unavailable, an automated deterministic fallback engine assesses the class locally with zero latency and zero API keys.

**What Worked**: Grounding observations in timestamped acoustic dead-air events and visual slide transitions eliminated generic feedback and produced actionable coaching recommendations.  
**Limitations**: Face engagement analysis is intentionally disabled to strictly uphold child privacy and eliminate biometric data risks.

---

## 🏗️ Architecture & Signal Pipeline

```
                       ┌──────────────────────────────┐
                       │    User Input (File/Link)    │
                       └──────────────┬───────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
     [OpenCV Video Pipeline]                       [Audio Extractor & VAD]
  - Extract 1 frame / 45s                       - Energy & Silence % Detection
  - Detect slide/scene changes                  - Pacing (Words/Min) & Dead Air
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │    Execution Path Router  │
                        └─────────────┬─────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
 ┌──────────────────────┐                           ┌──────────────────────┐
 │ Mode A: Gemini Flash │ (Default Free Cloud)      │ Mode B: Local Engine │ (Offline Fallback)
 │ • Audio/Frames ->    │                           │ • Local Whisper/Regex│
 │   pedagogical rubric │                           │ • Acoustic Telemetry │
 │ • JSON schema output │                           │ • Heuristic Rubric   │
 └──────────┬───────────┘                           └──────────┬───────────┘
            └─────────────────────────┬─────────────────────────┘
                                      ▼
                    ┌───────────────────────────────────┐
                    │ Verified Pedagogical Evaluation   │
                    │ (Poor / Average / Good / Excellent)│
                    │ + Actionable Timestamped Evidence │
                    └─────────────────┬─────────────────┘
                                      ▼
                    ┌───────────────────────────────────┐
                    │   Streamlit Interactive Dashboard │
                    └───────────────────────────────────┘
```

---

## 📊 Pedagogical Evaluation Rubric

| Criterion | Weight | What Distinguishes "Poor" vs. "Excellent" |
| :--- | :---: | :--- |
| **Student Engagement & Interactivity** | **30%** | **Poor:** Continuous teacher monologue with rhetorical questions. <br>**Excellent:** Dynamic dialogue, open-ended prompts, 40%+ student turn-taking. |
| **Instructional Clarity & Scaffolding** | **25%** | **Poor:** Unstructured concept jumping; no comprehension checks. <br>**Excellent:** Incremental scaffolding; checks understanding after every step. |
| **Praise & Emotional Climate** | **20%** | **Poor:** Cold, impatient, or purely generic acknowledgements (*"ok"*). <br>**Excellent:** Specific, enthusiastic reinforcement (*"wonderful reasoning on step 2!"*). |
| **Screen & Visual Utility** | **15%** | **Poor:** Screen stays static/frozen for 5+ mins; text unreadable. <br>**Excellent:** Dynamic slides, active digital annotation, visual cues. |
| **Pacing & Dead Air Management** | **10%** | **Poor:** Frequent unprompted dead air (>7s pauses) causing disengagement. <br>**Excellent:** Smooth transitions, prompt scaffolding when child hesitates. |

---

## 🚀 Quickstart & How to Run

### 1. Clone & Install Dependencies
```bash
git clone <your-repo-url>
cd "Humaniti AI-Class Evaluator"
pip install -r requirements.txt
```

### 2. Run the Application Locally
```bash
streamlit run app.py
```
The application will launch automatically at `http://localhost:8501`.

### 3. Optional: Configure Free Gemini API Key
* Get a 100% free API key from [Google AI Studio](https://aistudio.google.com/) (1,500 free requests per day, 0 cost).
* Enter it directly in the app's sidebar or set an environment variable:
  ```bash
  export GEMINI_API_KEY="your-free-key-here"
  ```
* *Note: Even without an API key, the app functions immediately using the built-in deterministic telemetry engine!*

---

## 🔒 Confidentiality & Child Privacy Notice
In strict accordance with the challenge guidelines:
* Class recordings are processed in ephemeral memory and discarded immediately upon analysis.
* No personal child identifiable information or faces are stored, published, or transmitted to third-party databases.
