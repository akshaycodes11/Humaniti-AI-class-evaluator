# AI Interaction & Engineering Log

**Project**: EduGrade AI — Automated Quality Assessment for 8x8 Virtual Classrooms  
**Author / Developer**: Recruitment Challenge Submission (Humaniti AI)  
**Tools & Assistants Used**: Google Antigravity AI Pair Programmer, Gemini 2.0 / 1.5 Flash  

---

## 1. Problem Formulation & Architectural Discovery Prompt

### Objective
Design an end-to-end, production-grade multimodal pipeline capable of evaluating 8x8 kids' online classes into 4 tiers (*Poor*, *Average*, *Good*, *Excellent*) with timestamped evidence, under severe latency (< 15s) and cost ($0.00) constraints.

### Core Architectural Decisions
* **Decision 1: Edge Acoustic & Visual Telemetry First**  
  * *Rationale*: Feeding 45 minutes of high-resolution video into multimodal LLMs costs $1.50–$3.50 per run and takes 60–120s. Instead, OpenCV and SciPy extract silence intervals, dead-air occurrences (> 7s), and screen transition dynamics in < 3s with zero cost.
* **Decision 2: Multimodal Grounding via Gemini Flash Free Tier**  
  * *Rationale*: Sends extracted keyframes (1 frame / 45s) and acoustic telemetry in a single prompt to Google Gemini Flash (15 RPM / 1,500 RPD free tier via Google AI Studio).
* **Decision 3: Zero-Fail Local Deterministic Engine**  
  * *Rationale*: Guarantees that if an evaluator tests the app without an API key or offline, the system never throws an unhandled exception and yields calibrated scores based on real physical signals.

---

## 2. Key Prompts & Pedagogical System Instructions

### Master Pedagogical Evaluator Prompt
```text
You are an expert pedagogical evaluator and Master Teacher Coach specializing in assessing online K-12 and children's classes (such as 8x8 virtual classrooms).

Your task is to conduct a thorough, rigorous, evidence-based quality assessment of a class recording.
You must classify the overall class quality into exactly one of four levels:
- "Poor": Significant pedagogical deficiencies; teacher monologues without checking for comprehension, excessive dead air or confusion, lack of positive encouragement, static or confusing visual aids.
- "Average": Basic instructional delivery, but largely teacher-centered; minimal interactive questioning, generic praise ("okay", "yes"), pacing is sluggish or uneven.
- "Good": Clear learning objectives, structured scaffolding, regular checks for understanding, supportive tone, appropriate use of screen and visual materials.
- "Excellent": Highly dynamic, engaging, and child-centered; student does substantial thinking/speaking; teacher provides targeted, specific praise ("great job borrowing from the tens place!"), adapts to student hesitations, transitions seamlessly with visual cues.

CRITICAL INSTRUCTION ON EVIDENCE:
- DO NOT produce generic, vague statements like "The teacher was good" or "Good communication".
- Provide specific, concrete observations with timestamps wherever possible (e.g. "[03:45] Teacher waited 8 seconds patiently for student to read the sentence, then praised their phonics accuracy").
- Pay close attention to what happens on the screen (slides, exercises, diagrams) AND how the teacher interacts with the child.
```

---

## 3. Human Ground-Truth Calibration (feedbacks.pdf Analysis)

Upon downloading and examining `dataset_samples/feedbacks.pdf` from the provided challenge Google Drive, the exact signals human evaluators look for were extracted:
1. **Rapport & Welcome**: Welcoming students, asking for introductions.
2. **One-to-One Addressing**: Calling students individually by name, asking individual questions during puzzle solving.
3. **Praise & Energy**: *"need more energetic communication and more pleasant interaction"*, *"need a better happy interaction with the kids"*.
4. **Homework (HW) Closure**: Verifying whether homework/practice was explicitly assigned before disconnect.
5. **Acoustic Dead Air**: Pauses exceeding 7 seconds indicate disengagement or waiting.

### Calibration Result:
* Tested on actual challenge recording: **B435 Intermediate (Sumit Ranjan)**
* Ground truth rating from human evaluators: **Average**
* EduGrade AI predicted rating: **Average (62/100)**
* Evidence detected:
  * `[02:17]` Dead-air pause (14.0s)
  * `[00:01]` Established greeting/rapport
  * `[37:00]` Homework (HW) omitted
  * Praise frequency: Only 3 praise cues across entire session
