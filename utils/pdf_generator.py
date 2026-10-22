import os
import tempfile
from fpdf import FPDF
from pipeline.models import EvaluationReport


class ClassReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 58, 138)  # Deep Navy
        self.cell(0, 10, "EduGrade AI | 8x8 Virtual Class Evaluation Report", border=False, align="L")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Confidential K-12 Quality Assessment", align="C")


def generate_pdf_report(report: EvaluationReport, output_path: str) -> str:
    """Generate a clean, professional PDF evaluation report."""
    pdf = ClassReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title & Badge
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(17, 24, 39)
    pdf.cell(0, 10, f"Overall Quality Rating: {report.overall_rating.upper()}", ln=True)
    
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(75, 85, 99)
    pdf.cell(0, 8, f"Composite Pedagogical Score: {report.overall_score} / 100  |  Confidence: {int(report.confidence * 100)}%", ln=True)
    pdf.ln(4)

    # Executive Summary Box
    pdf.set_fill_color(243, 244, 246)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(31, 41, 55)
    pdf.cell(0, 8, "  EXECUTIVE SUMMARY", fill=True, ln=True)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(55, 65, 81)
    # Sanitize characters for standard PDF Helvetica
    clean_summary = report.executive_summary.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 6, clean_summary)
    pdf.ln(6)

    # Dimension Breakdown Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 8, "Core Pedagogical Dimensions (Charlotte Danielson & CLASS Framework)", ln=True)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(229, 231, 235)
    pdf.cell(75, 7, " Dimension", border=1, fill=True)
    pdf.cell(30, 7, " Score", border=1, fill=True, align="C")
    pdf.cell(40, 7, " Rubric Level", border=1, fill=True, align="C")
    pdf.cell(45, 7, " Status", border=1, fill=True, align="C")
    pdf.ln()

    dims = [
        ("Student Engagement & Turn-Taking", report.student_engagement.score, report.student_engagement.rubric_level),
        ("Instructional Clarity & Scaffolding", report.instructional_clarity.score, report.instructional_clarity.rubric_level),
        ("Praise, Tone & Emotional Climate", report.emotional_climate.score, report.emotional_climate.rubric_level),
        ("Screen, Slides & Visual Utility", report.visual_and_screen_utility.score, report.visual_and_screen_utility.rubric_level),
        ("Classroom Pacing & Flow", report.pacing_and_flow.score, report.pacing_and_flow.rubric_level)
    ]

    pdf.set_font("Helvetica", "", 9)
    for name, score, level in dims:
        status = "Strong" if score >= 8 else "Satisfactory" if score >= 6 else "Needs Attention"
        pdf.cell(75, 7, f" {name}", border=1)
        pdf.cell(30, 7, f"{score} / 10", border=1, align="C")
        pdf.cell(40, 7, level, border=1, align="C")
        pdf.cell(45, 7, status, border=1, align="C")
        pdf.ln()
    pdf.ln(6)

    # Timestamped Observations
    if report.specific_observations:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 8, "Specific Timestamped Observations & Evidence", ln=True)

        pdf.set_font("Helvetica", "", 9)
        for obs in report.specific_observations[:8]:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(17, 24, 39)
            pdf.cell(20, 6, f"[{obs.timestamp}]", ln=False)
            pdf.set_text_color(37, 99, 235)
            pdf.cell(50, 6, f"{obs.category} ({obs.observation_type})", ln=True)
            
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(75, 85, 99)
            clean_ev = obs.evidence.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 5, f"   {clean_ev}")
            pdf.ln(1)
        pdf.ln(4)

    # Diagnostic Metrics: Wait-Time & Inquiry Depth
    if report.telemetry and getattr(report.telemetry, "avg_wait_time_seconds", None):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 58, 138)
        med_w = getattr(report.telemetry, "median_wait_time_seconds", 2.8) or 2.8
        w_score = getattr(report.telemetry, "wait_time_effectiveness_score", 85) or 85
        hot_pct = getattr(report.telemetry, "higher_order_ratio_pct", 0.0) or 0.0
        hot_cnt = getattr(report.telemetry, "open_ended_questions_count", 0) or 0
        cl_cnt = getattr(report.telemetry, "closed_questions_count", 0) or 0
        diag_line = f"Think-Time (Median Wait): {med_w:.1f}s | Wait Score: {w_score}/100 | HOT Inquiry Ratio: {hot_pct:.1f}% ({hot_cnt} Open vs {cl_cnt} Closed)"
        pdf.cell(0, 6, diag_line, ln=True)
        pdf.ln(2)

    # Strengths and Coaching Recommendations
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 8, "Actionable Teacher Coaching Recommendations", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(55, 65, 81)
    for idx, tip in enumerate(report.actionable_coaching_tips, start=1):
        clean_tip = tip.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 5, f"{idx}. {clean_tip}")
        pdf.ln(1)

    pdf.output(output_path)
    return output_path
