from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field


class ScoreDetail(BaseModel):
    score: int = Field(..., ge=1, le=10, description="Score on a scale from 1 to 10")
    rubric_level: Literal["Poor", "Average", "Good", "Excellent"] = Field(
        ..., description="Qualitative level matching the score"
    )
    summary: str = Field(..., description="1-2 sentences summarizing performance on this criterion")
    evidence: List[str] = Field(default_factory=list, description="Specific observed evidence from the class")


class Observation(BaseModel):
    timestamp: str = Field(..., description="Timestamp in MM:SS format, e.g. '04:12'")
    category: Literal[
        "Student Engagement",
        "Instructional Scaffolding",
        "Praise & Tone",
        "Screen & Visual Utility",
        "Classroom Pacing",
        "Other"
    ]
    observation_type: Literal["Strength", "Improvement Area", "Notable Event"]
    evidence: str = Field(
        ...,
        description="Specific observation grounded in what was said or shown on screen (avoid generic comments)"
    )


class TelemetrySummary(BaseModel):
    duration_seconds: float = Field(default=0.0)
    silence_ratio_pct: float = Field(default=0.0, description="Percentage of class duration with silence/dead air")
    dead_air_incidents: int = Field(default=0, description="Instances where silence exceeded 8 seconds")
    visual_change_rate: float = Field(default=0.0, description="Estimated slide/screen changes per minute")
    estimated_talk_pacing_wpm: Optional[float] = Field(default=None, description="Estimated words per minute")


class EvaluationReport(BaseModel):
    overall_rating: Literal["Poor", "Average", "Good", "Excellent"]
    overall_score: int = Field(..., ge=0, le=100, description="Calculated composite quality score out of 100")
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    executive_summary: str = Field(
        ...,
        description="Comprehensive yet concise executive summary of class quality and reasoning"
    )
    
    # 5 Key Pedagogical Dimensions
    student_engagement: ScoreDetail = Field(
        ...,
        description="Turn-taking, student participation opportunities, open vs closed questioning"
    )
    instructional_clarity: ScoreDetail = Field(
        ...,
        description="Concept breakdown, scaffolding, checking for comprehension"
    )
    emotional_climate: ScoreDetail = Field(
        ...,
        description="Patience, encouraging tone, constructive error handling, genuine praise"
    )
    visual_and_screen_utility: ScoreDetail = Field(
        ...,
        description="Readability of slides, relevance of screen share, visual engagement"
    )
    pacing_and_flow: ScoreDetail = Field(
        ...,
        description="Transitions, management of dead air, maintaining child attention span"
    )

    specific_observations: List[Observation] = Field(
        default_factory=list,
        description="Granular timestamped observations of what the teacher/student said or did"
    )
    
    top_strengths: List[str] = Field(default_factory=list)
    critical_weaknesses: List[str] = Field(default_factory=list)
    actionable_coaching_tips: List[str] = Field(
        default_factory=list,
        description="Concrete, practical coaching advice to help this teacher improve"
    )
    telemetry: Optional[TelemetrySummary] = None
