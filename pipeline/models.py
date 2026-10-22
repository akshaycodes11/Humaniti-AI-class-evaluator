from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field, field_validator


class ScoreDetail(BaseModel):
    score: int = Field(default=5, ge=1, le=10, description="Score on a scale from 1 to 10")
    rubric_level: Literal["Poor", "Average", "Good", "Excellent"] = Field(
        default="Average", description="Qualitative level matching the score"
    )
    summary: str = Field(default="", description="1-2 sentences summarizing performance on this criterion")
    evidence: List[str] = Field(default_factory=list, description="Specific observed evidence from the class")

    @field_validator("rubric_level", mode="before")
    @classmethod
    def normalize_rubric_level(cls, v):
        if not isinstance(v, str):
            return "Average"
        s = v.strip().lower()
        if "poor" in s or "weak" in s:
            return "Poor"
        if "good" in s:
            return "Good"
        if "excel" in s:
            return "Excellent"
        return "Average"


class Observation(BaseModel):
    timestamp: str = Field(default="00:00", description="Timestamp in MM:SS format, e.g. '04:12'")
    category: str = Field(
        default="Other",
        description="Category such as Student Engagement, Instructional Scaffolding, Praise & Tone, Screen & Visual Utility, Classroom Pacing, or Other"
    )
    observation_type: str = Field(
        default="Notable Event",
        description="Observation classification: Strength, Improvement Area, or Notable Event"
    )
    evidence: str = Field(
        default="",
        description="Specific observation grounded in what was said or shown on screen (avoid generic comments)"
    )

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v):
        if not isinstance(v, str):
            return "Other"
        s = v.strip().lower()
        if any(k in s for k in ["emotional", "climate", "praise", "tone"]):
            return "Praise & Tone"
        if "engagement" in s or "participat" in s or "interact" in s:
            return "Student Engagement"
        if any(k in s for k in ["scaffold", "instruction", "clarity", "concept", "explain"]):
            return "Instructional Scaffolding"
        if any(k in s for k in ["screen", "visual", "slide", "board", "utility"]):
            return "Screen & Visual Utility"
        if any(k in s for k in ["pacing", "flow", "silence", "dead air", "wait"]):
            return "Classroom Pacing"
        return v.strip().title() or "Other"

    @field_validator("observation_type", mode="before")
    @classmethod
    def normalize_observation_type(cls, v):
        if not isinstance(v, str):
            return "Notable Event"
        s = v.strip().lower()
        if "strength" in s or "positive" in s or "good" in s:
            return "Strength"
        if any(k in s for k in ["improv", "weak", "concern", "area", "delta", "need"]):
            return "Improvement Area"
        return "Notable Event"


class TelemetrySummary(BaseModel):
    duration_seconds: float = Field(default=0.0)
    silence_ratio_pct: float = Field(default=0.0, description="Percentage of class duration with silence/dead air")
    dead_air_incidents: int = Field(default=0, description="Instances where silence exceeded 7 seconds")
    visual_change_rate: float = Field(default=0.0, description="Estimated slide/screen changes per minute")
    estimated_talk_pacing_wpm: Optional[float] = Field(default=None, description="Estimated words per minute")
    vocal_turns_count: Optional[int] = Field(default=None, description="Total count of speaker turns / speech bursts")
    turn_taking_rate_per_min: Optional[float] = Field(default=None, description="Turn-taking interactions per minute")
    teacher_talk_ratio_pct: Optional[float] = Field(default=None, description="Dominant teacher talk percentage")
    student_talk_ratio_pct: Optional[float] = Field(default=None, description="Student talk percentage")
    longest_monologue_seconds: Optional[float] = Field(default=None, description="Longest continuous speech monologue without pause")
    avg_speech_burst_seconds: Optional[float] = Field(default=None, description="Average speech burst duration")
    questions_count: Optional[int] = Field(default=None, description="Total questions / prompts detected")
    praise_count: Optional[int] = Field(default=None, description="Positive reinforcement / praise cues detected")
    homework_mentioned: Optional[bool] = Field(default=None, description="Whether homework or practice was assigned")



class EvaluationReport(BaseModel):
    overall_rating: Literal["Poor", "Average", "Good", "Excellent"] = Field(default="Average")
    overall_score: int = Field(default=50, ge=0, le=100, description="Calculated composite quality score out of 100")
    confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    executive_summary: str = Field(
        default="",
        description="Comprehensive yet concise executive summary of class quality and reasoning"
    )

    @field_validator("overall_rating", mode="before")
    @classmethod
    def normalize_overall_rating(cls, v):
        if not isinstance(v, str):
            return "Average"
        s = v.strip().lower()
        if "poor" in s or "weak" in s:
            return "Poor"
        if "good" in s:
            return "Good"
        if "excel" in s:
            return "Excellent"
        return "Average"
    
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
