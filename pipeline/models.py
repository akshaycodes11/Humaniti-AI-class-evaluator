from typing import List, Literal, Optional, Dict, Any
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

    # Audio Glitch & Quality Health Telemetry
    audio_health_score: int = Field(default=92, ge=0, le=100, description="Overall audio fidelity and stability score (0-100)")
    audio_health_status: str = Field(default="Clean & Stable", description="Audio quality classification (Clean, Minor Glitches, Degraded)")
    audio_glitches_count: int = Field(default=0, description="Total detected audio glitches (clipping, dropouts, pops)")
    clipping_ratio_pct: float = Field(default=0.0, description="Percentage of audio with digital clipping/distortion")
    choppy_dropouts_count: int = Field(default=0, description="Count of micro-dropouts caused by network packet loss")
    estimated_snr_db: float = Field(default=22.0, description="Estimated Signal-to-Noise Ratio in decibels")
    audio_glitch_events: List[Dict[str, Any]] = Field(default_factory=list, description="Timestamped log of specific audio glitch incidents")

    # Pedagogical Wait Time (Think-Time) & Bloom's Taxonomy Question Depth
    avg_wait_time_seconds: Optional[float] = Field(default=3.2, description="Mean wait time after asking questions")
    median_wait_time_seconds: Optional[float] = Field(default=2.8, description="Median wait time (think time) in seconds")
    deep_think_time_count: Optional[int] = Field(default=0, description="Questions where teacher provided >= 2.5s think time")
    self_interrupted_questions_count: Optional[int] = Field(default=0, description="Questions interrupted by teacher before student answered")
    wait_time_effectiveness_score: Optional[int] = Field(default=85, ge=0, le=100, description="Pedagogical think-time score (0-100)")
    open_ended_questions_count: Optional[int] = Field(default=0, description="Higher-order inquiry questions asked (why, how, what if)")
    closed_questions_count: Optional[int] = Field(default=0, description="Procedural or recall closed questions")
    higher_order_ratio_pct: Optional[float] = Field(default=0.0, description="Percentage of questions that require higher-order thinking")
    bloom_categories: Optional[Dict[str, int]] = Field(default_factory=dict, description="Counts by Bloom's Taxonomy tier")
    sample_inquiry_questions: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Sample timestamped inquiry questions")



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
