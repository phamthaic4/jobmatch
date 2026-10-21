"""Pydantic request/response schemas."""

from pydantic import BaseModel, Field


class ExtractSkillsRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="Job description or CV text to extract skills from.",
        examples=["Yêu cầu 2 năm kinh nghiệm với Python, Docker và Kubernetes."],
    )
    normalize: bool = Field(
        True,
        description="Lower-case and canonicalise skill names (JavaScript, javascript, JS -> javascript).",
    )


class SkillSpan(BaseModel):
    skill: str
    start: int = Field(..., description="Character offset of the skill in the input text.")
    end: int = Field(..., description="Character offset just past the skill.")


class ExtractSkillsResponse(BaseModel):
    skills: list[str]
    spans: list[SkillSpan]
    count: int
    latency_ms: float
    model: str
    backend: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str
    model_dir: str
    version: str
