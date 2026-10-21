"""Skill extraction endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from ..config import settings
from ..dependencies import get_extractor
from ..schemas import ExtractSkillsRequest, ExtractSkillsResponse, SkillSpan
from ..services.ner_model import ModelUnavailableError, SkillExtractor
from ..services.normalizer import normalize_skill

router = APIRouter(tags=["skills"])


@router.post(
    "/extract-skills",
    response_model=ExtractSkillsResponse,
    summary="Extract IT skills from a job description or CV",
)
def extract_skills(
    payload: ExtractSkillsRequest,
    extractor: SkillExtractor = Depends(get_extractor),
) -> ExtractSkillsResponse:
    """Run the fine-tuned PhoBERT NER model over `text`.

    The model is the one trained in
    [vietnamese-job-skill-ner](https://github.com/phamthaic4/vietnamese-job-skill-ner)
    (test F1 0.6711, entity-level).
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="text must not be empty",
        )
    if len(text) > settings.max_text_length:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"text too long (max {settings.max_text_length} characters)",
        )

    try:
        spans, latency_ms = extractor.extract_with_latency(text)
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    skill_names = [
        normalize_skill(span.skill) if payload.normalize else span.skill for span in spans
    ]
    spans = [
        SkillSpan(skill=name, start=span.start, end=span.end)
        for name, span in zip(skill_names, spans)
    ]

    skills = [span.skill for span in spans]
    return ExtractSkillsResponse(
        skills=skills,
        spans=spans,
        count=len(skills),
        latency_ms=round(latency_ms, 2),
        model="vinai/phobert-base (fine-tuned)",
        backend=extractor.backend,
    )
