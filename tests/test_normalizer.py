"""Unit tests for the skill normaliser (no model needed)."""

import pytest

from app.services.normalizer import normalize_skill, normalize_skills


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Python", "python"),
        ("  Docker  ", "docker"),
        ("CI/CD", "ci/cd"),
        ("C/C++", "c/c++"),
        ("C#", "c#"),
        (".NET", ".net"),
        ("Spring Boot", "spring boot"),
    ],
)
def test_normalize_keeps_technical_symbols(raw, expected):
    assert normalize_skill(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("JS", "javascript"),
        ("JavaScript", "javascript"),
        ("NodeJS", "node.js"),
        ("ReactJS", "react"),
        ("VueJS", "vue"),
        ("K8s", "kubernetes"),
        ("Postgres", "postgresql"),
        ("Golang", "go"),
        ("Py", "python"),
        ("RESTful API", "rest api"),
        ("DotNet", ".net"),
        ("C++", "c/c++"),
        ("AWS", "aws"),
    ],
)
def test_synonyms_are_canonicalised(raw, expected):
    assert normalize_skill(raw) == expected


def test_normalize_skills_drops_duplicates_and_empties():
    assert normalize_skills(["Python", "python", "  ", "Docker"]) == ["python", "docker"]


def test_normalize_skills_preserves_order():
    raw = ["Docker", "Python", "Kubernetes"]
    assert normalize_skills(raw) == ["docker", "python", "kubernetes"]
