"""Skill name normalisation and a small synonym map.

The NER model returns whatever casing the source text used ("JavaScript",
"JAVASCRIPT", "ReactJS"). Matching a CV against a job description needs a
canonical form, otherwise "React" and "ReactJS" look like different skills.
"""

import re
from collections.abc import Iterable

# alias -> canonical name
SYNONYMS: dict[str, str] = {
    "js": "javascript",
    "java script": "javascript",
    "javascript": "javascript",
    "nodejs": "node.js",
    "node js": "node.js",
    "node": "node.js",
    "reactjs": "react",
    "react js": "react",
    "vuejs": "vue",
    "vue js": "vue",
    "angularjs": "angular",
    "golang": "go",
    "py": "python",
    "python3": "python",
    "postgres": "postgresql",
    "psql": "postgresql",
    "k8s": "kubernetes",
    "kube": "kubernetes",
    "docker-compose": "docker compose",
    "ci/cd": "ci/cd",
    "ci-cd": "ci/cd",
    "continuous integration": "ci/cd",
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "tf": "tensorflow",
    "sklearn": "scikit-learn",
    "gitlab ci": "gitlab ci",
    "restful": "rest api",
    "restful api": "rest api",
    "rest api": "rest api",
    "net": ".net",
    ".net": ".net",
    "dotnet": ".net",
    "c sharp": "c#",
    "c#": "c#",
    "cpp": "c/c++",
    "c++": "c/c++",
    "c/c++": "c/c++",
    "objective c": "objective-c",
    "amazon web services": "aws",
    "google cloud platform": "gcp",
    "microsoft azure": "azure",
    "postgres sql": "postgresql",
    "no sql": "nosql",
}

_WHITESPACE = re.compile(r"\s+")


def normalize_skill(skill: str) -> str:
    """Canonicalise a single skill name."""
    cleaned = _WHITESPACE.sub(" ", skill.strip()).strip(" \t\n\r,;:()[]{}\"'")
    if not cleaned:
        return ""
    key = cleaned.lower()
    return SYNONYMS.get(key, key)


def normalize_skills(skills: Iterable[str]) -> list[str]:
    """Canonicalise a list of skills, dropping empties and duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for skill in skills:
        normalized = normalize_skill(skill)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
