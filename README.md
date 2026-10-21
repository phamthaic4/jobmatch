# JobMatch

CV–JD skill gap analyzer. Paste a CV and a job description, the API extracts the IT skills from both with a **custom fine-tuned PhoBERT NER model**, compares them, and (coming in Week 7) turns the gaps into interview questions with Gemini.

> **Status:** v0.1.0 — Week 6 checkpoint: `POST /extract-skills` + `GET /health` working, ONNX inference at **12.2 ms** per sentence on CPU.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX--Runtime-purple.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/phamthaic4/jobmatch/actions/workflows/ci.yml/badge.svg)](https://github.com/phamthaic4/jobmatch/actions/workflows/ci.yml)

## The model is mine, not a hardcoded list

The skill extractor is the PhoBERT model trained in
[vietnamese-job-skill-ner](https://github.com/phamthaic4/vietnamese-job-skill-ner) —
350 hand-labelled Vietnamese job-description sentences, entity-level **test F1 0.6711**
(`seqeval`), with a full error analysis. This repo serves it over HTTP.

```
┌──────────────┐   POST /extract-skills   ┌──────────────┐   tokenize + decode   ┌──────────────┐
│  CV / JD     │ ───────────────────────► │  FastAPI     │ ────────────────────► │  PhoBERT NER │
│  text        │ ◄─────────────────────── │  backend     │ ◄──────────────────── │  (ONNX)      │
└──────────────┘   { skills, spans }       └──────────────┘   spans + offsets     └──────────────┘
```

## Quick start

```bash
git clone https://github.com/phamthaic4/jobmatch.git
cd jobmatch

python -m venv .venv && source .venv/Scripts/activate   # Windows
pip install -r requirements.txt

uvicorn app.main:app --reload
```

The checkpoint (~513 MB) is downloaded automatically from the Proj 1 release on first start and verified by SHA-256. Then open <http://127.0.0.1:8000/docs> for the Swagger UI.

```bash
curl -X POST http://127.0.0.1:8000/extract-skills \
  -H "Content-Type: application/json" \
  -d '{"text": "Ứng tuyển Backend Developer. Yêu cầu 3 năm kinh nghiệm với Python, FastAPI, PostgreSQL, Docker và Kubernetes."}'
```

```json
{
  "skills": ["python", "fastapi", "postgresql", "docker", "kubernetes"],
  "spans": [{"skill": "python", "start": 59, "end": 65}, "..."],
  "count": 5,
  "latency_ms": 20.55,
  "model": "vinai/phobert-base (fine-tuned)",
  "backend": "onnx"
}
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe + model status |
| `POST` | `/extract-skills` | Extract skills from a CV / job description |
| `GET` | `/docs` | Swagger UI |

`POST /extract-skills` takes `{"text": str, "normalize": bool}` and returns the skill list, the character spans of every skill in the source text, and the inference latency.

## Inference backends

The same weights can be served two ways — `JOBMATCH_MODEL_BACKEND=onnx|transformers`.

| Backend | Short sentence | Full JD (62 tokens) | Notes |
|---|---|---|---|
| **ONNX Runtime** | **12.2 ms** | 37.3 ms | default — 2.5× faster than PyTorch on CPU |
| transformers (PyTorch) | 30.5 ms | 54.3 ms | easier to debug, higher memory |

Reproduce with `python scripts/benchmark.py`. The short-sentence numbers match Proj 1's published benchmark (ONNX 12.55 ms, PyTorch 35.39 ms), which confirms this service runs the same model in the same way.

## Skill normalisation

The model returns the casing used in the source text, so `ReactJS`, `reactjs` and `K8s` would look like three different skills. `app/services/normalizer.py` canonicalises them through a synonym map:

| In the text | Normalised |
|---|---|
| `ReactJS` | `react` |
| `NodeJS` | `node.js` |
| `K8s` | `kubernetes` |
| `Postgres` | `postgresql` |
| `Golang` | `go` |
| `RESTful API` | `rest api` |
| `C++` | `c/c++` |

Set `"normalize": false` to get the raw model output.

## Layout

```
jobmatch/
├── app/
│   ├── main.py              # FastAPI app, /health, CORS
│   ├── config.py            # env-driven settings
│   ├── schemas.py           # Pydantic v2 request/response models
│   ├── dependencies.py      # model singleton
│   ├── routers/skills.py    # /extract-skills
│   └── services/
│       ├── ner_model.py     # checkpoint bootstrap + inference (ONNX / PyTorch)
│       └── normalizer.py    # synonym map
├── models/phobert-ner/       # checkpoint (gitignored — auto-downloaded)
├── scripts/benchmark.py     # latency benchmark
├── tests/                   # 28 tests: normaliser unit + API end-to-end
└── .github/workflows/       # CI: lint + unit tests + API tests with weights
```

## Tests

```bash
python -m pytest              # everything (downloads the weights on first run)
python -m pytest -m "not slow"  # only the tests that need no weights
ruff check app scripts tests
```

CI runs all three: lint, unit tests, and the API tests with the real checkpoint pulled from the Proj 1 release.

## Roadmap

- [x] **Week 6** — FastAPI skeleton, model integration, `POST /extract-skills`, `GET /health`
- [ ] **Week 7** — `POST /match` (CV skills vs JD skills → matched / missing / extra) and `POST /interview-prep` (Gemini)
- [ ] **Week 8** — Streamlit frontend
- [ ] **Week 9** — Dockerfile + docker-compose + pytest API suite
- [ ] **Week 10** — deploy (backend + frontend), public demo URL

## Related project

- **[vietnamese-job-skill-ner](https://github.com/phamthaic4/vietnamese-job-skill-ner)** — the model this API serves: dataset, training, ablations (PhoBERT vs XLM-R, LR sweep), OOV analysis and error breakdown.
