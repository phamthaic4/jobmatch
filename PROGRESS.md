# Progress Log — JobMatch

> Project 2 of the intern portfolio. Backend for the CV–JD skill gap analyzer.
> Model: [vietnamese-job-skill-ner](https://github.com/phamthaic4/vietnamese-job-skill-ner) (v1.0.0).

## Week 6 — 2026-10-21 (Backend skeleton + model integration)

### Done
- [x] Repo `jobmatch` (public) — <https://github.com/phamthaic4/jobmatch>
- [x] `app/config.py` — settings đọc từ env (`JOBMATCH_MODEL_BACKEND`, `JOBMATCH_MODEL_DIR`, …)
- [x] `app/schemas.py` — Pydantic v2 (`ExtractSkillsRequest` / `ExtractSkillsResponse` / `SkillSpan` / `HealthResponse`)
- [x] `app/services/ner_model.py` — **tự tải checkpoint từ release của Proj 1** (verify sha256) + inference 2 backend
- [x] `app/services/normalizer.py` — synonym map (`ReactJS`→`react`, `K8s`→`kubernetes`, `C++`→`c/c++`…)
- [x] `app/routers/skills.py` — `POST /extract-skills`
- [x] `app/main.py` — FastAPI app, `GET /health`, CORS, warm model lúc startup
- [x] `scripts/benchmark.py` + `scripts/profile_inference.py`
- [x] `tests/` — 28 test (normaliser unit + API end-to-end), ruff config trong `pyproject.toml`
- [x] `.github/workflows/ci.yml` — lint + unit test + API test (tải weights từ release Proj 1) → **CI xanh**
- [x] README + LICENSE + `.env.example` + `models/README.md`

### Kết quả đo được
| Backend | Câu ngắn (28 token) | Câu JD đầy đủ (69 token) |
|---|---|---|
| ONNX Runtime | **12.2 ms** | 37.3 ms |
| transformers (PyTorch) | 30.5 ms | 54.3 ms |

Số của Proj 1 (ONNX 12.55 ms / PyTorch 35.39 ms) → **service này chạy đúng model đó, đúng cách đó**.

### Bài học
- `lru_cache` trên **method** → ruff B019 (giữ instance sống mãi). Chuyển sang dict trên instance.
- ONNX **chậm hơn** PyTorch khi set `intra_op_num_threads=1` (94 ms vs 47 ms). Bỏ option → ONNX 12.2 ms.
- PhoBERT **chỉ có slow tokenizer** → `word_ids()` không dùng được (transformers v5). Phải tokenize từng từ + tự track word index. Cache `word → ids` giúp tokenizer chỉ tốn 0.04 ms.
- `type(span)(...)` trong router tạo instance của **dataclass**, không phải Pydantic model → validation error. Phải import đúng class `SkillSpan` từ `schemas`.
- Ruff bắt được **2 dict key trùng** trong synonym map (`"c sharp"`, `"node js"`) — lỗi im lặng, key sau ghi đè key trước.
- `Invoke-RestMethod` với body tiếng Việt phải encode **UTF-8 bytes** + `charset=utf-8`, nếu không JSON parse fail.

### Next week (Week 7)
- [ ] `POST /match` — CV skills vs JD skills → `{ matched, missing, extra }`
- [ ] `POST /interview-prep` — Gemini sinh 5 câu hỏi từ skill thiếu
- [ ] `app/prompts/interview.txt` — prompt template
- [ ] `.env` + error handling khi thiếu `GOOGLE_API_KEY`
