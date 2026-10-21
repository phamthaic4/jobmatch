"""Where does the time go? Split tokenization / inference / post-processing.

    python scripts/profile_inference.py

Useful when the API feels slow: it shows whether the bottleneck is the slow
PhoBERT tokenizer, the model forward pass, or the span decoding.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.ner_model import SkillExtractor  # noqa: E402

MODEL_DIR = ROOT / "models" / "phobert-ner"

TEXT = (
    "Thành thạo ít nhất 1 ngôn ngữ Backend: Ưu tiên Python, Go, C/C++ hoặc Java (Spring Boot). "
    "Có kinh nghiệm làm việc với hệ quản trị cơ sở dữ liệu PostgreSQL, MySQL, Redis và "
    "triển khai Docker, Kubernetes, CI/CD."
)

N_RUNS = 30


def timed(fn, runs: int = N_RUNS) -> float:
    for _ in range(3):
        fn()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - start) / runs * 1000


def main() -> None:
    extractor = SkillExtractor(MODEL_DIR, backend="onnx")
    words, input_ids, _, _ = extractor._tokenize(TEXT)

    print(f"{len(words)} words -> {len(input_ids)} subword ids")
    print(f"  tokenize (cached) : {timed(lambda: extractor._tokenize(TEXT)):6.2f} ms")
    print(f"  onnx forward      : {timed(lambda: extractor._predict(input_ids, [1] * len(input_ids))):6.2f} ms")
    print(f"  full extract      : {timed(lambda: extractor.extract(TEXT)):6.2f} ms")


if __name__ == "__main__":
    main()
