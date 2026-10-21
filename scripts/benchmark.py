"""Benchmark the extraction pipeline (ONNX vs transformers) on CPU.

    python scripts/benchmark.py

The short sentence is the one used to produce the Proj 1 numbers
(ONNX 12.55 ms vs PyTorch 35.39 ms), so the two projects stay comparable.
"""

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.ner_model import SkillExtractor  # noqa: E402

MODEL_DIR = ROOT / "models" / "phobert-ner"

SHORT = "Yêu cầu 2 năm kinh nghiệm với Python, Docker và Kubernetes."
JD = (
    "Thành thạo ít nhất 1 ngôn ngữ Backend: Ưu tiên Python, Go, C/C++ hoặc Java (Spring Boot). "
    "Có kinh nghiệm làm việc với hệ quản trị cơ sở dữ liệu PostgreSQL, MySQL, Redis và "
    "triển khai Docker, Kubernetes, CI/CD."
)

N_RUNS = 50
N_WARMUP = 5


def bench(extractor: SkillExtractor, text: str) -> tuple[float, int]:
    for _ in range(N_WARMUP):
        extractor.extract(text)

    samples = []
    count = 0
    for _ in range(N_RUNS):
        spans, ms = extractor.extract_with_latency(text)
        samples.append(ms)
        count = len(spans)
    return statistics.median(samples), count


def main() -> None:
    results: dict[str, dict[str, float]] = {}
    for backend in ("onnx", "transformers"):
        extractor = SkillExtractor(MODEL_DIR, backend=backend)
        short_ms, short_n = bench(extractor, SHORT)
        jd_ms, jd_n = bench(extractor, JD)
        results[backend] = {
            "short_ms": short_ms,
            "jd_ms": jd_ms,
            "short_n": short_n,
            "jd_n": jd_n,
        }
        print(f"[{backend}]")
        print(f"  short sentence : {short_ms:6.2f} ms  ({short_n} skills)")
        print(f"  JD sentence    : {jd_ms:6.2f} ms  ({jd_n} skills)")

    onnx = results["onnx"]["short_ms"]
    torch = results["transformers"]["short_ms"]
    print(f"\nSpeedup on the short sentence: {torch / onnx:.2f}x (ONNX vs PyTorch)")
    print("Proj 1 reference: ONNX 12.55 ms, PyTorch 35.39 ms -> 2.82x")


if __name__ == "__main__":
    main()
