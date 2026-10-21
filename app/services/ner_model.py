"""PhoBERT skill NER — checkpoint bootstrap and inference.

The checkpoint is the fine-tuned model from Proj 1
(https://github.com/phamthaic4/vietnamese-job-skill-ner) and is fetched from
its v1.0.0 release on first use, then cached under `models/phobert-ner/`.

Two backends are supported:
  * ``onnx``        — ONNX Runtime, ~12.5 ms per call on CPU (default)
  * ``transformers`` — plain PyTorch, ~35 ms per call on CPU

The tokenization + post-processing rules are ported from Proj 1
(`src/predict.py`) so both projects extract exactly the same spans.
"""

import hashlib
import re
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

# Keeps technical symbols attached to their word: C/C++, CI/CD, C#, .NET
TOKEN_RE = re.compile(r"(?:[^\W_]+[./+#_]*)+|_+|[^\w\s]+", re.UNICODE)

MAX_LENGTH = 128

RELEASE_FILES = {
    "model.onnx": "a0548653873512d3750033ea2d8089f3f6bf761d0d1f1b83ef3ed248d8ec06ed",
    "model.safetensors": "f047aea77ab031a4e8d826da8a379886b571d755156177daea9c888611557db2",
    "tokenizer.zip": "59c747474ae5e17eb7ef96d59e6867d3178f5a8ccfd3510e7fc5adbe245801a4",
}

TOKENIZER_MEMBERS = (
    "config.json",
    "tokenizer_config.json",
    "vocab.txt",
    "bpe.codes",
    "added_tokens.json",
)


class ModelUnavailableError(RuntimeError):
    """Raised when the checkpoint is missing and cannot be downloaded."""


@dataclass(frozen=True)
class SkillSpan:
    skill: str
    start: int
    end: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Download to a .part file first, so an interrupted download never looks complete."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        total = int(response.headers.get("Content-Length", 0) or 0)
        downloaded = 0
        last_report = 0.0
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now - last_report > 2:
                percent = f"{downloaded / total * 100:5.1f}%" if total else "  ?  %"
                print(
                    f"    {percent}  {downloaded / 1e6:,.1f} MB"
                    + (f" / {total / 1e6:,.1f} MB" if total else ""),
                    flush=True,
                )
                last_report = now
    tmp.replace(dest)


def ensure_model(
    model_dir: Path,
    backend: str = "onnx",
    release_base: str = "",
    allow_download: bool = True,
) -> Path:
    """Make sure the checkpoint is present locally; download it if needed.

    Returns the model directory. Raises `ModelUnavailableError` when files are
    missing and downloading is disabled.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    weights = "model.onnx" if backend == "onnx" else "model.safetensors"
    needed = [weights] + list(TOKENIZER_MEMBERS)
    if all((model_dir / name).exists() for name in needed):
        return model_dir

    if not allow_download:
        raise ModelUnavailableError(
            f"missing files in {model_dir}: "
            f"{[n for n in needed if not (model_dir / n).exists()]} "
            "(set JOBMATCH_ALLOW_DOWNLOAD=1 to fetch them automatically)"
        )

    for name in needed:
        if name in RELEASE_FILES and not (model_dir / name).exists():
            url = f"{release_base}/{name}"
            print(f"[jobmatch] downloading {name} from {url} ...")
            _download(url, model_dir / name)
            expected = RELEASE_FILES[name]
            actual = _sha256(model_dir / name)
            if actual != expected:
                (model_dir / name).unlink(missing_ok=True)
                raise ModelUnavailableError(
                    f"checksum mismatch for {name}: expected {expected}, got {actual}"
                )

    if not (model_dir / "vocab.txt").exists():
        zip_path = model_dir / "tokenizer.zip"
        if not zip_path.exists():
            url = f"{release_base}/tokenizer.zip"
            print(f"[jobmatch] downloading tokenizer.zip from {url} ...")
            _download(url, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(model_dir)
        zip_path.unlink(missing_ok=True)

    return model_dir


class SkillExtractor:
    """Extract SKILL entities from Vietnamese/English job-description text."""

    def __init__(
        self,
        model_dir: Path,
        backend: str = "onnx",
        release_base: str = "",
        allow_download: bool = True,
    ) -> None:
        ensure_model(model_dir, backend, release_base, allow_download)
        self.model_dir = Path(model_dir)
        self.backend = backend

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        config = AutoConfig.from_pretrained(str(self.model_dir))
        self.id2label = {int(k): v for k, v in config.id2label.items()}
        self._word_cache: dict[str, tuple[int, ...]] = {}

        if backend == "onnx":
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                str(self.model_dir / "model.onnx"),
                providers=["CPUExecutionProvider"],
            )
            self.input_names = {i.name for i in self.session.get_inputs()}
        else:
            import torch
            from transformers import AutoModelForTokenClassification

            self.torch = torch
            self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
            self.model.eval()

        self._warmup()

    def _warmup(self, runs: int = 3) -> None:
        """Run a few throwaway inferences so the first real request is fast."""
        probe = "Yêu cầu 2 năm kinh nghiệm với Python, Docker và Kubernetes."
        try:
            for _ in range(runs):
                self.extract(probe)
        except Exception:  # noqa: BLE001 - warmup must never break startup
            pass

    def _predict(self, input_ids: list[int], attention_mask: list[int]) -> list[int]:
        if self.backend == "onnx":
            import numpy as np

            feeds = {
                "input_ids": np.array([input_ids], dtype=np.int64),
                "attention_mask": np.array([attention_mask], dtype=np.int64),
            }
            feeds = {k: v for k, v in feeds.items() if k in self.input_names}
            logits = self.session.run(None, feeds)[0]
            return logits[0].argmax(axis=-1).tolist()

        torch = self.torch
        with torch.no_grad():
            logits = self.model(
                input_ids=torch.tensor([input_ids], dtype=torch.long),
                attention_mask=torch.tensor([attention_mask], dtype=torch.long),
            ).logits
        return logits[0].argmax(dim=-1).tolist()

    def _tokenize(self, text: str) -> tuple[list[str], list[int], list[int], list[int]]:
        """Return (words, input_ids, attention_mask, word_ids) for one text.

        PhoBERT only ships a slow (Python) tokenizer, so `word_ids()` is not
        available — each word is tokenized on its own and the subword count is
        tracked to map labels back to words. Word -> ids is memoised because
        job descriptions repeat the same connector words over and over.
        """
        words = [m.group(0) for m in TOKEN_RE.finditer(text)]
        cls_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        sep_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id

        input_ids = [cls_id]
        attention_mask = [1]
        word_ids = [-1]

        for word_index, word in enumerate(words):
            subwords = self._encode_word(word)
            if not subwords:
                continue
            input_ids.extend(subwords)
            attention_mask.extend([1] * len(subwords))
            word_ids.extend([word_index] * len(subwords))

        input_ids.append(sep_id)
        attention_mask.append(1)
        word_ids.append(-1)

        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[: MAX_LENGTH - 1] + [sep_id]
            attention_mask = attention_mask[: MAX_LENGTH - 1] + [1]
            word_ids = word_ids[: MAX_LENGTH - 1] + [-1]

        return words, input_ids, attention_mask, word_ids

    def _encode_word(self, word: str) -> tuple[int, ...]:
        """Token ids for one word — memoised because JDs repeat the same words."""
        cached = self._word_cache.get(word)
        if cached is None:
            cached = tuple(self.tokenizer.encode(word, add_special_tokens=False))
            self._word_cache[word] = cached
        return cached
    def extract(self, text: str) -> list[SkillSpan]:
        """Extract skill spans (with character offsets) from `text`."""
        if not text.strip():
            return []

        words, input_ids, attention_mask, word_ids = self._tokenize(text)
        predictions = self._predict(input_ids, attention_mask)

        # keep the label of the first subword of each word
        word_labels: dict[int, str] = {}
        for word_id, prediction in zip(word_ids, predictions):
            if word_id >= 0 and word_id not in word_labels:
                word_labels[word_id] = self.id2label.get(int(prediction), "O")

        # group consecutive B-/I-SKILL into spans; "(" and ")" never break a span
        raw: list[list] = []
        current: list[int] = []
        last_index: int | None = None

        for index in range(len(words)):
            word = words[index]
            label = word_labels.get(index, "O")

            if word in {"(", ")"}:
                continue

            if label == "B-SKILL":
                if current and last_index == index - 1:
                    current.append(index)
                else:
                    raw.append(current)
                    current = [index]
                last_index = index
            elif label == "I-SKILL":
                current.append(index)
                last_index = index
            else:
                raw.append(current)
                current = []
                last_index = None
        raw.append(current)

        spans: list[SkillSpan] = []
        for group in raw:
            if not group:
                continue
            skill = self.post_process(" ".join(words[i] for i in group))
            if not skill:
                continue
            start = self._char_offset(text, group[0])
            end = self._char_offset(text, group[-1]) + len(words[group[-1]])
            # the tokenizer keeps a trailing '.' inside the token ("CI/CD.") —
            # trim it from the offsets so the span matches the skill name
            while end > start and text[end - 1] in ".,;:)]}\"'":
                if skill.endswith(text[end - 1]):
                    break
                end -= 1
            spans.append(SkillSpan(skill=skill, start=start, end=end))

        return spans

    @staticmethod
    def _char_offset(text: str, word_index: int) -> int:
        for match_index, match in enumerate(TOKEN_RE.finditer(text)):
            if match_index == word_index:
                return match.start()
        return 0

    @staticmethod
    def post_process(span: str) -> str:
        """Clean one raw span: strip punctuation, keep `.NET` intact."""
        token = re.sub(r"\s+", " ", span.strip()).strip(" \t\n\r,;:()[]{}\"'")
        if token.endswith(".") and not token.startswith("."):
            token = token[:-1]
        if not token or (len(token) <= 1 and token.lower() not in {"c", "r"}):
            return ""
        return token

    def extract_with_latency(self, text: str) -> tuple[list[SkillSpan], float]:
        start = time.perf_counter()
        spans = self.extract(text)
        return spans, (time.perf_counter() - start) * 1000
