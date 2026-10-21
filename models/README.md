# `models/phobert-ner`

This directory is **gitignored** — the checkpoint is ~513 MB.

The model is the fine-tuned PhoBERT from
[vietnamese-job-skill-ner](https://github.com/phamthaic4/vietnamese-job-skill-ner)
(v1.0.0, test F1 0.6711 entity-level).

## Automatic download (default)

On first start the API fetches the release assets and verifies their SHA-256:

```
model.onnx         513 MB   a0548653873512d3750033ea2d8089f3f6bf761d0d1f1b83ef3ed248d8ec06ed
model.safetensors  513 MB   f047aea77ab031a4e8d826da8a379886b571d755156177daea9c888611557db2
tokenizer.zip      0.8 MB   59c747474ae5e17eb7ef96d59e6867d3178f5a8ccfd3510e7fc5adbe245801a4
```

```bash
curl -L -o models/phobert-ner/model.onnx \
  https://github.com/phamthaic4/vietnamese-job-skill-ner/releases/download/v1.0.0/model.onnx
```

Set `JOBMATCH_ALLOW_DOWNLOAD=0` to disable downloading (production images).

## Manual layout

```
models/phobert-ner/
  config.json
  model.onnx            # ONNX backend
  model.safetensors     # transformers backend
  tokenizer_config.json
  vocab.txt
  bpe.codes
  added_tokens.json
```
