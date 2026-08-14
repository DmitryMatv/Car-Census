# Retrieval-First MMR Reuse

Car-Census uses a project-local, provenance-preserving retrieval cache before calling TrafficEye rather than training a local classifier. Exact image/request matches may be reused immediately. Embedding-based near-duplicate matches use OpenRouter's multimodal `google/gemini-embedding-2` model at 768 dimensions and require both a perceptual-hash gate and a calibrated cosine-distance gate before enforce-mode reuse.

## Consequences

- Stored evidence keeps the original crop, request contract, raw API response, and versioned visual representation.
- Exact and approximate reuse expose their resolution method separately from `api_confirmed` evidence provenance.
- Near matches reuse make/model/generation and only retain variation when nearby evidence agrees; image-specific fields are not copied.
- Embeddings are versioned by model and dimension, cached by image SHA, and may be unavailable without invalidating exact retrieval.
- pHash filtering happens before paid embedding requests. OpenRouter failures fall through to TrafficEye while retaining the API result.
- Calibration requires evidence from both same-identity and conflicting-identity pairs and fails closed when the cache is too small or the distributions overlap.
- Legacy records are retained unchanged; migration creates superseding records that point to the original evidence and image bytes.
- The shared cache defaults to `.mmr-cache` relative to `project.output_root` and can be relocated with `project.retrieval_cache_dir`.
