# Retrieval-First MMR Reuse

Car-Census uses a project-local, provenance-preserving retrieval cache before calling TrafficEye rather than training a local classifier. Exact image/request matches may be reused immediately; embedding-based near-duplicate matches default to shadow mode and must fall through to TrafficEye until their precision is calibrated, because reducing API cost must not introduce false local vehicle identities.

## Consequences

- Stored evidence keeps the original crop, request contract, raw API response, and versioned visual representation.
- Exact and approximate reuse expose their resolution method separately from `api_confirmed` evidence provenance.
- Near matches reuse make/model/generation and only retain variation when nearby evidence agrees; image-specific fields are not copied.
- The first visual representation is a deterministic normalized-pixel vector plus perceptual hash, not a trained semantic classifier; its version is part of the stored record.
- The shared cache defaults to `.mmr-cache` relative to `project.output_root` and can be relocated with `project.retrieval_cache_dir`.
