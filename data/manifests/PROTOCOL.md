# Paper Evaluation Protocol v1

- Benchmarks: PushT and OGBench Cube
- Evaluation seeds: 32, 42, 52
- Episodes per seed/horizon: 50
- Horizons: 25, 50, 75, 100, 125, 150 environment steps
- Candidate budget: 300; CEM: 30 rounds, topk=30
- Each result must consume the matching manifest in this directory.
- Evaluators must not create or resample starts for paper tables.

## Manifest provenance

- Each manifest preserves the exact episode, start frame, goal frame, horizon,
  record order, and sample seed used by the paper evaluation.
- Machine-local source paths were removed during release sanitization.
- `semantic_manifest_sha256` authenticates the ordered semantic records inside
  each JSON file. `SHA256SUMS` authenticates the released files byte for byte.
