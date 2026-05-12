# Historical project docs

These are kept for reference and provenance — they describe how the project evolved through its development phases (NES emulator → CUDA batching → Gymnasium/SB3 API → ROM-backed cuda-console → throughput benchmarks). They are **not** part of the current training flow.

For the current state of the project, see:

- [../training.md](../training.md) — how to actually train an SMB agent
- [../architecture.md](../architecture.md) — current system design
- [../phase6-report.md](../phase6-report.md) — current A100 throughput benchmark methodology and results
- [../benchmark-gpu-vs-cpu.md](../benchmark-gpu-vs-cpu.md) — local GTX 1050 Ti validation

| File | What it covers |
|---|---|
| `phases.md` | Phase 0–7 narrative; the staged build-up of CPU/PPU/CUDA/API |
| `phase5-results.md` | Pre-phase-6 calibration data (superseded by `phase6-report.md`) |
| `phase6-readiness.md` | Entry gate that decided phase 6 was ready to run |
