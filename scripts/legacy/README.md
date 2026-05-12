# Legacy scripts

These shell scripts target an older workflow built around cold-booting SMB through the title screen. That path no longer works because of a known PPU-timing bug in the title-screen state machine; the current flow uses save-state reset (see `docs/training.md`). Kept here for reference; not part of the current training pipeline.

| File | What it did |
|---|---|
| `smoke_phase2_user_rom.sh` | Phase 2 user-ROM smoke test |
| `reproduce_phase6.sh` | Reproduce the phase-6 A100 benchmark |
| `compare_openemu_state.sh` | Pixel-compare against an OpenEmu reference frame |
| `render_openemu_state.sh` | Render an OpenEmu save state through our PPU |
