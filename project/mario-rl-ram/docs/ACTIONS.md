# Action Space

The first full run used `--action-mode all`, which exposes the full NES controller as `MultiBinary(9)`.

That is expressive, but it includes controller states that are bad for learning:

- `START` and `SELECT`
- `LEFT + RIGHT`
- action combinations Mario cannot use meaningfully
- many variants that differ only by useless buttons

The recommended final run uses:

```bash
--action-mode mario
```

This maps a compact discrete action space onto intentional Mario controls:

| Index | Action | Buttons |
|---:|---|---|
| 0 | `NOOP` | none |
| 1 | `RIGHT` | `RIGHT` |
| 2 | `RIGHT_RUN` | `RIGHT + B` |
| 3 | `RIGHT_JUMP` | `RIGHT + A` |
| 4 | `RIGHT_RUN_JUMP` | `RIGHT + B + A` |
| 5 | `JUMP` | `A` |
| 6 | `RUN_JUMP` | `B + A` |
| 7 | `LEFT` | `LEFT` |
| 8 | `LEFT_JUMP` | `LEFT + A` |
| 9 | `DOWN` | `DOWN` |
| 10 | `RIGHT_DOWN` | `RIGHT + DOWN` |

This is not right-only mode. It keeps the core platforming moves while removing the controller noise that made jump-spam too easy to discover.
