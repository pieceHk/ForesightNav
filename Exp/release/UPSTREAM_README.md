# 3DONav

3DONav has exactly three owners: `Mapping`, `Navigation`, and `Exp`.

- Mapping produces online geometric-semantic candidates.
- Navigation consumes candidates and owns action, verification and recovery.
- Exp declares datasets/profiles and records reproducible evidence.

All project runtime source, datasets and weights resolve inside this repository. Machine-local Python interpreter
paths are configured in `Exp/runtime.local.toml`; validate the boundary before launch:

```bash
python Exp/scripts/validate.py
python Exp/scripts/validate.py --require-clean  # before a reproducible release
PYTHONPATH=Mapping/src:Navigation/src pytest -q
```
