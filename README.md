# qwen3-27b-binary-frontier

Public Qwen3.6 27B binary frontier task pack.

Run locally from the repo root:

```bash
python3 competition_packs/qwen3_27b_binary_frontier/prepare.py
python3 competition_packs/qwen3_27b_binary_frontier/benchmark.py
```

Editable surfaces are defined by the coordinator task configuration.

Generated Python bytecode/cache artifacts are omitted from this task pack. Submitted patches are accepted only for the coordinator `allowed_patch_paths`, and manual bytecode/cache patch paths are rejected:

- `__pycache__/`
- `*.pyc`
- `*.pyo`
