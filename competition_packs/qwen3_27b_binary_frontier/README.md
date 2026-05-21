# Qwen3.6 27B Mostly-Binary Compression Frontier

This validator-owned pack measures hidden heldout PPL against true compression
under the Qwen3.6 27B contract.

The included starter trains a deterministic layerwise output-matching proxy:
each layer may emit `binary_basis_scales` so the fixed validator can replay the
same additive binary bases and q4 rescue mask.

Primary submission surface:

- `artifact_uri` pointing at a compressed artifact in a public Hugging Face repo
  or another public HTTPS location
- `artifact_sha256` and `artifact_size_bytes` for the exact bytes validators
  must download

Recipe/code patches are optional metadata; coordinators store only URI/integrity
metadata.
Validators download, load, account for, and score the artifact directly.

Artifact accounting is validator-computed. The hard checks are:

- parameter count within the Qwen3.6 27B binary-compatible cap
- compressed representation size within a 90% binary plus 10% scaled q4 rescue budget
- rescue values submitted as signed q4 codes plus fp16 scales per 128-code group; missing or out-of-range q4 codes reject
- non-binary q4 rescue fraction at or below `10%`

Miners can use additive binary bases, residuals, side tables, or other designs
as long as every extra component is declared in `artifact.accounting.extra_entries`
or a layer `extra_components` list so the validator counts the parameters, bits,
and rescue usage.

What stays fixed:

- the benchmark harness
- the hidden shard selection contract
- the layer manifest and inference contract
- the hard eligibility filters
- the `0.02`-nat PPL resolution used before accepting a new PPL best
