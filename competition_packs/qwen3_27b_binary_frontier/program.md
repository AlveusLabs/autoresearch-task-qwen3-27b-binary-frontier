# Program

Submit a mostly-binary artifact under the Qwen3.6 27B binary compression contract.

Primary validation is artifact-first: miners train off-chain and submit the
compressed artifact through `artifact_uri`. Use a public Hugging Face artifact
URL or another public HTTPS URL; submit `artifact_sha256` and
`artifact_size_bytes` with it. The coordinator stores URI/integrity metadata,
not artifact bytes.
Recipe/code patches are optional metadata and are not replayed as the normal
acceptance gate.

Artifacts may include learned `binary_basis_scales` for additive binary layers;
the benchmark replays those scales deterministically before applying the rescue
allowance.

Hard gates:

- shape valid
- parameter count within the Qwen3.6 27B binary-compatible cap
- compressed representation size within a 90% binary plus 10% scaled q4 rescue budget
- submitted rescue payloads must be signed q4 codes plus fp16 scales per 128-code group; the validator rejects missing or out-of-range q4 codes
- non-binary q4 rescue fraction `<= 10%`
- artifact loadable by the fixed validator
- heldout quality floor

Extra residuals, side tables, and other representation pieces are allowed, but
they must be declared in `artifact.accounting.extra_entries` or layer
`extra_components` so the validator counts their parameters, bits, and rescue
usage.

Survivors are ranked on `heldout_ppl` (lower is better).

New best acceptance uses a `0.02` heldout-PPL resolution in nats. A submission
can win by reducing PPL past that band. A smaller packed model can also claim
SOTA while remaining inside the incumbent's PPL band.
