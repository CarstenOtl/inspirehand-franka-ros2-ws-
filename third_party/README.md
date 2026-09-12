# Third-party source

The Python runtime package under `flow_matching/flow_matching/` is a
byte-for-byte copy of `reference/forgeUltra/flow_matching/flow_matching` from
the `franka-chi` checkout at commit
`3fa5a545fbd0e2e4a465645e8e6b02c13a7e609c` (Flow Matching 1.0.10). Its
upstream README and packaging metadata are included; upstream examples, docs,
tests, and CI configuration are intentionally omitted because rollout does not
use them.

Its upstream license is retained at `flow_matching/LICENSE` and is CC BY-NC
4.0. Keep that non-commercial restriction in mind when distributing or using
the policy-rollout app.
