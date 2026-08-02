# Contributing / Workflow

This is a solo portfolio project, but it follows a real branching discipline so the Git history reflects actual engineering practice — and so it stays readable to anyone reviewing it.

## Branching Convention

- `main` is protected: no direct pushes, all changes land via pull request.
- Feature branches are named `feature/M<n>-<short-slug>`, tying each branch to a project milestone.
  Examples: `feature/M1-lane-state-contract`, `feature/M3-acados-solver`, `feature/M4-int8-quantization`.
- One branch per meaningful unit of work — not one branch per milestone. A milestone typically spans several PRs.

## Commit Messages

Short imperative summary, optionally a body explaining *why* for non-obvious changes:

```
Add clock-skew-corrected latency measurement to bridge

Two-way sum cancels VM/host clock drift; single-direction
measurement was showing clock skew, not latency.
```

## Pull Requests

- PR description references the requirement(s) it implements, e.g. `Implements FR-8, FR-18`.
- Squash merge into `main` — keeps history one entry per logical change.
- CI must pass once the pipeline exists (from M5 onward). Before that, the PR itself is the checkpoint: does this change work, is it documented if needed.

## Requirement Traceability

Every non-trivial PR should reference the SRS requirement(s) it addresses (FR-n / NFR-n). This keeps the link between the spec, the roadmap, and the actual code auditable — see `docs/srs.md`.
