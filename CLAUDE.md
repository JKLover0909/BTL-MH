# Claude Code Execution Guide — Food Safety Graph ML

Read `AGENTS.md` before planning, editing, running commands, or interpreting results. `AGENTS.md` is the authoritative policy; this file specifies how Claude Code should apply it.

## Critical invariants

- The project supports inspection prioritization only; it must not be represented as an automated regulatory decision-maker.
- The prediction target is an establishment's next eligible inspection, with a failure definition and horizon registered before model evaluation.
- The current scope is **Phase 1 — project foundation and governance only**. Do not OCR documents, preprocess data, produce derived artifacts, train models, or claim Graph ML results in this phase.
- Raw input data is read-only. Derived data, graphs, checkpoints, embeddings, and reports must never overwrite it.
- Do not commit raw data, credentials, local OCR output, generated artifacts, model weights, or other large binaries.
- Define the prediction timestamp before feature engineering. Every feature and relation must be based only on information available **strictly before** that timestamp.
- Never use the target inspection's violations, score, disposition, pass/fail outcome, free text, corrections, follow-ups, or later information as model inputs.
- Use Conda environment `meibook-dev` only. Do not use system Python or create a different environment without explicit maintainer approval.
- Later deep-learning training must use compatible CUDA/PyTorch software and must fail clearly instead of silently running a declared GPU experiment on CPU.
- Tesseract discovery and OCR belong to Phase 2, not Phase 1.

## Phase gate

1. Identify the requested phase before acting.
2. If work crosses into a later phase, stop and ask the project owner to approve that phase.
3. During Phase 1, the permitted project files are only `.gitignore`, `AGENTS.md`, and `CLAUDE.md`. Do not modify `food-inspections.csv` or `X.pdf`; do not create derived data, OCR output, models, scripts, source packages, notebooks, or experiment directories.
4. Do not claim that a later phase has been completed until its requested outputs are created and verified.

## Required environment

Use the existing `meibook-dev` Conda environment for all project Python commands:

```bash
conda run --no-capture-output -n meibook-dev python <script-or-module>
```

Use `conda activate meibook-dev` only in an initialized interactive shell. Never use bare `python`, system Python, or a newly created environment without explicit authorization.

For a later CUDA training task, run and report both checks before training:

```bash
nvidia-smi
conda run -n meibook-dev python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

A requested GPU run must use explicit device selection (for example, `CUDA_VISIBLE_DEVICES`) and should fail clearly if CUDA is not available. CPU fallback is allowed only if the owner explicitly accepts a CPU-labelled run.

## Planned structure

Create only when the corresponding later phase is approved:

```text
data/raw/                 # ignored immutable source inputs
data/interim/             # ignored intermediate results
data/processed/           # ignored model-ready tables
data/graphs/              # ignored temporal graph artifacts
data/splits/              # versioned split manifests
src/food_safety_gnn/      # reusable package
scripts/                  # parameterized entry points
configs/                  # versioned configurations
tests/                    # reusable-logic tests
notebooks/                # exploration only
artifacts/                # ignored embeddings/checkpoints/logs
docs/ and reports/        # versioned documentation/summaries
```

The intended phase sequence is:

1. Project setup and governance.
2. Tesseract discovery and `X.pdf` context extraction to `context_X.md`.
3. Inspection-data EDA, cleaning, and graph-ready storage.
4. CUDA unsupervised GraphSAGE graph construction, training, and embedding export.
5. Embedding-based inspection-failure classifiers and Accuracy/ROC-AUC/F1 evaluation.

## Operating procedure for future changes

Before changing code or data:

1. Inspect the repository status and read `AGENTS.md`.
2. Confirm the phase boundary, requested deliverable, and whether inputs are raw/ignored.
3. For raw-data-consuming work, verify the recorded file hash and expected schema before preprocessing or training.
4. Use repository-relative/config-driven paths through `pathlib`; never hard-code a user-specific Tesseract path, input path, or output path.
5. Design the prediction timestamp, target definition, temporal split, entity-resolution policy, and graph snapshot cutoff before feature construction.

While implementing:

- Use PEP 8, type hints, concise public-function docstrings, structured logging, and reusable `src/` code.
- Keep production data, feature, graph, and training logic out of notebooks.
- Make each transformation and graph snapshot deterministic from a versioned configuration and documented seed.
- Fit preprocessing only on training-era data and prevent future graph topology, node attributes, labels, random walks, sampling distributions, or embeddings from leaking into earlier snapshots.
- Store derived artifacts in ignored paths with input hashes, configuration hash, code revision, command, seed, environment, CUDA/device, and checksum metadata.

After changes:

1. Run the relevant tests/lint/type checks established by the project.
2. Run `git diff --check`.
3. Report files changed, commands executed, results, unresolved assumptions, data-quality findings, and blockers.
4. Do not overwrite a completed run directory or conceal unavailable CUDA/Tesseract, schema drift, or leakage uncertainty.

## OCR behavior for Phase 2

When Phase 2 is approved, the OCR workflow must discover Tesseract with `shutil.which("tesseract")`, allow a documented `TESSERACT_CMD` override, validate the binary and language packs, and fail with actionable setup guidance when unavailable. Record the executable/version, languages, page-level provenance, source hash, method, settings, timestamp, and human validation state. Compare native PDF text extraction with OCR before treating OCR as authoritative.

## Raw input handling

The current source files are local and Git-ignored:

- `food-inspections.csv` — read-only input for Phase 3 onward.
- `X.pdf` — read-only research-context input for Phase 2.

Their acquisition provenance is unverified. Do not characterize them as current official exports without independent source validation. Keep the source documents and all generated output separate; no raw input may be overwritten or committed.
