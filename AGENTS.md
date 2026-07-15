# Food Safety Graph ML — Project Policy

## Project objective

This repository supports research for **prioritizing food-safety inspections**, not making regulatory decisions. The intended downstream task is to estimate the probability that an establishment's **next eligible food-safety inspection** has a pre-registered failure outcome, using historical data available at the prediction time.

`AGENTS.md` is the authoritative, tool-neutral policy for anyone or any agent working in this repository. Claude Code-specific operating procedures live in `CLAUDE.md`.

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

## Current phase and roadmap

Only begin a phase after the project owner approves it.

1. **Phase 1 — complete:** create `.gitignore`, `CLAUDE.md`, and `AGENTS.md`; establish raw-data governance and environment/CUDA/OCR policies.
2. **Phase 2 — complete:** discover and validate Tesseract; extract text from `X.pdf`; save the result as `context_X.md` with extraction provenance.
3. **Phase 3 — complete via notebook:** `notebooks/01_data_preprocessing.ipynb` performs EDA, cleaning, and Parquet graph-ready export.
4. **Phase 4 — complete via notebook:** `notebooks/02_graphsage_embedding.ipynb` builds a temporal multi-relational graph and trains unsupervised GraphSAGE on CUDA.
5. **Phase 5 — complete via notebook:** `notebooks/03_predictive_modeling.ipynb` trains leakage-safe classifiers on embeddings and reports Accuracy, ROC-AUC, F1, PR-AUC, and confusion matrix.

## Planned repository layout

Create these paths **only when the relevant later phase begins**:

```text
data/
  raw/                    # immutable local source inputs; ignored
  interim/                # local intermediate outputs; ignored
  processed/              # local model-ready data; ignored
  graphs/                 # local graph snapshots/artifacts; ignored
  splits/                 # versioned temporal split manifests and cutoffs
src/food_safety_gnn/
  data/
  features/
  graphs/
  models/
  evaluation/
scripts/                  # runnable, parameterized entry points
configs/                  # versioned preprocessing/experiment configurations
tests/                    # tests for reusable transformations and graph logic
notebooks/                # exploration only; no production-only logic
artifacts/                # ignored embeddings, checkpoints, logs, reports
docs/                     # versioned documentation
reports/                  # versioned, human-readable summaries where appropriate
```

Source, tests, schema definitions, checked-in configurations, split manifests, compact provenance records, and human-readable reports are versioned. Raw inputs, generated tables, graph artifacts, model weights, embeddings, run logs, and OCR output remain local/ignored or go to an immutable external artifact store.

## Environment and CUDA policy

Run all project Python commands through the existing Conda environment:

```bash
conda run --no-capture-output -n meibook-dev python <script-or-module>
```

Use `conda activate meibook-dev` only in an initialized interactive shell. Dependency additions or upgrades must record an environment export and package versions.

Before any future GPU training, verify CUDA explicitly:

```bash
nvidia-smi
conda run -n meibook-dev python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

- Set `CUDA_VISIBLE_DEVICES` explicitly for GPU experiments.
- Record GPU model, NVIDIA driver, CUDA version, PyTorch version, PyG/DGL version, seed values, and determinism settings in each run manifest.
- CPU is acceptable for small checks and tabular baselines only; label it as CPU. Never silently replace a declared CUDA experiment with CPU.
- Neighbor sampling, scatter/reduce operations, and GPU kernels may remain nondeterministic. Record controls and variation across pre-registered seeds rather than promising bitwise reproducibility.

## Raw data, provenance, and artifacts

Current local raw inputs are:

| Input | Role | Handling |
| --- | --- | --- |
| `food-inspections.csv` | Source inspection records for preprocessing and graph construction | Read-only and Git-ignored |
| `X.pdf` | Research context document for Phase 2 extraction | Read-only and Git-ignored |

Acquisition source, license, download time, and downloader are currently unverified. Do not claim that either input is a current official export based only on its name or location.

Before future preprocessing or training, verify the raw file hash and schema against a recorded data contract. Stop and report unexpected hash, schema, or date-range drift. Every derived artifact must record at least: parent raw-data hash, schema version, preprocessing configuration identifier/hash, code revision, and creation command.

Use atomic writes into unique run directories. Never overwrite a completed experiment. Retain the run configuration, split manifest hash, graph snapshot hash, Git revision, environment/hardware metadata, metrics, seed, and artifact checksums.

## Prediction contract and temporal leakage prevention

These requirements must be decided in a versioned configuration before any metric is calculated:

- positive and negative inspection outcomes;
- handling of conditional, incomplete, out-of-business, no-entry, and similar inspection outcomes;
- next-inspection prediction horizon;
- eligible population and censoring rules for establishments with no later inspection;
- deterministic entity-resolution policy; and
- deterministic ordering/tie policy for date-only timestamps.

Do not infer target mapping, exclusions, or a prediction horizon after inspecting held-out prevalence or test metrics.

The default evaluation design is chronological or rolling-origin with content-hashed membership manifests and documented cutoff timestamps. Build separate temporal graph snapshots per split cutoff. Fit encoders, imputers, scalers, IDF/TF-IDF, feature-selection methods, similarity indexes, graph thresholds, and sampling distributions using the training era only; apply them to validation/test without refitting.

Never construct a graph from all historical records and hide only test labels. Future nodes, edges, attributes, topology, validation/test labels, or future inspection events must not contribute to training messages, normalization statistics, random walks, negative sampling, aggregations, or embeddings. At evaluation time, only historical data available at that cutoff may be present.

All edge types are potential leakage paths. Geography, ZIP, facility type, and violation-similarity relationships must be calculated from attributes available at the graph snapshot time. Treat rolling temporal evaluation and grouped/cold-start evaluation as distinct claims, and name the protocol used in every result.

## Graph and model quality requirements

For each future graph snapshot, record:

- node and edge counts per relation;
- degree quantiles and isolated-node rate;
- directionality, duplicate-edge, and self-loop policies;
- missing-coordinate treatment; and
- deterministic radius, `k`, threshold, degree-cap, and neighbor-sampling settings.

Use comparable target eligibility, temporal snapshots, split manifests, feature fitting, and evaluation populations for all baselines and GNNs. Establish non-graph/tabular baselines before claiming a GraphSAGE gain. Select graph thresholds, class weights, focal-loss settings, early-stopping criteria, decision thresholds, and capacity-based ranking metrics on training/validation data only; lock them before evaluating the final test period.

At minimum, report class counts/prevalence, Accuracy, ROC-AUC, F1-score, and PR-AUC for an imbalanced outcome. Report seed-level variation rather than only the best seed. Calibration and Recall@K should be reported when their operational definitions are pre-registered.

## OCR policy for Phase 2

OCR is deferred until Phase 2. The eventual extraction script must:

1. locate Tesseract with `shutil.which("tesseract")` and allow a documented `TESSERACT_CMD` override;
2. validate the executable with `tesseract --version` and `pytesseract.get_tesseract_version()`;
3. inspect installed languages using `tesseract --list-langs`;
4. fail with an actionable error when unavailable rather than hard-coding a machine path;
5. preserve source hash, page number, extraction method, engine/version, languages, DPI/settings, timestamp, and validation status; and
6. compare native PDF text extraction with OCR where a text layer exists before treating OCR as authoritative.

Keep `X.pdf`, page images, and OCR output separate from leakage-safe predictive features unless an approved future research design specifies otherwise.

## Code and research conventions

- Use Python 3.10+ in `meibook-dev`, `pathlib` paths, PEP 8, type annotations, concise public-function docstrings, and structured logging.
- Prefer a `src/` package and parameterized scripts/configurations over notebook-only production logic.
- Test reusable data transformations, entity resolution, temporal splitting, and graph construction.
- Preserve raw columns until a documented transformation explicitly replaces them in a derived artifact.
- Keep seeds in configuration and set Python, NumPy, PyTorch, and graph-library random number generators where applicable.
- State assumptions, data-quality limitations, leakage uncertainty, unavailable CUDA/Tesseract, and phase-boundary conflicts as blockers; do not guess or silently continue.
