# Paper

`main.tex` with one file per section in `sections/`, references in `refs.bib`.

```bash
make -C paper          # runs make_generated.py, then pdflatex + bibtex; fails on undefined references
```

Every number in the paper comes from a result file: `make_generated.py` reads the Monte Carlo
outputs (`experiments/phase2/results/`), the small language model (`experiments/phase3/results/`),
the TPU benchmarks (`benchmarks/tpu/results/`), the tuning and confirmatory runs (`runs/tuning/`,
`runs/main/`), the Phase 08 analysis (`analysis/results/`) and the Lean audit
(`formal/audit/axioms_output.txt`), and writes `generated/macros.tex`, the tables in `generated/`
and copies of the analysis figures. A result that does not exist yet prints as **[pending]**, so
the paper builds at every stage. To regenerate the inputs from the raw logs first:

```bash
python analysis/tuning_report.py
python analysis/phase08.py && python analysis/write_decision.py
python analysis/compute_budget.py
```

Requirements: a TeX Live installation with `pdflatex`, `bibtex`, `natbib`, `cleveref`, `booktabs`,
`microtype` and `lmodern`; Python with the project installed (`pip install -e ".[dev]"`).

The map from each result sentence of the paper to its row in the claim–evidence matrix
(`research/ledger/claims.md`) is in `research/ledger/claim_audit.md`.
