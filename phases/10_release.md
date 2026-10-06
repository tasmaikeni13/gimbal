# Phase 10 — Release: cleanup, comments, PEP 8, humanized prose, final organization

## Purpose

Make the repository ready to be read and reused by others: clean code with useful comments, PEP 8
formatting, a natural, human-sounding README and paper, and an organized layout.

## Depends on

Phase 09 (paper), all earlier phases passed.

## Tasks

1. **Inventory.** List every file; delete dead code, scratch notebooks and duplicated scripts (move
   anything still referenced by the paper into `experiments/` or `analysis/`). Never delete raw results
   that support a claim.
2. **Code quality.**
   * Format and lint: `ruff format .` and `ruff check . --fix` (PEP 8; line length 100 unless the
     project config says otherwise); fix remaining warnings by hand.
   * Docstrings for every public function and class (NumPy style): purpose, shapes, units, references
     to the theory document (e.g. "Theorem 4").
   * Comments explain *why* (numerics, invariants, paper references), not what the next line does.
   * Type hints on public APIs; `pytest -q` and the Lean build pass after every change.
3. **Humanize the prose** of `README.md` and the paper (and `theory.md` if desired) with the
   `humanize` skill:
   * Clone it: `git clone --depth 1 https://github.com/tasmaikeni13/humanize /tmp/humanize`.
   * If the clone fails because the repository is private: first try an authenticated clone (GitHub
     app / token available to the agent). If that is impossible, the owner has authorized this
     procedure: make the repository public, clone it, and **immediately** make it private again
     (`gh repo edit tasmaikeni13/humanize --visibility public --accept-visibility-change-consequences`,
     clone, then `--visibility private`). Verify and record that it is private again.
   * Follow `SKILL.md` and its references (`reference/02-revision-process.md`,
     `reference/06-genre-playbooks.md` for academic and technical genres,
     `reference/08-quality-and-evaluation.md`). Protect every number, citation, theorem statement,
     equation, code identifier and claim strength; humanizing must not change meaning. Run the
     skill's faithfulness check and diff numbers before/after.
4. **Organize.** Final layout (adjust if earlier phases changed it):
   `src/gimbal/` (library), `formal/` (Lean), `theory/` and `theory.md`, `experiments/`, `analysis/`,
   `configs/`, `paper/`, `phases/`, `research/`, `tests/`, `docs/`.
5. **Release metadata.** `README.md` (what, why, results table with uncertainty, how to reproduce,
   citation), `CITATION.cff`, `LICENSE` (ask the owner which license if not already chosen),
   `CHANGELOG.md`, pinned environments (`requirements*.txt` / `pyproject.toml`, `lean-toolchain`).
6. **Final verification** from a fresh clone: install, `pytest -q`, `lake build`, regenerate one table
   from raw logs, build the paper.
7. Commit, push, and tag a release (`v1.0`).

## Exit gate

* G10.1 Fresh-clone verification passes.
* G10.2 `ruff check .` clean; no `print` debugging left; no TODOs without an issue link.
* G10.3 Humanized documents keep every fact, number and claim (diff audit recorded in
  `research/ledger/decisions.md`); the `humanize` repository's visibility is restored if it was
  changed.
* G10.4 README results match `analysis/results/` exactly.
