# CLAUDE.md — llm-style-fingerprints

Project instructions for Claude Code.

## Architecture

```
src/
  stylometry.py   Subclass of stylometry-python's StyleAnalyzer (min_words=20)
  stats.py        bootstrap_ci, permutation_test, pairwise_tests, intra_variance
  data.py         load_corpus / load_originals / load_llm_corpora — single source of truth
  __init__.py     Re-exports from the three modules above
generate_results.py  CLI — regenerates all figures in results/
tests/
  test_stats.py   21 pytest tests — always green before committing
data/
  human/{zola,maupassant}.json
  {gpt4,claude3,mistral,gemini}/rewrites.json
results/          Generated PNGs (not hand-edited, always reproducible via make results)
notebooks/        Exploratory — mirror generate_results.py, use src.data for loading
```

## Critical constraints

- **Never import from the installed `stylometry` package directly** in `src/`.  Always use `from src.stylometry import ...`. The file `src/stylometry.py` shadows the installed package when `src/` is in `sys.path`; the project uses the project root in `sys.path` to avoid this.
- **Never hardcode group sizes** (e.g. `y = [0]*16 + [1]*16`). Use `len()` per class from the loaded data.
- **min_words=20** is intentional — the library default of 50 would reject many corpus excerpts.

## Common tasks

```bash
make install   # create .venv + install all deps
make test      # run 21 pytest tests
make results   # generate all 12 figures (slow t-SNE included)
make fast      # same but skip t-SNE (< 30 s)
make lint      # ruff on src/ and tests/
make clean     # remove __pycache__
```

## Adding a new LLM

See CONTRIBUTING.md — summary: add `data/<slug>/rewrites.json`, register slug in `src/data.py`, add colour in `src/stylometry.py::PALETTE`, run `make results`.

## Statistical pipeline

1. `StyleAnalyzer.shift(original, rewrite)` → cosine distance in function-word space
2. `bootstrap_ci(shifts)` → 95 % percentile CI (n_boot=5000)
3. `permutation_test(group_a, group_b)` → two-sample mean-difference test (n_perm=10000)
4. `pairwise_tests(shifts_dict, correction="bonferroni")` → all C(n,2) pairs, corrected p-values

## Do not

- Commit directly to `main` without running `make test`.
- Edit files under `results/` — they are always regenerated from data.
- Add API calls or network requests to the core `src/` modules — the corpus is fully offline.
