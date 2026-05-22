# Contributing

## Adding a new LLM to the corpus

### 1. Prepare the rewrite data

Create a JSON file at `data/<model_slug>/rewrites.json` with this schema:

```json
{
  "model": "<model-id-string>",
  "prompt": "<the rewriting instruction you used>",
  "rewrites": [
    {
      "original_id": 0,
      "source": "zola",
      "rewrite": "... text ..."
    },
    ...
  ]
}
```

- `model_slug` is a short lowercase identifier (e.g. `llama3`, `mixtral`).
- `original_id` is the 0-based index of the text in the source corpus (`zola` or `maupassant`).
- Include all 16 originals (8 Zola + 8 Maupassant) to keep group sizes balanced.
- Each rewrite should be at least 20 words (the library's minimum for reliable stylometric signals).

### 2. Register the model in `src/data.py`

Add the slug to the `MODELS` tuple:

```python
MODELS: tuple[str, ...] = ("gpt4", "claude3", "mistral", "gemini", "your_slug")
```

Add a human-readable label in the `LABELS` dict:

```python
LABELS: dict[str, str] = {
    ...
    "your_slug": "Your Model Name",
}
```

### 3. Add the palette colour in `src/stylometry.py`

```python
PALETTE = {
    ...
    "Your Model Name": "#AABBCC",   # pick a hex colour
}
```

### 4. Run the pipeline

```bash
make results   # regenerates all figures in results/
make test      # 21 tests — should still be green
```

### 5. Open a PR

- Include the new `data/<model_slug>/rewrites.json`.
- Attach the updated `results/` figures so reviewers can see the impact at a glance.
- Note the exact model version and sampling parameters (temperature, top-p) in the PR description.

## Running the test suite

```bash
make install   # first time only
make test
```

## Code style

We use [ruff](https://docs.astral.sh/ruff/) with the settings in `pyproject.toml`:

```bash
make lint
```

All CI checks must pass before merge.
