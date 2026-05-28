---
layout: post
title: "How LLMs Transform Writing Style: A Stylometric Experiment"
date: 2026-05-28
author: Riad Maouchi
description: "Measuring stylistic drift in LLM rewrites — not detecting AI, but fingerprinting it. 4 models, 80 French passages, 41 function words."
---


*Measuring stylistic drift in LLM rewrites — not detecting AI, but fingerprinting it*

![How LLMs transform writing style — drift in PCA function-word space, 4 models, 80 French passages](https://raw.githubusercontent.com/riadmaouchi/llm-style-fingerprints/main/results/hero.png)

---

Four models. One Zola paragraph. Eighty texts.

The same passage rewritten four times — each rewrite following the same instruction, each ending up somewhere different in style-space. Not random variation: consistent, measurable, reproducible differences that persist across 80 texts and two distinct prompts.

That is the observation this project starts from.

The question it asks is not *"is this AI-generated?"* — a binary that is increasingly fragile and theoretically contested. The question is: **when an LLM rewrites a human text, where does it move the style?** And does each model move it in a characteristic direction?

Code and data: [github.com/riadmaouchi/llm-style-fingerprints](https://github.com/riadmaouchi/llm-style-fingerprints). Paper: [doi.org/10.5281/zenodo.20402754](https://doi.org/10.5281/zenodo.20402754).

---

## What stylometry actually measures

Stylometry is the quantitative study of writing style. The discipline is about sixty years old, predating machine learning by decades.

The founding result: in 1964, Mosteller and Wallace used function-word frequencies to settle a long-standing historical dispute — which of Hamilton or Madison had authored the contested *Federalist Papers*. Their method worked not by analysing what was *said*, but by counting how it was said: the relative frequency of words like *upon*, *while*, *whilst*, *to*, *enough*.

The insight that made this work, and that still underlies the field: the words that carry *style* are not the dramatic ones. They are the invisible ones — prepositions, articles, pronouns, conjunctions. A writer who habitually opens subordinate clauses with *although* rather than *while*, or who reaches for *however* rather than *but*, does so semi-automatically, without deliberate choice. These habits accumulate across thousands of decisions and become statistically distinctive.

In 2013, Patrick Juola applied four stylometric features — including function-word distributions — to identify J.K. Rowling as the author behind the pseudonym Robert Galbraith. The book was *The Cuckoo's Calling*, published without public disclosure of its real author. The analysis held up.

The hypothesis this project tests: **do LLMs have analogous stylistic habits?** When instructed to rewrite a text, do they systematically reach for certain connectors, certain clause structures, certain grammatical patterns — consistently enough to leave a measurable trace?

---

## The setup

**Corpus**: 80 French literary passages — 40 from Zola (*Germinal*, *L'Assommoir*, *Nana*, *La Débâcle*), 40 from Maupassant (*Boule de suif*, *La Parure*, *Yvette*). Each passage is roughly 120 words. 19th-century literary prose: syntactically dense, stylistically distinct between authors.

**Rewrites**: each passage sent to GPT-4, Claude 3, Mistral 7B, and Gemini Pro with the instruction:
> *"Réécris ce texte dans un style neutre et factuel, en conservant le sens."*

A second prompt — simpler register, aimed at a general audience — was applied to the same 80 texts to test whether stylistic signatures hold across different instructions.

**Representation**: each text mapped to a 41-dimensional vector of L2-normalised function-word frequencies. No embeddings. No transformers. No fine-tuning. The 41 words are grammatical particles: *le, la, les, un, une, et, ou, que, qui, dans, sur, il, elle, ne, pas, bien, tout, tandis, pourtant, néanmoins, notamment, afin…*

The list includes some words — *tandis*, *pourtant*, *néanmoins*, *notamment* — that were hypothesised to be characteristic of formal LLM output. This is a methodological choice worth noting: it introduces a prior toward detecting formally-inclined models. The limitations section returns to this.

**Shift metric**: for each original text `t` and its rewrite `t'`, the stylistic shift is the cosine distance between their function-word vectors:

```python
shift(t, t') = cosine_distance(v(t), v(t'))
```

A shift of 0.0 means the rewrite is stylistically identical to the original in function-word space. A shift of 0.230 — Gemini's average — is substantial. All outputs are pre-generated and committed to the repository. No API key required to reproduce.

---

## The same paragraph, four ways

Here is the opening of *Germinal*, and what each model does with it:

**Original — Zola:**
> *Quand Étienne Lantier descendit dans la fosse, il faisait nuit encore. La pluie tombait par rafales sur la plaine rase, un vent du nord-est soufflait, glacial. L'homme s'arrêta au bord du puits, la lampe à la main, et regarda l'abîme. Au fond, dans les ténèbres, on entendait gronder la machine.*

**GPT-4** — strips the syntax to its functional minimum, *et* as the dominant connector:
> *Étienne Lantier descendait dans la mine alors qu'il faisait encore nuit. Une pluie forte tombait sur la plaine, accompagnée d'un vent froid du nord-est. Il s'arrêta au bord du puits avec sa lampe et observa l'obscurité. On entendait le bruit de la machine en contrebas.*

**Claude 3** — formal subordination, *tandis que*, participial constructions:
> *Tandis qu'Étienne Lantier descendait dans la fosse, la nuit n'était pas encore levée. Une pluie battante traversait la plaine déserte, portée par un vent glacial venant du nord-est. Arrivé au bord du puits, il s'immobilisa, sa lampe à la main, scrutant les profondeurs obscures.*

**Mistral 7B** — conservative paraphrase, surface-level substitutions, tense inconsistency:
> *Étienne Lantier descendait dans la mine. Il faisait encore nuit. La pluie tombait fort sur la plaine et le vent soufflait du nord-est. Il s'est arrêté au bord du puits avec sa lampe et a regardé en bas.*

**Gemini Pro** — explicit causal connectors, *en raison de*, *afin de*, *notamment*; the original atmosphere converted into analytical prose:
> *Alors qu'Étienne Lantier entamait sa descente dans la fosse, l'obscurité de la nuit régnait encore. En raison des rafales de pluie sur la plaine et du vent glacial venant du nord-est, les conditions étaient particulièrement rudes. Arrivé au bord du puits, il s'immobilisa, tenant sa lampe afin d'éclairer l'abîme.*

A careful reader would notice the differences without any measurement. GPT-4 evacuates atmosphere; Claude adds formal subordination that was not in the original; Mistral stays close to the surface but loses internal consistency; Gemini adds explanatory machinery that transforms mood into description. The stylometric measurement confirms and quantifies what reading suggests.

---

## Measuring the drift

Each rewrite lands at a measurable distance from its original in the 41-dimensional function-word space. Averaged across 80 texts, with bootstrap confidence intervals (5,000 resamples):

| Model | Mean shift | 95% CI |
|-------|:----------:|:------:|
| Gemini Pro | **0.230** | [0.204, 0.256] |
| Claude 3 | **0.170** | [0.147, 0.195] |
| Mistral 7B | **0.139** | [0.124, 0.157] |
| GPT-4 | **0.132** | [0.113, 0.152] |

The ranking is consistent. But the raw numbers understate the structure in the data.

![Mean stylistic shift per model with 95% CI — horizontal bars sorted by displacement. Gemini stands apart; GPT-4 and Mistral overlap entirely](https://raw.githubusercontent.com/riadmaouchi/llm-style-fingerprints/main/results/bootstrap_ci.png)

![At-a-glance — mean cosine shift and bootstrap 95% CI for each model](https://raw.githubusercontent.com/riadmaouchi/llm-style-fingerprints/main/results/summary_card.png)

The CIs do not overlap between Gemini and the other three. Between GPT-4 and Mistral, the CIs nearly coincide. This already suggests the story is not "four distinct fingerprints" — it is something more constrained.

---

## The structure in the data: two effective clusters, not four

Pairwise permutation tests (Bonferroni-corrected, 10,000 permutations) on the shift distributions make the structure explicit:

| Pair | Corrected p | Significant |
|------|:-----------:|:-----------:|
| GPT-4 vs Gemini | < 0.001 | yes |
| Mistral vs Gemini | < 0.001 | yes |
| Claude vs Gemini | 0.007 | yes |
| GPT-4 vs Claude | 0.103 | no |
| Claude vs Mistral | 0.262 | no |
| **GPT-4 vs Mistral** | **1.000** | **no** |

GPT-4 and Mistral are statistically indistinguishable (p = 1.0) in this experiment. Their shift magnitudes (~0.132 and ~0.139) and distributions overlap almost entirely. Claude occupies a grey zone — measurably larger shift than GPT-4/Mistral, but not significantly so after correction. Gemini stands apart from all three.

The practical picture is not four fingerprints but two groups with one outlier:

```
GPT-4 ≈ Mistral 7B  ←————————————→  Gemini Pro
    (shift ~0.135)                   (shift ~0.230)
          ↑
     Claude 3 (~0.170)
     not distinguishable from either cluster
```

The drift trajectories plot makes this spatial: arrows go from the human centroid (★) to each model's centroid in PCA-projected style-space. Arrow length is proportional to mean shift. The visual confirms what the statistics say — Gemini's arrow is ~75% longer than GPT-4's; GPT-4 and Mistral point in similar directions at similar distances.

![Drift trajectories in PCA function-word space — ★ = human centroid, ✕ = LLM centroid. Arrows show direction and magnitude of stylistic displacement. Gemini's arrow is ~75% longer than GPT-4's](https://raw.githubusercontent.com/riadmaouchi/llm-style-fingerprints/main/results/drift_trajectories.png)

The LDA cluster plot and the dendrogram both reinforce the same structure: two effective groups, with Claude as an ambiguous intermediate.

![LDA fingerprints — clusters in shift-vector space. The GPT-4 and Mistral ellipses are nearly identical in position and spread](https://raw.githubusercontent.com/riadmaouchi/llm-style-fingerprints/main/results/pca_clusters.png)

![Dendrogram — hierarchical clustering of individual shift vectors. The tree resolves into 2–3 groups, not 4](https://raw.githubusercontent.com/riadmaouchi/llm-style-fingerprints/main/results/dendrogram.png)

This does not mean GPT-4 and Mistral produce identical texts. Their function-word profiles differ in specific markers. What it means is that their *total displacement* from the human baseline — across 41 dimensions, averaged over 80 texts — is within noise at this corpus size.

---

## What the drift looks like in function-word space

The aggregate shifts conceal the directional structure. Looking at which specific function words deviate from the human baseline, per model:

- **GPT-4**: elevated *et* and *de* — simple additive coordination, functional syntax stripped of literary weight
- **Claude 3**: high *tandis*, *pourtant*, *néanmoins* — formal concessive and contrastive subordination, absent from the original Zola register
- **Mistral 7B**: distribution close to the human baseline across most markers — paraphrase that preserves surface statistics while altering content words
- **Gemini Pro**: high *en*, *afin*, *notamment* — analytical connectors, explicitly explanatory; converts literary atmosphere into reportage

The heatmap below shows the deviation from the human baseline per function word, in percent. Red = overuse relative to human baseline, blue = underuse. Reading across a row gives the model's stylistic pressure; reading down a column shows which models agree or diverge on a specific marker.

![Stylistic profiles — % deviation from human baseline per function word. Gemini's row shows the strongest positive deviations](https://raw.githubusercontent.com/riadmaouchi/llm-style-fingerprints/main/results/function_word_heatmap.png)

![SVM feature importance — most discriminative function words between models](https://raw.githubusercontent.com/riadmaouchi/llm-style-fingerprints/main/results/feature_importance.png)

These profiles are qualitatively consistent with what an attentive reader would notice in the rewrites. The value of the measurement is not that it *discovers* something invisible — it is that it makes the observation reproducible, quantitative, and testable across 80 texts rather than one.

---

## Can a classifier tell the models apart?

Given that there are measurable differences, can a classifier reliably attribute a rewrite to its model?

Eight feature configurations tested with leave-one-out cross-validation (LOO) on 320 examples (80 texts × 4 models), 4-class logistic regression, random baseline 25%:

| Method | Dim | Accuracy |
|--------|:---:|:--------:|
| **Surface stats extended + cosine shift ★** | 16 | **57.5%** |
| Surface stats extended (15 features) | 15 | 56.2% |
| Shift vectors (function words) | 41 | 40.6% |
| Surface stats basic (5 features) | 5 | 41.9% |
| Shift + surface basic | 46 | 43.8% |
| Raw rewrite FW vectors | 41 | 39.4% |
| Char n-grams TF-IDF (3–6) | 5000 | 33.1% |
| Original text vector (sanity) | 41 | 0.0% |
| Random baseline | — | 25.0% |

Four findings worth examining separately:

**Surface statistics dominate.** Mean sentence length, type-token ratio, punctuation density, comma frequency, em-dash usage, average word length — 15 structural features with zero grammatical sophistication — reach 56.2% alone. This is a striking result: the structural footprint of each model is more discriminative than its function-word signature.

**The scalar cosine shift adds +1.3 points.** A single number — the distance between the original and the rewrite in function-word space — lifts the combined accuracy to 57.5%. It encodes the *intensity* of the stylistic transformation, a signal absent from surface features.

**The shift subtraction over function words adds marginal value.** Shift vectors (rewrite minus original in 41-dim function-word space) score 40.6% vs 39.4% for raw rewrite frequencies. The delta helps slightly, but the scalar summary does most of the work.

**The dominant confusion is structural: GPT-4 ↔ Mistral.** This pair confuses the classifier across all methods, consistent with the permutation test result (p = 1.0 after Bonferroni correction). They produce structurally similar texts in this corpus.

![Confusion matrix — LOO classification, 4 classes. The GPT-4 / Mistral off-diagonal entries are the most frequent errors](https://raw.githubusercontent.com/riadmaouchi/llm-style-fingerprints/main/results/confusion_matrix.png)

![Shift distributions — KDE per model. GPT-4 and Mistral occupy nearly the same region of shift-magnitude space](https://raw.githubusercontent.com/riadmaouchi/llm-style-fingerprints/main/results/shift_distributions.png)

---

## Prompt robustness: aggregate vs per-text

The same 80 passages were rewritten under a second, stylistically different prompt (simpler register, general audience). This tests whether a model's stylistic signature is stable regardless of instruction — a prerequisite for treating it as a genuine fingerprint rather than a prompt artefact.

Pearson correlation between per-text shifts under P1 and P2, with bootstrap CIs (5,000 resamples):

| Model | r (P1 vs P2) | 95% CI | p |
|-------|:---:|:---:|:---:|
| GPT-4 | 0.28 | [0.04, 0.51] | 0.011 |
| Mistral | 0.28 | [−0.00, 0.51] | 0.011 |
| Gemini | 0.06 | [−0.17, 0.30] | 0.60 |
| Claude | 0.03 | [−0.21, 0.32] | 0.78 |

Two levels to read here.

The **aggregate ranking is stable**: Gemini produces the largest shift under both prompts; GPT-4 the smallest. The ordering survives the change of instruction.

The **per-text correlation is weak or null for all models**. For Gemini and Claude, the CI includes zero; p > 0.5. For GPT-4 and Mistral, the correlation is nominally significant but the CI is wide enough ([0.04, 0.51] for GPT-4) to describe a very noisy relationship. Knowing that a specific passage was strongly displaced under P1 tells you almost nothing about how it will be displaced under P2.

![Per-text robustness — P1 shift vs P2 shift per passage, per model. Wide scatter = low per-text consistency](https://raw.githubusercontent.com/riadmaouchi/llm-style-fingerprints/main/results/prompt_robustness.png)

The implication: **stylistic fingerprints here are distributional properties of models, not stable per-document signatures**. The aggregate is real. The individual-text signal is not reliable enough for attribution claims.

---

## A methodological note on code stylometry

The project uses literary text. The same approach — shift vectors in a function-token space — applies directly to source code.

Developers leave stylistic fingerprints in code through semi-automatic choices: naming conventions (`camelCase` vs `snake_case`; `get_` vs `fetch_` prefix habits), type annotation density, docstring style, import grouping conventions, blank line usage around conditionals, tendency to use comprehensions vs explicit loops, preference for early returns vs nested conditionals.

These are the code equivalents of function words: choices made below the level of deliberate reflection, accumulated across thousands of decisions per codebase.

A concrete example: two implementations of the same function.

```python
# Developer A — dense type hints, early return, no intermediate variable
def find_user(user_id: int, db: Session) -> Optional[User]:
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()

# Developer B — verbose, intermediate variable, no type hints
def find_user(user_id, db):
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        return user
    return None
```

Both are correct. The differences — type hints, early return, intermediate variable — are stylistic habits. They are statistically stable across a developer's codebase and partially identifiable without reading the logic.

As Copilot and similar tools become standard, the question becomes: does AI-assisted code develop characteristic patterns? Do completions from a specific model tend to favour `Optional[T]` over `T | None`, `enumerate` over manual indexing, inline f-strings over `.format()`? Do these tendencies accumulate at the file or project level into a measurable stylistic pressure?

Applying shift vectors to code would require rebuilding the "function-word" vocabulary from scratch — not grammatical particles but syntactic tokens: annotation markers, structural keywords, naming affixes. The methodology would transfer. The vocabulary would need to be constructed empirically, probably from a large mixed corpus of human and AI-assisted code.

This is unexplored. The project does not have code data; this section is speculative. But it is tractable and interesting — and a cleaner framing than "detecting Copilot", which carries forensic implications the method cannot support.

The figure below is explicitly synthetic — it illustrates what such an experiment might look like, using two hypothetical axes (type annotation density, early-return rate) and three developer clusters. The arrows show the direction of stylistic pressure from AI-assisted completions: not convergence to a single point, but a measurable bias toward a particular stylistic region.

![Code stylometry (synthetic/illustrative) — developer style fingerprints in a 2D structural-habit space. Arrows show the direction of AI-assisted drift. Axes are real measurable quantities, data is not from the project](https://raw.githubusercontent.com/riadmaouchi/llm-style-fingerprints/main/results/code_stylometry.png)

---

## Limitations

These results come from a narrow experimental setup. Several constraints matter:

**Single language and register.** 19th-century French literary prose. Signals in this register may not transfer to contemporary French, English, technical writing, or conversational text. Stylometric signals are known to degrade across domain boundaries.

**The function-word list introduces a prior.** The 41-word vocabulary was assembled with LLM stylistic markers in mind. Words like *tandis*, *pourtant*, *néanmoins*, *notamment* were selected partly because they were hypothesised to be characteristic of formal LLM output. This structurally advantages detection of models that skew formal — Gemini's strong showing may partially reflect vocabulary design, not only model behaviour.

**Corpus size limits resolution.** 80 originals × 4 models = 320 examples. The classification results are directionally meaningful but noisy at this scale. The wide bootstrap CIs on the Pearson correlations document their own uncertainty; they should be read as ranges, not point estimates.

**Model versions are snapshots.** The rewrites were generated at a specific moment with specific checkpoints (gpt-4o, claude-sonnet-4-6, mistral-small-latest, gemini-2.5-flash). All four models are updated continuously. The stylistic signatures measured here are not permanent properties of model families.

**No human control condition.** The comparison is between LLM rewrites and literary originals — not between LLM rewrites and human rewrites following the same instruction. Without the human paraphrase baseline, it is difficult to separate "LLM stylistic pressure" from "what following this instruction does to any text".

**Individual-text attribution is not reliable.** This point deserves emphasis: the aggregate patterns are real and statistically supported. Per-document attribution is not. The method should not be used to claim that any specific text was written by a specific model.

---

## Open research questions

The experiment raises more questions than it answers. A partial list:

1. Do stylistic shift magnitudes vary with model *size* within a family (Mistral 7B vs Mistral Large, GPT-4o-mini vs GPT-4o)?
2. Does instruction-tuning intensity — measured by RLHF data volume or fine-tuning steps — correlate with homogenisation?
3. Do shifts in function-word space correlate with shifts in embedding space? Are the two signals measuring the same underlying transformation?
4. Would the GPT-4/Mistral indistinguishability hold with a larger corpus (n=500)? Or does it reflect a genuine equivalence at this register?
5. Do the stylistic signatures hold across languages? Would French and English literary corpora produce the same ordinal ranking?
6. Is stylistic drift larger or smaller for domain-specific fine-tuned models vs general-purpose models?
7. Does prompt engineering (e.g., explicitly instructing a model to preserve the original's style) reduce the measured shift?
8. Do AI coding assistants reduce stylistic variance within a codebase over time, as more code gets completed by the same model?
9. Can a code formatter (Black, Prettier) erase developer fingerprints measured in the structural markers the formatter does not touch (naming, annotation density, comprehension preference)?
10. Do stylistic fingerprints cluster differently when measured on outputs vs on model weights — are the fingerprints a training artefact or an inference artefact?
11. Would a model fine-tuned on Zola's prose reduce its measured shift toward zero — and what would that tell us about what the shift is measuring?
12. Is the shift magnitude correlated with human perceptual distance — do readers judge higher-shift rewrites as more stylistically alien?

Most of these are tractable experiments. Some are well-defined enough to run on this corpus with a few days of additional data collection.

---

## What this suggests

The finding that survives the caveats most clearly: **LLMs appear to transform literary style more than they erase it**.

The original text's function-word distribution — its particular balance of *et*, *que*, *dans*, *pourtant*, *néanmoins* — does not survive the rewrite. But it is not simply deleted. It is displaced: moved to a region of style-space characteristic of the model doing the rewriting. Different models move it to different regions. The regions partially overlap. The displacement is consistent at the aggregate level across 80 texts.

The fingerprints are real but limited. At 57.5% accuracy over 4 classes — more than double the 25% random baseline — they do not support reliable individual attribution. They are, as the analysis shows, better understood as **statistical attractors** — tendencies in the direction of drift — than as unique identifiers. The signal lives in surface structure (sentence rhythm, vocabulary richness, punctuation habits), not only in grammatical function words.

What makes this interesting is not the classification number. It is that the drift exists at all, that it is directional, and that it is consistent enough to survive across two distinct prompts at the aggregate level. Each model appears to have preferred regions of style-space it moves text toward, independently of what the original text was.

Whether this reflects training data composition, instruction-tuning objectives, or something else is an open question. What this experiment does is establish that it is measurable — with a 41-word vocabulary, cosine distance, and 80 public-domain literary texts.

---

## Further reading

- Mosteller & Wallace (1964). *Inference and Disputed Authorship: The Federalist.* — The methodological founding of statistical stylometry.
- Juola (2015). *The Rowling Case.* Digital Scholarship in the Humanities. — Stylometry applied to a real pseudonymity question.
- Stamatatos (2009). *A survey of modern authorship attribution methods.* JASIST 60(3). — Still the most comprehensive survey of the field.
- Koppel, Schler & Argamon (2009). *Computational methods in authorship attribution.* JASIST 60(1). — Function-word methods in depth.
- Sadasivan et al. (2023). *Can AI-Generated Text be Reliably Detected?* arXiv:2303.11156. — The theoretical argument for why binary detection is hard.
- Liang et al. (2024). *GPT detectors are biased against non-native English writers.* Patterns 5(7). — Documented harm from over-confident detection.
- Uchendu et al. (2023). *TURINGBENCH.* EMNLP Findings. — Multi-model attribution benchmark.

---

**GitHub**: [riadmaouchi/llm-style-fingerprints](https://github.com/riadmaouchi/llm-style-fingerprints) — code, data, notebooks, 21 tests  
**Paper**: [doi.org/10.5281/zenodo.20402754](https://doi.org/10.5281/zenodo.20402754)  
**Notebooks** (no install): [Binder](https://mybinder.org/v2/gh/riadmaouchi/llm-style-fingerprints/HEAD?urlpath=lab/tree/notebooks/01_shift_analysis.ipynb) · [nbviewer](https://nbviewer.org/github/riadmaouchi/llm-style-fingerprints/blob/main/notebooks/01_shift_analysis.ipynb)

All LLM outputs are pre-generated in the repository. No API key required.
