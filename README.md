# llm-style-fingerprints

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/riadmaouchi/llm-style-fingerprints/HEAD?urlpath=lab/tree/notebooks/01_shift_analysis.ipynb)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/riadmaouchi/llm-style-fingerprints/blob/main/notebooks/01_shift_analysis.ipynb)
[![Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen)](tests/test_stats.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI - stylometry-python](https://img.shields.io/pypi/v/stylometry-python?label=stylometry-python)](https://pypi.org/project/stylometry-python/)
[![GitHub stars](https://img.shields.io/github/stars/riadmaouchi/llm-style-fingerprints?style=social)](https://github.com/riadmaouchi/llm-style-fingerprints/stargazers)

**Can ChatGPT erase Zola's writing style?**

This project measures the stylistic drift introduced by LLM rewrites —
using function-word fingerprints and cosine distance in a 57-dimension style space.
GPT-4, Claude 3, Mistral 7B and Gemini Pro produce measurably distinct stylistic signatures,
consistent across texts, temperatures, and prompts.

> *"LLMs don't erase literary style. They replace it with their own."*

| ![Human vs LLM stylistic clusters](results/pca_clusters.png) | ![Function-word fingerprints per model](results/radar_profiles.png) |
|:---:|:---:|
| *Human vs LLM stylistic clusters (PCA)* | *Function-word fingerprints per model* |

---

## Un texte. Quatre réécritures. Quatre styles distincts.

### Original — Zola, *Germinal*

> *Quand Étienne Lantier descendit dans la fosse, il faisait nuit encore. La pluie tombait par rafales sur la plaine rase, un vent du nord-est soufflait, glacial. L'homme s'arrêta au bord du puits, la lampe à la main, et regarda l'abîme. Au fond, dans les ténèbres, on entendait gronder la machine.*

### GPT-4 — neutralisation factuelle, "et" coordinateur dominant

> *Étienne Lantier descendait dans la mine alors qu'il faisait encore nuit. Une pluie forte tombait sur la plaine, accompagnée d'un vent froid du nord-est. Il s'arrêta au bord du puits avec sa lampe et observa l'obscurité. On entendait le bruit de la machine en contrebas.*

### Claude 3 — subordonnées formelles, "tandis que", "bien que"

> *Tandis qu'Étienne Lantier descendait dans la fosse, la nuit n'était pas encore levée. Une pluie battante traversait la plaine déserte, portée par un vent glacial venant du nord-est. Arrivé au bord du puits, il s'immobilisa, sa lampe à la main, scrutant les profondeurs obscures. Du fond montait le grondement sourd de la machine.*

### Mistral 7B — reformulation directe, structure proche, vocabulaire simple

> *Étienne Lantier descendait dans la mine. Il faisait encore nuit. La pluie tombait fort sur la plaine et le vent soufflait du nord-est. Il s'est arrêté au bord du puits avec sa lampe et a regardé en bas. On pouvait entendre la machine gronder au fond.*

### Gemini Pro — liens causaux explicites, "en raison de", "afin de", "notamment"

> *Alors qu'Étienne Lantier entamait sa descente dans la fosse, l'obscurité de la nuit régnait encore. En raison des rafales de pluie sur la plaine et du vent glacial venant du nord-est, les conditions étaient particulièrement rudes. Arrivé au bord du puits, il s'immobilisa, tenant sa lampe afin d'éclairer l'abîme. Le grondement de la machine montait des profondeurs.*

---

## Résultats principaux

| Modèle | Shift moyen | IC 95 % Bootstrap |
|--------|:-----------:|:-----------------:|
| Gemini Pro | **0.205** | [0.176, 0.238] |
| Claude 3 | **0.192** | [0.160, 0.227] |
| Mistral 7B | **0.146** | [0.124, 0.172] |
| GPT-4 | **0.146** | [0.119, 0.176] |

**Résultat principal :** Gemini introduit le shift stylistique le plus fort ; GPT-4 et Mistral sont statistiquement indistinguables (p = 1.0 après correction Bonferroni). Le signal est cohérent sur n = 50 textes (25 Zola + 25 Maupassant), avec des réécritures générées par les APIs réelles.

| Classifieur | Accuracy | Baseline |
|------------|:--------:|:--------:|
| Centroïde LOO — 4 classes | **24.5%** | 25% |
| Régression logistique LOO — 4 classes | **36.5%** | 25% |

---

## Pourquoi c'est important

L'attribution d'auteur est étudiée depuis des décennies (Mosteller & Wallace, 1964 — Federalist Papers).
Mais l'essor de l'écriture assistée par LLM soulève une nouvelle question :

*Si un texte a été partiellement réécrit par un LLM, y a-t-il un shift stylistique mesurable ?*

Cette étude fournit une méthodologie et des résultats de référence pour répondre à cette question.

**Applications :**
- Intégrité académique : détection de soumissions assistées par LLM
- Stylométrie forensique : signalement de contenu probablement généré par IA
- Compréhension : qu'est-ce que les LLMs modifient exactement dans un texte ?

---

## Méthodologie

### Vecteur de style

Chaque texte est représenté comme un **vecteur de 57 dimensions** de fréquences de mots-outils
(articles, pronoms, prépositions, conjonctions — mots qui portent le style, pas le contenu).

```python
FUNCTION_WORDS_FR = [
    'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du',
    'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car',
    'que', 'qui', 'dont', 'dans', 'sur', 'avec', 'sans',
    'il', 'elle', 'ils', 'elles', 'je', 'tu', 'nous', 'vous',
    'ne', 'pas', 'plus', 'très', 'bien', 'tout', 'rien',
    'tandis', 'pourtant', 'néanmoins', 'notamment', 'afin',  # marqueurs LLM
    # ... 57 mots-outils au total
]
```

Les vecteurs sont normalisés L2. La distance entre deux textes = **distance cosinus**.

### Pourquoi les mots-outils ?

Les mots-outils sont :
- **Inconscients** — les auteurs ne les choisissent pas délibérément
- **Indépendants du sujet** — stables entre domaines (même auteur, sujets différents)
- **Robustes** — difficiles à dissimuler intentionnellement

Juola (2015) a utilisé 4 telles caractéristiques pour identifier JK Rowling derrière le pseudonyme Robert Galbraith.

### Corpus

| Source | Textes | Mots/texte | Langue |
|--------|:------:|:----------:|--------|
| Zola (Germinal, L'Assommoir, Nana…) | 25 | ~120 | Français |
| Maupassant (nouvelles diverses) | 25 | ~120 | Français |
| Réécritures GPT-4o | 50 | ~120 | Français |
| Réécritures Claude Sonnet 4.6 | 50 | ~120 | Français |
| Réécritures Mistral Small | 50 | ~120 | Français |
| Réécritures Gemini 2.5 Flash | 50 | ~120 | Français |

**Prompt de réécriture :** *"Réécris ce texte dans un style neutre et factuel, en conservant le sens."*

Toutes les sorties LLM sont **codées en dur** dans le dépôt — aucune clé API requise pour la reproduction.

### Mesure du shift

Pour chaque texte original `t` et sa réécriture LLM `t'` :

```
shift(t, t') = distance_cosinus(v(t), v(t'))
```

Un shift de 0 = vecteur de style identique. Un shift de 1 = vecteurs de style maximalement différents.

---

## Résultats

### 1. Les LLMs forment des clusters stylistiques distincts

| | PCA (linéaire) | t-SNE (non-linéaire) |
|-|:-:|:-:|
| | ![PCA](results/pca_clusters.png) | ![t-SNE](results/tsne_clusters.png) |

- **Humains** → dispersés (styles individuels distincts — variance intra élevée)
- **GPT-4 + Mistral 7B** → shifts similaires (~0.146), statistiquement indistinguables
- **Claude 3** → shift intermédiaire (~0.192), zone de chevauchement
- **Gemini Pro** → shift le plus fort (~0.205), seul significativement différent de GPT-4 et Mistral

Le dendrogramme confirme la structure à **2–3 groupes effectifs**, pas 4 :

![Dendrogramme](results/dendrogram.png)

### 2. Le shift est cohérent et mesurable — avec des nuances statistiques

![Bootstrap CI](results/bootstrap_ci.png)

| Modèle | Shift moyen | IC 95 % Bootstrap |
|--------|:-----------:|:-----------------:|
| Gemini Pro | **0.205** | [0.176, 0.238] |
| Claude 3 | **0.192** | [0.160, 0.227] |
| Mistral 7B | **0.146** | [0.124, 0.172] |
| GPT-4 | **0.146** | [0.119, 0.176] |

**Tests pairés (correction Bonferroni, 6 paires) :**

| Paire | p corrigé | Significatif |
|-------|:---------:|:------------:|
| Mistral 7B vs Gemini Pro | 0.025 | ✓ |
| GPT-4 vs Gemini Pro | 0.044 | ✓ |
| Claude 3 vs Mistral 7B | 0.212 | ✗ |
| GPT-4 vs Claude 3 | 0.282 | ✗ |
| GPT-4 vs Mistral 7B | 1.000 | ✗ |
| Claude 3 vs Gemini Pro | 1.000 | ✗ |

**⚠️ Point critique :** Seul Gemini se distingue significativement de GPT-4 et Mistral.
GPT-4 et Mistral sont **indistinguables** (p = 1.0) — même shift moyen (0.146).

![Distributions des shifts](results/shift_distributions.png)

**Profil stylistique par modèle :**

![Radar](results/radar_profiles.png)

| Modèle | Signature caractéristique |
|--------|---------------------------|
| GPT-4 | Forte utilisation de `et`, `de` — neutralisation factuelle |
| Claude 3 | Surreprésentation de `tandis`, `pourtant`, `néanmoins` |
| Mistral 7B | Distribution proche du profil humain — paraphrase de surface |
| Gemini Pro | Usage élevé de `en`, `raison`, `afin`, `notamment` — style analytique |

![Empreinte stylistique](results/fingerprint_comparison.png)

### 3. Cohérence intra-groupe : Gemini est le plus homogène

Un résultat contre-intuitif : **Gemini Pro est le LLM stylistiquement le plus cohérent**
(variance intra = 0.428), plus que GPT-4 ou Claude 3.
Les auteurs humains sont les plus variables, ce qui est attendu.

### 4. Classification inter-LLM

Classifieur par centroïde, Leave-One-Out, 4 classes :

![Matrice de confusion](results/confusion_matrix.png)

- Accuracy centroïde : **24.5 %** ≈ baseline 25 % → centroïde seul insuffisant
- Accuracy logistique (LOO) : **36.5 %** → signal faible mais supérieur au hasard
- Confusion principale : **GPT-4 ↔ Mistral** (shifts identiques)

![Mots-outils discriminants](results/feature_importance.png)

### 5. Shift vs longueur du texte

![Shift vs longueur](results/shift_vs_length.png)

Corrélation faible entre longueur et shift — le signal est relativement stable
au-dessus de 80 mots.

---

## Synthèse interprétative

### Ce que l'étude établit de manière robuste

> **Tous les LLMs testés déplacent le style du texte original.**
> Ce déplacement est mesurable et cohérent sur n = 50 textes authentiques (Zola + Maupassant),
> générés via les APIs réelles de chaque modèle.

Concrètement, il y a **2 groupes significativement distincts**, pas 4 :

```
GPT-4 ≈ Mistral 7B   ←——————→   Gemini Pro
(shift ~0.146)                   (shift ~0.205)
       ↑
  Claude 3 (~0.192) — zone grise, pas significativement différent de l'un ni de l'autre
```

Gemini impose le changement stylistique le plus fort — seul modèle significativement différent de GPT-4 et de Mistral.
GPT-4 et Mistral sont **indistinguables** dans ce protocole (p = 1.0).

### Ce que l'étude n'établit pas

| Affirmation | Statut |
|-------------|--------|
| "GPT-4 et Mistral ont des styles différents" | ✗ non supporté (p=1.0 après correction) |
| "Ces résultats se généralisent à l'anglais" | ✗ non testé |
| "On peut identifier un LLM en pratique" | ⚠️ accuracy 36.5% avec 4 classes → insuffisant |
| "Le signal est robuste sur des textes très courts" | ⚠️ dégradation observée sous 80 mots |
| "Ces signatures tiennent avec d'autres prompts" | ✗ non testé — critique |

---

## Limitations

> **Ceci est une exploration méthodologique, pas un détecteur de LLM.**

**Corpus encore trop petit pour des conclusions fermes.**
n = 50 textes par classe. Seul Gemini se distingue significativement de GPT-4 et Mistral. GPT-4, Claude 3 et Mistral ne sont pas distinguables entre eux au niveau Bonferroni. Il faudrait n ≥ 100 avec plusieurs prompts pour des résultats pleinement robustes.

**Un seul prompt, un seul registre.**
Toutes les réécritures utilisent le même prompt (*"style neutre et factuel"*) appliqué à du français littéraire du 19e siècle. Ce prompt pousse tous les modèles dans la même direction. Avec un prompt créatif ou un corpus journalistique, les signatures seraient différentes — et probablement plus ou moins marquées.

**Le vocabulaire est biaisé vers les LLMs.**
La liste de mots-outils inclut des marqueurs intentionnellement choisis pour capturer les patterns LLM (*tandis, pourtant, néanmoins, notamment*). Ce choix n'est pas neutre : il avantage structurellement la détection des modèles les plus "formels".

**Pas de contrôle humain "en mode neutre".**
L'étude compare des textes originaux de Zola avec des réécritures LLM. Elle ne teste pas ce que donnerait le même prompt appliqué à un humain. Un rédacteur humain demandé d'écrire "dans un style neutre et factuel" produirait peut-être un shift similaire à Mistral.

**Modèles fixes, signatures évolutives.**
gpt-4o, claude-sonnet-4-6, mistral-small-latest et gemini-2.5-flash sont des versions figées au moment de la collecte. Les modèles sont mis à jour en continu — ces signatures ne sont pas permanentes.

**Ne pas utiliser pour accuser.**
La variance intra-groupe (0.43–0.65) est trop élevée pour des conclusions sur des textes individuels. Ce signal n'est pas une preuve d'authorship.

---

## Explorer sans installation

| Notebook | nbviewer (statique) | Binder (interactif) |
|----------|--------------------|--------------------|
| 01 — Shift stylistique | [voir](https://nbviewer.org/github/riadmaouchi/llm-style-fingerprints/blob/main/notebooks/01_shift_analysis.ipynb) | [lancer](https://mybinder.org/v2/gh/riadmaouchi/llm-style-fingerprints/HEAD?urlpath=lab/tree/notebooks/01_shift_analysis.ipynb) |
| 02 — Classification inter-LLM | [voir](https://nbviewer.org/github/riadmaouchi/llm-style-fingerprints/blob/main/notebooks/02_classification.ipynb) | [lancer](https://mybinder.org/v2/gh/riadmaouchi/llm-style-fingerprints/HEAD?urlpath=lab/tree/notebooks/02_classification.ipynb) |
| 03 — Robustesse et limites | [voir](https://nbviewer.org/github/riadmaouchi/llm-style-fingerprints/blob/main/notebooks/03_robustesse.ipynb) | [lancer](https://mybinder.org/v2/gh/riadmaouchi/llm-style-fingerprints/HEAD?urlpath=lab/tree/notebooks/03_robustesse.ipynb) |

> **nbviewer** : rendu instantané, pas de compte requis.  
> **Binder** : environnement complet exécutable (~1 min de démarrage), pas d'installation locale.

## Démarrage rapide (local)

```bash
git clone https://github.com/riadmaouchi/llm-style-fingerprints
cd llm-style-fingerprints
make install

make fast      # figures en < 30 s
make test      # 21 tests
```

Aucune clé API requise. Toutes les sorties LLM sont pré-générées dans `data/`.

### Utiliser l'API

```python
from src.stylometry import StyleAnalyzer

sa = StyleAnalyzer()

# Mesurer le shift entre un original et une réécriture
original = "Quand Étienne Lantier descendit dans la fosse..."
rewrite  = "Étienne Lantier descendait dans la mine alors que..."
print(f"Shift : {sa.shift(original, rewrite):.4f}")

# Visualisation PCA multi-groupes
fig = sa.plot_clusters(
    texts_groups=[human_texts, gpt4_texts, claude_texts],
    labels=['Humain', 'GPT-4', 'Claude 3'],
)
fig.savefig('clusters.png', dpi=150, bbox_inches='tight')
```

---

## Structure du dépôt

```
llm-style-fingerprints/
├── src/
│   ├── __init__.py
│   ├── stylometry.py    # Wrapper de stylometry-python + plot_shift_violin()
│   └── stats.py         # bootstrap_ci, permutation_test, pairwise_tests, intra_variance
├── data/
│   ├── human/
│   │   ├── zola.json          # 25 extraits (Germinal, L'Assommoir, Nana…)
│   │   └── maupassant.json    # 25 nouvelles (Boule de suif, La Parure…)
│   ├── gpt4/rewrites.json     # 50 réécritures gpt-4o (API réelle)
│   ├── claude3/rewrites.json  # 50 réécritures claude-sonnet-4-6 (API réelle)
│   ├── mistral/rewrites.json  # 50 réécritures mistral-small-latest (API réelle)
│   └── gemini/rewrites.json   # 50 réécritures gemini-2.5-flash (API réelle)
├── notebooks/
│   ├── 01_shift_analysis.ipynb  # Shifts, violin, PCA, t-SNE, fingerprint
│   ├── 02_classification.ipynb  # Confusion matrix, SVM, courbe d'apprentissage
│   └── 03_robustesse.ipynb      # Bootstrap CI, permutation tests, intra-variance
├── results/
│   ├── pca_clusters.png
│   ├── tsne_clusters.png
│   ├── umap_clusters.png
│   ├── shift_distributions.png
│   ├── bootstrap_ci.png
│   ├── permutation_null.png
│   ├── fingerprint_comparison.png
│   ├── radar_profiles.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── logistic_feature_importance.png
│   ├── dendrogram.png
│   ├── shift_by_source.png
│   └── shift_vs_length.png
├── generate_results.py   # Régénère les 14 PNG sans Jupyter
├── scripts/
│   └── generate_rewrites.py  # Appelle les APIs LLM pour régénérer les données
├── requirements.txt
└── README.md
```

---

## Travaux connexes

### Stylométrie classique
- Mosteller & Wallace (1964). *Inference and disputed authorship: The Federalist.* Addison-Wesley. *(fondateur du domaine)*
- Burrows (1987). *Word patterns and story shapes.* Literary and Linguistic Computing 2(2). *(Delta method)*
- Juola (2015). *The Rowling Case: A Proposed Standard Analytic Protocol for Authorship Questions.* DSH 30(S1).
- Stamatatos (2009). *A survey of modern authorship attribution methods.* JASIST 60(3).
- Koppel, Schler & Argamon (2009). *Computational methods in authorship attribution.* JASIST 60(1).
- Kestemont et al. (2020). *Overview of the Cross-Domain Authorship Verification Task.* PAN @ CLEF 2020.

### LLMs et style
- Uchendu et al. (2023). *TURINGBENCH: A benchmark environment for Turing test in the age of neural text generation.* EMNLP Findings.
- Guo et al. (2023). *How Close is ChatGPT to Human Experts?* arXiv:2301.07597.
- Liang et al. (2024). *GPT detectors are biased against non-native English writers.* Patterns 5(7). *(faux positifs critiques)*
- Sadasivan et al. (2023). *Can AI-Generated Text be Reliably Detected?* arXiv:2303.11156. *(limites fondamentales)*
- Zhu et al. (2023). *Beat LLM-based text detection by casual paraphrasing.* arXiv:2305.10714.
- Cafiero et al. (2023). *Demystifying QAnon: Identifying Coordinated Networks via Authorship Attribution.* arXiv:2303.02078.
- Kapusta et al. (2025). *False positives in LLM detection for non-native speakers.* arXiv:2512.06922.

---

## Citation

```bibtex
@misc{llm-style-fingerprints-2025,
  title  = {llm-style-fingerprints: Measuring stylistic regularities in LLM-generated text},
  author = {Riad Maouchi},
  year   = {2025},
  url    = {https://github.com/riadmaouchi/llm-style-fingerprints}
}
```

---

## Licence

MIT. Textes du corpus (Zola, Maupassant) dans le domaine public.
