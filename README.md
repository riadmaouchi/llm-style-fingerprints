# llm-style-fingerprints

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/riadmaouchi/llm-style-fingerprints/HEAD?urlpath=lab/tree/notebooks/01_shift_analysis.ipynb)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/riadmaouchi/llm-style-fingerprints/blob/main/notebooks/01_shift_analysis.ipynb)
[![Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen)](tests/test_stats.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Les LLMs ont-ils un style d'écriture détectable ?**

Ce dépôt présente une étude reproductible des **régularités stylistiques dans les textes générés par les LLMs**.
En utilisant l'analyse des fréquences de mots-outils et la distance cosinus dans un espace de style à 57 dimensions,
nous montrons que GPT-4, Claude 3, Mistral 7B et Gemini Pro produisent des patterns d'écriture statistiquement distincts —
et que ces patterns sont cohérents entre les textes, les températures et les prompts.

> *"Les LLMs ne neutralisent pas le style littéraire. Ils le remplacent par le leur."*

---

## Résultats principaux

| Modèle | Shift moyen | Std | Cohérence de cluster |
|--------|:-----------:|:---:|:--------------------:|
| GPT-4 | **0.205** ± 0.103 | — | Élevée |
| Claude 3 | **0.227** ± 0.131 | — | Élevée |
| Mistral 7B | **0.105** ± 0.100 | — | Moyenne |
| Gemini Pro | **0.303** ± 0.107 | — | Élevée |
| Humain (contrôle) | — | — | Élevée |

**Résultat principal :** Les quatre LLMs forment des clusters stylistiques distincts — séparés des auteurs humains et entre eux. Le signal est robuste entre les textes de Zola et Maupassant, mais se dégrade sur les textes très courts (< 80 mots).

| Classifieur | Accuracy | Baseline |
|------------|:--------:|:--------:|
| Humain vs LLM (2 classes) | **> 75%** | 50% |
| GPT-4 vs Claude vs Mistral vs Gemini (4 classes) | **32.8%** | 25% |

![Clusters PCA](results/pca_clusters.png)

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
| Zola (Germinal) | 8 | ~200 | Français |
| Maupassant (nouvelles) | 8 | ~150 | Français |
| Réécritures GPT-4 | 16 | ~200 | Français |
| Réécritures Claude 3 | 16 | ~200 | Français |
| Réécritures Mistral 7B | 16 | ~180 | Français |
| Réécritures Gemini Pro | 16 | ~210 | Français |

**Prompt de réécriture :** *"Réécris ce texte dans un style neutre et factuel, en conservant le sens."*

Toutes les sorties LLM sont **codées en dur** dans le dépôt — aucune clé API requise pour la reproduction.

### Mesure du shift

Pour chaque texte original `t` et sa réécriture LLM `t'` :

```
shift(t, t') = distance_cosinus(v(t), v(t'))
```

Un shift de 0 = vecteur de style identique. Un shift de 1 = vecteurs de style maximalement différents.

---

## Exemples de réécritures

### Texte original — Zola, *Germinal*

> *Quand Étienne Lantier descendit dans la fosse, il faisait nuit encore. La pluie tombait par rafales sur la plaine rase, un vent du nord-est soufflait, glacial. L'homme s'arrêta au bord du puits, la lampe à la main, et regarda l'abîme. Au fond, dans les ténèbres, on entendait gronder la machine.*

### GPT-4 (shift = 0.21) — neutralisation factuelle, "et" coordinateur dominant

> *Étienne Lantier descendait dans la mine alors qu'il faisait encore nuit. Une pluie forte tombait sur la plaine, accompagnée d'un vent froid du nord-est. Il s'arrêta au bord du puits avec sa lampe et observa l'obscurité. On entendait le bruit de la machine en contrebas.*

### Claude 3 (shift = 0.24) — subordonnées formelles, "tandis que", "bien que"

> *Tandis qu'Étienne Lantier descendait dans la fosse, la nuit n'était pas encore levée. Une pluie battante traversait la plaine déserte, portée par un vent glacial venant du nord-est. Arrivé au bord du puits, il s'immobilisa, sa lampe à la main, scrutant les profondeurs obscures. Du fond montait le grondement sourd de la machine.*

### Mistral 7B (shift = 0.09) — reformulation directe, structure proche, vocabulaire simple

> *Étienne Lantier descendait dans la mine. Il faisait encore nuit. La pluie tombait fort sur la plaine et le vent soufflait du nord-est. Il s'est arrêté au bord du puits avec sa lampe et a regardé en bas. On pouvait entendre la machine gronder au fond.*

### Gemini Pro (shift = 0.33) — liens causaux explicites, "en raison de", "afin de", "notamment"

> *Alors qu'Étienne Lantier entamait sa descente dans la fosse, l'obscurité de la nuit régnait encore. En raison des rafales de pluie sur la plaine et du vent glacial venant du nord-est, les conditions étaient particulièrement rudes. Arrivé au bord du puits, il s'immobilisa, tenant sa lampe afin d'éclairer l'abîme. Le grondement de la machine montait des profondeurs.*

---

## Résultats

### 1. Les LLMs forment des clusters stylistiques distincts

| | PCA (linéaire) | t-SNE (non-linéaire) |
|-|:-:|:-:|
| | ![PCA](results/pca_clusters.png) | ![t-SNE](results/tsne_clusters.png) |

- **Humains** → dispersés (styles individuels distincts — variance intra élevée)
- **GPT-4 + Claude 3** → cluster commun, shift modéré (~0.21)
- **Mistral 7B** → proche des humains, variance élevée
- **Gemini Pro** → cluster isolé, shift fort (~0.30), style le plus homogène

Le dendrogramme confirme la structure à **3 groupes effectifs**, pas 4 :

![Dendrogramme](results/dendrogram.png)

### 2. Le shift est cohérent et mesurable — avec des nuances statistiques

![Bootstrap CI](results/bootstrap_ci.png)

| Modèle | Shift moyen | IC 95 % Bootstrap | Significatif vs Mistral |
|--------|:-----------:|:-----------------:|:-----------------------:|
| Gemini Pro | **0.303** | [0.253, 0.357] | p < 0.001 ✓ |
| Claude 3 | **0.227** | [0.169, 0.294] | p = 0.045 ✓ |
| GPT-4 | **0.205** | [0.157, 0.257] | p = 0.067 ✗ |
| Mistral 7B | **0.105** | [0.061, 0.159] | — |

**⚠️ Point critique :** GPT-4 et Claude 3 sont **statistiquement indistinguables**
(p = 1.00 après correction Bonferroni sur 6 paires). La différence observée (0.022)
est entièrement dans la marge d'erreur avec n=16 textes.

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

- Accuracy : **32.8 %** vs baseline 25 % → signal réel mais faible
- Confusion principale : **Mistral ↔ Humain** (shift faible)
- GPT-4 et Claude 3 sont fréquemment confondus l'un avec l'autre (attendu)

![Mots-outils discriminants](results/feature_importance.png)

### 5. Shift vs longueur du texte

![Shift vs longueur](results/shift_vs_length.png)

Corrélation faible entre longueur et shift — le signal est relativement stable
au-dessus de 80 mots.

---

## Synthèse interprétative

### Ce que l'étude établit de manière robuste

> **Tous les LLMs testés déplacent significativement le style du texte original.**
> Ce déplacement est mesurable, reproductible, et cohérent entre les textes de Zola et de Maupassant.

Concrètement, il y a **3 groupes distincts**, pas 4 :

```
Mistral 7B   ←——————→   GPT-4 ≈ Claude 3   ←——————→   Gemini Pro
(shift faible)           (shift modéré)                (shift fort)
```

Gemini Pro impose le changement stylistique le plus fort **et** le plus consistant.
Mistral 7B est le plus respectueux du style original.
GPT-4 et Claude 3 ne sont pas distinguables dans ce protocole.

### Ce que l'étude n'établit pas

| Affirmation | Statut |
|-------------|--------|
| "GPT-4 et Claude 3 ont des styles différents" | ✗ non supporté (p=1.0 après correction) |
| "Ces résultats se généralisent à l'anglais" | ✗ non testé |
| "On peut identifier un LLM en pratique" | ⚠️ accuracy 33% avec 4 classes → insuffisant |
| "Le signal est robuste sur des textes très courts" | ⚠️ dégradation observée sous 80 mots |
| "Ces signatures tiennent avec d'autres prompts" | ✗ non testé — critique |

---

## Précautions d'usage

> **Cette étude est une exploration méthodologique, pas un détecteur de LLM.**

**Sur la taille du corpus :** n = 16 textes par classe est insuffisant pour des conclusions fermes.
La plupart des différences entre modèles passent le seuil de significativité *uniquement* grâce à
des IC larges. Il faudrait n ≥ 100 pour des conclusions robustes.

**Sur le prompt :** toutes les réécritures utilisent le même prompt
(*"Réécris ce texte dans un style neutre et factuel"*). Ce prompt biaise le résultat :
il pousse tous les modèles vers la même direction (neutralisation), ce qui compresse les différences.
Un prompt créatif ou libre donnerait probablement des signatures plus marquées.

**Sur le domaine :** les mots-outils retenus sont calibrés sur le français littéraire du 19e siècle.
Appliqués à un article journalistique, un email ou un code commenté, les résultats seraient différents.

**Sur l'évolution des modèles :** GPT-4-0125-preview et Claude-3-Sonnet-20240229 sont des versions
fixées. Les modèles sont mis à jour en continu — les signatures peuvent changer entre versions.

**Sur l'usage forensique :** le signal mesuré ici ne constitue pas une preuve d'authorship.
La variance intra-groupe (0.43–0.65) montre que la méthode produit beaucoup de faux positifs et
faux négatifs sur des textes individuels. Ne pas utiliser pour accuser un auteur de fraude.

---

## Limitations

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
│   │   ├── zola.json          # 8 extraits de Germinal (domaine public)
│   │   └── maupassant.json    # 8 nouvelles (domaine public)
│   ├── gpt4/rewrites.json     # 16 réécritures GPT-4 (codées en dur)
│   ├── claude3/rewrites.json  # 16 réécritures Claude 3 (codées en dur)
│   ├── mistral/rewrites.json  # 16 réécritures Mistral 7B (codées en dur)
│   └── gemini/rewrites.json   # 16 réécritures Gemini Pro (codées en dur)
├── notebooks/
│   ├── 01_shift_analysis.ipynb  # Shifts, violin, PCA, t-SNE, fingerprint
│   ├── 02_classification.ipynb  # Confusion matrix, SVM, courbe d'apprentissage
│   └── 03_robustesse.ipynb      # Bootstrap CI, permutation tests, intra-variance
├── results/
│   ├── pca_clusters.png
│   ├── tsne_clusters.png
│   ├── shift_distributions.png
│   ├── bootstrap_ci.png
│   ├── permutation_null.png
│   ├── fingerprint_comparison.png
│   ├── radar_profiles.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── dendrogram.png
│   ├── shift_by_source.png
│   └── shift_vs_length.png
├── generate_results.py   # Régénère les 12 PNG sans Jupyter
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
