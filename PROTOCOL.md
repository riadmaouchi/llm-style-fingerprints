# Protocole expérimental — llm-style-fingerprints

## 1. Question de recherche

**Est-il possible de distinguer des modèles de langage (LLMs) à partir de leurs seuls patrons stylistiques de mots-outils, mesurés sur une tâche de réécriture littéraire ?**

Question dérivée : le fingerprint stylistique d'un modèle est-il stable selon le prompt utilisé (robustesse inter-prompt) ?

---

## 2. Design de l'étude

### 2.1 Type d'étude

Étude observationnelle comparative. Pas de groupe contrôle aléatoire. Mesure de différences stylistiques entre groupes définis a priori (4 LLMs + 2 auteurs humains).

### 2.2 Unité d'analyse

Le **vecteur de shift** d'un texte : `rewrite_vec − original_vec`, où chaque vecteur est une fréquence normalisée L2 de 57 mots-outils français.

---

## 3. Corpus

### 3.1 Textes sources humains

**Auteurs** : Émile Zola et Guy de Maupassant (domaine public français).

**Critère de sélection des œuvres** :
- Prose narrative (romans, nouvelles)
- Enregistrées au Projet Gutenberg ou Wikisource
- Langue française originale (pas de traductions)

**Critère de sélection des extraits** :
- Longueur : 80–250 mots par extrait
- Présence d'action ou de description (éviter les dialogues purs, qui faussent les fréquences de mots-outils)
- Représentation équilibrée des œuvres : aucun roman ne dépasse 25 % des extraits d'un auteur

**Œuvres de Zola utilisées** :
- Germinal (extraits 1–10)
- L'Assommoir (extraits 11–15)
- Nana (extraits 16–19)
- Au Bonheur des Dames (extraits 20–22)
- La Terre (extraits 23–25)
- La Débâcle (extraits 26–28)
- La Bête humaine (extraits 29–31)
- Le Ventre de Paris (extraits 32–34)
- La Curée (extraits 35–37)
- Pot-Bouille (extraits 38–40)

**Œuvres de Maupassant utilisées** :
- Boule de suif (extraits 26–29)
- La Parure (extraits 30–32)
- Bel-Ami (extraits 33–36)
- Le Horla (extraits 37–39)
- Pierre et Jean (extraits 40–42)
- Fort comme la mort (extraits 43–45)
- Une vie (extraits 46–48)
- Contes de la Bécasse (extraits 49–51)
- La Maison Tellier (extraits 52–54)
- Miss Harriet (extraits 55–58)
- Yvette (extraits 59–62)
- Notre cœur (extraits 63–65)

**Taille du corpus humain** : 40 textes Zola + 40 textes Maupassant = **80 textes sources**.

### 3.2 Contrainte de longueur minimale

`min_words = 20` (paramètre du StyleAnalyzer). Les extraits inférieurs à 20 mots sont rejetés automatiquement. En pratique, tous les extraits dépassent 80 mots.

---

## 4. Modèles LLM

| Label corpus | Modèle API | Version | Provider |
|---|---|---|---|
| GPT-4 | gpt-4o | gpt-4o | OpenAI |
| Claude 3 | claude-sonnet-4-6 | claude-sonnet-4-6 | Anthropic |
| Mistral 7B | mistral-small-latest | mistral-small-latest | Mistral AI |
| Gemini Pro | gemini-2.5-flash | gemini-2.5-flash | Google |

**Date de collecte des données** : mai 2026.

**Note** : les labels "GPT-4", "Claude 3", "Mistral 7B", "Gemini Pro" sont des noms d'affichage pour la lisibilité. Les versions exactes sont dans les fichiers `data/*/rewrites*.json`.

---

## 5. Prompts de réécriture

Deux prompts ont été utilisés pour tester la robustesse inter-prompt.

### Prompt 1 (P1) — réécriture neutre

```
Réécris ce texte dans un style neutre et factuel, en conservant le sens.
```

**Intention** : neutraliser le style littéraire de l'original, révéler le style de base du LLM.

### Prompt 2 (P2) — simplification

```
Reformule ce texte en simplifiant le vocabulaire, pour le rendre accessible au grand public.
```

**Intention** : tâche différente (simplification vs neutralisation) — teste si le fingerprint stylistique du modèle est prompt-dépendant ou intrinsèque.

### Paramètres communs

- Température : `0.7`
- Tokens maximum : `400`
- Système : `"Tu es un assistant qui réécrit des textes littéraires français."`

---

## 6. Collecte des données

### 6.1 Procédure

Pour chaque texte source `t` ∈ corpus (80 textes) et chaque prompt `p` ∈ {P1, P2} :
1. Appel API au modèle LLM avec le texte `t` et le prompt `p`
2. Stockage de la réécriture dans `data/{model}/rewrites_{p}.json`
3. Délai entre appels : 0.5s (respect des rate limits)
4. En cas d'erreur 503/429 : backoff exponentiel (max 5 tentatives)

### 6.2 Format de stockage

```json
{
  "model": "GPT-4",
  "model_version": "gpt-4o",
  "prompt": "Réécris ce texte dans un style neutre et factuel, en conservant le sens.",
  "prompt_id": "p1",
  "temperature": 0.7,
  "collection_date": "2026-05",
  "rewrites": [
    {"id": 1, "original_id": 1, "text": "..."}
  ]
}
```

### 6.3 Reproductibilité

Le script `scripts/generate_rewrites.py` reproduit l'ensemble de la collecte à partir des textes sources. Les clés API sont requises (non incluses dans le dépôt). Les fichiers `data/*/rewrites*.json` sont committés pour reproductibilité sans appels API.

---

## 7. Extraction des features

### 7.1 Features principales — mots-outils (57 dimensions)

**Bibliothèque** : `stylometry-python` (StyleAnalyzer)

**Liste de mots-outils** : 57 mots-outils français prédéfinis dans `FUNCTION_WORDS_FR`, incluant les pronoms, déterminants, prépositions, conjonctions, et adverbes de liaison courants.

**Normalisation** : L2 (vecteur unité). La fréquence absolue de chaque mot-outil est divisée par la norme du vecteur.

**Distance** : cosinus. Le shift d'un texte est `1 − cos(original, rewrite)`.

### 7.2 Features secondaires (exploratoires)

Testées mais n'améliorant pas la classification pour cette tâche :

| Feature | Méthode | Résultat |
|---|---|---|
| `hedge_density` | Fréquence de marqueurs d'incertitude / 100 tokens | Non discriminant (CV 0.06–0.27, plat entre modèles) |
| `burstiness` | std / mean longueur phrases | Non discriminant (0.36–0.40 pour tous) |
| `punct_entropy` | Entropie Shannon sur ponctuation | Signal faible pour Gemini uniquement (coeff LR = 1.02) |

Conclusion : pour une tâche de réécriture littéraire, les LLMs égalisent leurs patterns de surface. Seules les fréquences de mots-outils restent discriminantes.

---

## 8. Pipeline statistique

```
Textes sources (80)
    ↓  StyleAnalyzer.fit_transform()
Vecteurs fonction-mots (80 × 57)
    ↓  pour chaque LLM
Shift vectors = rewrite_vec − original_vec  (80 × 57)
    ↓
Shift scalaire = cosine_distance(original, rewrite)  ∈ [0, 1]
    ↓
bootstrap_ci(shifts, n_boot=5000)     → IC 95 % percentile
permutation_test(a, b, n_perm=10000)  → p-value (différence de moyennes)
pairwise_tests(correction="bonferroni") → C(4,2)=6 tests corrigés
intra_variance(corpus, sa)            → variance cosinus intra-groupe
```

### 8.1 Classification

- **Centroïde LOO** : `NearestCentroid`, Leave-One-Out — baseline naïf
- **Logistique LOO** : `LogisticRegression(C=1.0, solver="lbfgs")` + `StandardScaler`, Leave-One-Out — inclut les 3 features secondaires

### 8.2 Visualisation

- **LDA (vecteurs de shift)** : `LinearDiscriminantAnalysis(n_components=2)` — maximise la séparation inter-LLM pour la visualisation
- **t-SNE / UMAP** : sur les vecteurs bruts (6 groupes), visualisation de la structure locale

---

## 9. Limites et biais connus

| Limite | Impact |
|---|---|
| Un seul corpus source (Zola + Maupassant) | Résultats potentiellement spécifiques au style littéraire français du XIXe | 
| Vocabulaire de mots-outils conçu pour capturer les patterns LLM | Avantage structurel pour la détection des modèles formels |
| Pas de contrôle humain "en style neutre" | Impossible de comparer à un rédacteur humain sur la même tâche |
| Modèles figés (mai 2026) | Les signatures ne sont pas permanentes — les modèles évoluent |
| Variance intra-groupe élevée (0.32–0.50) | Signal insuffisant pour des conclusions sur des textes individuels |
| n=80 par classe | Statistiquement solide pour les tests pairés, mais jeu de données modeste pour la classification |

---

## 10. Logiciels et versions

Voir `pyproject.toml` pour les versions exactes. Dépendances principales :

- Python 3.11+
- `stylometry-python` (StyleAnalyzer, FUNCTION_WORDS_FR, stats)
- `scikit-learn` (LDA, PCA, NearestCentroid, LogisticRegression, SVC, t-SNE)
- `umap-learn` (UMAP)
- `scipy` (gaussian_kde, hierarchical clustering, cosine distance)
- `seaborn` / `matplotlib` (visualisation)
- `numpy`

---

## 11. Checklist de reproductibilité

- [ ] Clés API configurées dans `.env` (voir `.env.example`)
- [ ] `make install` — installe les dépendances dans `.venv`
- [ ] `python scripts/generate_rewrites.py` — régénère les réécritures
- [ ] `make test` — 21 tests verts
- [ ] `make results` — régénère toutes les figures
- [ ] Les fichiers `data/` committés permettent `make results` sans appels API
