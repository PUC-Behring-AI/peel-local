# peel-local

PEEL-Local is a computationally augmented interpretive inquiry framework for running phases derived from *Protocols for Epistemically Engaged Literacy in AI (PEEL)* locally with complementary AI and data analysis tools.

PEEL-Local, stemming from PEEL, aims at not automating interpretation.

Instead, it provides computational support for:
- semantic inspection
- lexical organization
- clustering
- structured interpretive analysis

Interpretation remains researcher-driven.


The current implementation focuses on **PEEL Phase 1**, supporting:
- lexical-semantic analysis
- Word Sense Disambiguation (WSD)
- semantic clustering
- structured interpretive workflows

The project is designed to support researcher-driven interpretation rather than interpretive automation.

---

## Technologies

### NLP
- [spaCy](https://spacy.io)
- [NLTK](https://www.nltk.org)
- [WordNet](https://wordnet.princeton.edu)

### Word Sense Disambiguation
- [GlossBERT](https://github.com/HSLCY/GlossBERT)

### Semantic Embeddings
- [Sentence-Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### Clustering
- [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/)

---

## Installation

Install dependencies:

```bash
pip install torch transformers sentence-transformers
pip install spacy nltk hdbscan numpy tqdm
```

Download spaCy model:

```bash
python -m spacy download en_core_web_sm
```

Download WordNet resources:

```python
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")
```

---

## Methodological Position

PEEL-Local does not automate interpretation.

Instead, it provides computational support for:
- semantic inspection
- lexical organization
- clustering
- structured interpretive analysis

Interpretation remains researcher-driven.
