"""PHAROS — Pharmacovigilance Agent for Retrieval-Ordered Synthesis.

A lighthouse, not a prescription.

PHAROS is a retrieval-augmented synthesis system over a real corpus of 215,063
patient-authored drug reviews (Drugs.com, via the UCI ML Repository / Kaggle).
It is built around a single thesis: *patient-narrative corpora are epistemically
hazardous for naive RAG*, and the fix is architectural rather than prompt-level.

Three mechanisms carry that thesis:

1. **Stratified Evidence Sampling** (:mod:`pharos.retrieval.stratified`) treats
   retrieval as sampling from a population rather than as ranked lookup, so the
   evidence panel handed to the generator reflects the cohort's true outcome
   distribution instead of the corpus's self-selection bias.

2. **Computed cohort statistics** (:mod:`pharos.data.cohort`) move every numeric
   claim out of the language model and into deterministic, auditable arithmetic:
   Wilson intervals, proportional reporting ratios, and chi-square screening
   drawn from the standard pharmacovigilance toolkit.

3. **Claim-level verification** (:mod:`pharos.agent.nodes.verify`) decomposes the
   draft answer into atomic claims and refuses to emit any claim that is not
   entailed by a retrieved evidence unit or a computed statistic.

See ``docs/METHODS.md`` for the full technical report.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
