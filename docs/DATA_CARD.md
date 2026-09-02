# Data Card — Drugs.com Patient Review Corpus

Following Gebru et al. (2021), *Datasheets for Datasets*. This card describes the
source corpus **and** the transformations PHAROS applies to it, because several of
those transformations materially change the data.

## Motivation

**Why the dataset was created.** Collected by Gräßer et al. (2018) to study
cross-domain and cross-data sentiment transfer on patient drug reviews.

**Why PHAROS uses it.** It is one of very few large, public, freely
redistributable corpora of patient-authored medication narratives with a
structured outcome variable attached to every document. That outcome variable —
the 1–10 rating — is what makes the central experiment possible: without it there
is no ground-truth distribution to measure a retrieved panel against.

## Composition

| property | value |
|---|---:|
| Instances | 215,063 reviews |
| Train / test as published | 161,297 / 53,766 |
| Distinct drugs | 3,671 |
| Distinct condition labels (raw) | 917 |
| Date range | 2008-02-24 – 2017-12-12 |
| Mean / median review length | 458 / 456 characters |
| Longest review | 10,787 characters |

**Fields:** `uniqueID`, `drugName`, `condition`, `review`, `rating` (1–10),
`date`, `usefulCount`.

**Rating distribution — sharply bimodal.** 68.1% of reviews sit at the extremes
(1–2 or 9–10); 31.6% are 10/10 and 13.4% are 1/10. Corpus mean 7.17, median 9.0.
This is the shape that makes naive top-*k* retrieval hazardous and normal-theory
confidence intervals inappropriate.

**Highly imbalanced by indication.** Birth Control alone accounts for 38,436
reviews (17.9%); the median condition has fewer than 20.

**No demographic attributes.** No age, sex, race, geography, comorbidity, or
socioeconomic data. Subgroup analysis is impossible, and any claim about
representativeness is unfalsifiable from the data.

## Collection

Scraped from Drugs.com, a consumer drug-information site. Reviews are voluntary
and pseudonymous; reviewers are self-selected. **Selection bias is severe and
structural**: people write reviews when a medication is remarkable, so the
population is enriched for both dramatic success and dramatic failure relative to
the population actually taking these drugs. No sampling method applied downstream
corrects this.

Reviewers consented to public posting on Drugs.com. They did not consent to
inclusion in a research corpus or to use in a retrieval system. See
[ETHICS.md](ETHICS.md).

## Preprocessing applied by PHAROS

Every transformation is counted, because a cleaning step whose effect cannot be
quantified cannot be defended. Counts over the full 215,063-row corpus.

| transformation | rows | note |
|---|---:|---|
| HTML entities decoded | 140,533 | 62% of reviews contain `&#039;` and similar; left in place, tokenizers read `039` as a token |
| Export quote wrapping removed | 215,063 | all reviews are double-quote wrapped by the source export |
| Condition label — trailing repair | 16,499 | see below |
| Condition label — leading repair | 2,927 | see below |
| Condition set null (review footer) | 1,171 | `"3</span> users found this comment helpful."` captured in the condition field |
| Condition set null (leaked drug name) | 329 | truncated drug names; true condition unrecoverable, **not guessed** |
| Rows dropped (no condition) | 2,573 | |
| Rows dropped (below length floor) | 7,741 | |
| Rows dropped (condition below support floor) | 3,222 | conditions with <25 reviews |
| Exact duplicates removed | 109 | keyed on (drug, text) — same text under two drugs is two genuine reports |
| Near duplicates removed | 42 | MinHash Jaccard ≥ 0.92, blocked by drug |
| Rows de-identified | 57 | emails, phone numbers, URLs, long digit runs → typed placeholders |

**Result:** 215,063 → 201,376 retained (93.6%). Default build subsamples 25,000
reviews (condition-stratified, $\sqrt{n}$ allocation, seeded) → **81,402
evidence units**. `sample_size: null` uses the full corpus.

## The condition-label defect

**10.3% of the corpus (22,089 rows, 175 distinct labels) has a corrupted
condition label.** Two independent systematic deletions, introduced upstream of
publication, present in every public copy examined.

**D1 — trailing.** Word-final `[or]+` deleted corpus-wide. *Evidence:* of 917
distinct labels, **zero end in "er"** — impossible for a medical vocabulary
containing Cancer, Ulcer, Fever, Bladder, Disorder, Shoulder, Zoster.
`Tinea Versicolor → Tinea Versicol` shows the deleted class is `[or]+`, not `r`.

**D2 — leading.** Span matching `^.*?f[or]*` deleted from a subset of rows.
*Evidence:* `Glioblastoma Multiforme` survives as `me` (2 rows) and
`Dermatitis Herpetiformis` as `mis` (5 rows) — both cut at the first internal
`f`, both otherwise inexplicable.

**Largest affected groups:**

| corrupted label | rows | repaired to |
|---|---:|---|
| `Bipolar Disorde` | 5,604 | Bipolar Disorder |
| `ibromyalgia` | 2,370 | Fibromyalgia |
| `Major Depressive Disorde` | 2,131 | Major Depressive Disorder |
| `Panic Disorde` | 1,932 | Panic Disorder |
| `Generalized Anxiety Disorde` | 1,542 | Generalized Anxiety Disorder |
| `Overactive Bladde` | 917 | Overactive Bladder |

**Repair policy** (`configs/condition_repairs.yaml`, one justified entry per
label):

1. Repair only where the inverse is unique and yields an attested medical term.
2. Whitelist legitimate lowercase-initial labels — `von Willebrand's Disease`
   would otherwise be "repaired" by the leading-deletion heuristic.
3. Set null rather than reconstruct where the true value is unrecoverable.

Reproduce with `pharos audit-labels`.

**Practical consequence.** PHAROS retrieves under metadata constraints. Unrepaired,
a filter for `condition == "Bipolar Disorder"` matches **zero rows**. Any published
analysis grouping by raw condition label is grouping a silently mislabelled
population.

## Uses

**Suitable:** retrieval and RAG research; studies of patient-reported experience
in aggregate; NLP on lay clinical language; evaluation-methodology research.

**Unsuitable:** anything clinical or regulatory; drug safety signal detection
(these are not spontaneous adverse-event reports); comparative effectiveness;
prevalence or incidence estimation; any use assuming the reviewer population
resembles the patient population.

## Distribution and licence

Distributed via the [UCI ML Repository](https://archive.ics.uci.edu/dataset/462/)
(CC BY 4.0) and [Kaggle](https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018).

**PHAROS does not redistribute the data.** `pharos fetch-data` downloads it from
the original sources and verifies SHA-256 digests and exact row counts before any
stage trusts it. If a digest differs but row counts match, the run proceeds with a
warning that a different mirror was used.

## Maintenance

Static; last updated 2018 by its authors. PHAROS pins the expected digests and
row counts and fails loudly if a source changes, rather than silently building on
different data.

## Citation

Gräßer, F., Kallumadi, S., Malberg, H., & Zaunseder, S. (2018). Aspect-Based
Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning.
*Proceedings of the 2018 International Conference on Digital Health*, 121–125.
