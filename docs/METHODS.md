# PHAROS — Technical Report

**Cohort-Grounded Retrieval-Augmented Synthesis over Patient Drug Reviews**

---

## Abstract

Retrieval-augmented generation over opinion corpora suffers a failure mode that
is invisible to the metrics normally used to evaluate it. Similarity search
returns passages most similar to the query; in patient-narrative corpora,
similarity correlates with rhetorical emphasis, emphasis correlates with outcome
extremity, and the evidence panel presented to the generator is therefore
systematically unrepresentative of the population it purports to summarise. The
generator reproduces that skew faithfully, producing an answer in which every
citation is genuine, every sentence is grounded, and the aggregate claim is
false.

We name this **retrieval-induced valence skew**, give it two operational
measures, and address it architecturally through **Stratified Evidence
Sampling** (SES): retrieval reformulated as quota-constrained sampling from a
population whose outcome distribution is estimated over the full corpus rather
than over the retrieved candidates. On 300 metadata-derived queries against
81,402 evidence units drawn from 215,063 Drugs.com reviews, SES reduces
Jensen–Shannon divergence between panel and cohort outcome distributions by 75%
(0.0324 → 0.0082) and mean cohort rating error by 43% (0.667 → 0.377 stars)
relative to a conventional dense top-*k* retriever, at a cost of 0.052 nDCG@10.
All differences are significant under a paired bootstrap (*p* < 0.001).

Two supporting contributions accompany the main result. First, all numeric
content is removed from the generative path: quantities are computed by
deterministic pharmacovigilance estimators and passed to the model as
pre-formatted, individually addressable statistics. Second, a claim-level
verifier rejects *disguised quantity claims* — assertions such as "most
reviewers report X" that contain no numeral, cite genuine evidence, satisfy
entailment, and are nonetheless unsupported quantitative claims.

We additionally document a previously unreported systematic character-deletion
defect affecting 10.3% of the source corpus.

---

## 1. Problem statement

### 1.1 Formalisation

Let $C$ be a corpus of reviews, each carrying a drug $d$, an indication $c$, and
an ordinal outcome rating $r \in \{1,\dots,10\}$. For a query $q$ concerning the
cohort $K(d,c) = \{x \in C : x.\text{drug}=d,\ x.\text{cond}=c\}$, let $P_K$
denote the cohort's outcome distribution over strata
$S = \{\text{neg},\text{mix},\text{pos}\}$.

A retriever returns a panel $\Pi_q$ of $k$ units. Let $P_{\Pi}$ be the panel's
outcome distribution. Conventional retrieval optimises a relevance objective
$\sum_{u\in\Pi} \mathrm{rel}(q,u)$ and places **no constraint whatsoever** on
$P_\Pi$.

Define **Valence Skew Divergence**:

$$\mathrm{VSD}(\Pi, K) = \mathrm{JSD}(P_{\Pi} \parallel P_K) \in [0,1]$$

and **Cohort Rating Error**:

$$\mathrm{CRE}(\Pi, K) = \left| \bar{r}_{\Pi} - \bar{r}_{K} \right|$$

The claim under test is that conventional retrieval yields $\mathrm{VSD} \gg 0$
in a way that scales with $|K|$, and that this is correctable without
unacceptable loss of relevance.

**Why Jensen–Shannon and not Kullback–Leibler.** KL diverges to infinity when a
stratum has zero mass in $P_\Pi$ but positive mass in $P_K$ — precisely the
badly-skewed panels the metric exists to detect. A metric that returns infinity
exactly where it matters most is unusable. JSD is symmetric, bounded in $[0,1]$
under base-2 logarithms, and finite in that case.

**Why CRE as well.** VSD is the principled measure but is not interpretable to a
reader. CRE answers "by how many stars would someone reading only this panel be
misled?" on the same 1–10 scale the data uses.

### 1.2 Scope of the claim

SES does **not** debias the corpus. Self-selection in patient reviews is real and
no sampling scheme over $C$ removes it; a drug whose reviewers are
disproportionately unhappy will still appear that way, correctly. What SES
removes is the *additional* distortion retrieval imposes on top — the gap between
what the corpus says and what the retriever showed the model. That gap is
entirely an artifact of the system, and therefore the system's to close.

---

## 2. Corpus

### 2.1 Provenance

Drugs.com patient reviews collected by Gräßer et al. (2018), deposited at the UCI
Machine Learning Repository (dataset 462) and mirrored on Kaggle.

| property | value |
|---|---:|
| Reviews | 215,063 |
| Distinct drugs | 3,671 |
| Distinct condition labels (raw) | 917 |
| Date range | 2008-02-24 – 2017-12-12 |
| Mean review length | 458 characters |

Acquisition verifies SHA-256 digests and exact row counts (161,297 train /
53,766 test) before any downstream stage trusts the files. A silently truncated
download would otherwise surface as an inexplicable metric shift three stages
later.

### 2.2 Defect inventory

Five defects were identified. Each is handled explicitly and *counted*, because
a cleaning step whose effect cannot be quantified cannot be defended.

| defect | rows | handling |
|---|---:|---|
| HTML entity escaping | 140,533 | decoded to fixed point |
| Export quote wrapping | 215,063 | unwrapped |
| Condition label — trailing deletion | 16,499 | repaired from documented table |
| Condition label — leading deletion | 2,927 | repaired from documented table |
| Condition field contains review footer | 1,171 | set null, row retained |
| Condition field contains truncated drug name | 329 | set null, not guessed |
| Missing condition | 1,194 | dropped |
| Exact duplicates on (drug, text) | 109 | first kept |
| Near duplicates (MinHash ≥ 0.92) | 42 | first kept |
| Volunteered identifiers | 57 | typed placeholders |

### 2.3 The character-deletion defect

The `condition` column carries two independent systematic deletions introduced
upstream of publication. Both are established by evidence internal to the corpus.

**D1 — trailing deletion.** A word-final `[or]+` (case-insensitive) was deleted
corpus-wide.

*Evidence.* Of 917 distinct labels, **exactly zero end in the bigram "er"**. For a
medical vocabulary demonstrably containing Cancer, Ulcer, Fever, Bladder,
Disorder, Shoulder and Zoster, the probability of this under any plausible null
is negligible. The deleted class is `[or]+` rather than `r` alone: `Tinea
Versicolor` survives as `Tinea Versicol`, having lost two characters.

*Scale.* 16,499 rows (7.7%). The largest single group is `Bipolar Disorde`
(5,604 rows).

**D2 — leading deletion.** For a subset of rows, the span matching
`^.*?f[or]*` (case-insensitive) was deleted.

*Evidence.* The pattern is confirmed by two labels no analyst would reconstruct
by inspection:

| corrupted | rows | true label | deleted span |
|---|---:|---|---|
| `me` | 2 | Glioblastoma Multi**for**me | `Glioblastoma Multifor` |
| `mis` | 5 | Dermatitis Herpeti**for**mis | `Dermatitis Herpetifor` |

Both are cut at the first internal `f` and both survive as semantically empty
stubs. Twenty-four further labels fit the same pattern (`ibromyalgia` ←
Fibromyalgia, `mance Anxiety` ← Performance Anxiety, `t Care` ← Foot Care).
Unlike D1, D2 affected only a subset of rows: `Performance Anxiety` appears both
intact and as `mance Anxiety`.

**Consequence.** This is not cosmetic. PHAROS retrieves under metadata
constraints; a filter for `condition == "Bipolar Disorder"` matches zero rows,
because all 5,604 are filed under the corrupted label. Any cohort statistic
computed over raw labels is computed over a silently mislabelled population.

**Repair policy.** Repairs are enumerated in `configs/condition_repairs.yaml`,
one justified entry per corrupted label, applied artifact → leading → trailing.
Three constraints govern the table: (i) repairs are applied only where the
inverse is unique and yields an attested medical term; (ii) legitimate
lowercase-initial labels are whitelisted (`von Willebrand's Disease` would
otherwise be "repaired"); (iii) 329 rows in which a truncated *drug* name leaked
into the condition column are set to null rather than reconstructed, since the
true condition is not recoverable from the row.

### 2.4 Evidence units

Reviews are segmented into clause-level evidence units rather than chunked by
token count. The motivation is specific to this corpus: a review is not prose
with a topic but a sequence of distinct claims compressed into a few sentences.
*"Worked great for my anxiety within a week, but the weight gain was awful"*
indexed whole produces a single vector retrieved identically for "does it work?"
and "does it cause weight gain?", and the generator receives both answers mixed.

Segmentation splits on sentence boundaries **and** on discourse connectives
marking polarity or topic shift (`but`, `however`, `although`, `unfortunately`),
because in this register those connectives are the strongest available signal
that an adverse-effect claim is beginning. Fragments below 60 characters are
merged into a neighbour; units above 420 characters are split at whitespace.

Result: 24,994 reviews → **81,402 evidence units** (3.26 per review, mean 130
characters).

Each unit is labelled with aspects (efficacy, adverse effect, onset/duration,
dosing, access/cost, discontinuation, comparison, context) by a deterministic
weak labeller: weighted surface cues plus a lay clinical lexicon, with
adverse-effect mentions suppressed under negation scope. The lexicon is
deliberately *lay* — patients write "knocked me out", not "somnolence"; a MedDRA
or SNOMED vocabulary matches almost nothing here.

The labeller is deterministic rather than neural for three reasons: it is
auditable (every label traces to a readable cue, which matters because the labels
shape downstream statistics); it is free (labelling 81k units with an LLM
introduces a cost and a model-version dependency, defeating reproducibility); and
its error rate is bounded and reportable rather than assumed away.

### 2.5 Sampling

The default build uses a deterministic condition-stratified subsample of 25,000
reviews, allocating slots proportional to $\sqrt{n_c}$ per condition. Plain
random sampling would let the 38,436-review Birth Control cohort dominate and
starve the long tail; square-root allocation keeps the head representative while
giving small cohorts enough rows to support an interval. `sample_size: null`
builds the full corpus.

---

## 3. Retrieval

### 3.1 Pipeline

```
parse ──► metadata filter ──► dense ──┐
                              lexical ─┴─► RRF ──► MMR ──► SES ──► panel
```

**Parsing** is deterministic vocabulary matching against the corpus's own drug
and condition names, longest-first, on word boundaries. An LLM extractor would be
more flexible and would also make retrieval metrics depend on a model version —
the numbers would then move when the model moved, for reasons unrelated to
retrieval.

**Metadata filtering** produces a candidate array *before* scoring. This is a
correctness property, not an optimisation. Vector stores typically apply metadata
filters *after* approximate nearest-neighbour search, so a filter for a drug with
40 reviews in an 81,402-unit index frequently returns nothing — none of the
approximate neighbours happened to be that drug. The system then answers from an
empty panel or silently widens, and the user cannot tell which. Filtering first
means small cohorts are retrieved exhaustively and exactly.

**Fusion** uses reciprocal rank fusion rather than score interpolation. BM25
scores are unbounded sums over query terms; cosine similarities lie in $[-1,1]$.
Any weighted sum is really a weighted sum of one arm and the *noise* of the
other, and the weight that works for a three-word query is wrong for a twenty-word
one. RRF discards magnitudes and is invariant to both scales.

**MMR** ($\lambda = 0.65$) controls redundancy. This corpus is unusually
repetitive: hundreds of reviews of the same contraceptive contain near-identical
sentences about the same effect. Plain top-*k* returns twelve paraphrases of one
claim, and the generator, seeing twelve, reports overwhelming consensus. Here MMR
trades relevance for *accuracy*, not merely variety.

### 3.2 Stratified Evidence Sampling

Given panel size $k$ and reference distribution $P_K$:

1. **Estimate $P_K$ over the full cohort**, not over retrieved candidates. This
   is the load-bearing step: a distribution estimated from what similarity search
   returned already carries the bias SES exists to remove.
2. **Apportion $k$ slots** to strata proportional to $P_K$ using
   largest-remainder (Hamilton) apportionment.
3. **Guarantee** $\ge$ 1 slot to every stratum with non-zero cohort mass.
4. **Fill** each quota from that stratum's own highest-ranked candidates, capping
   contributions per parent review at 2.
5. **Reallocate** unfillable quota to strata that can absorb it, so the panel is
   always exactly $k$ when $k$ units exist.

**On apportionment.** Independent rounding of $k \cdot P_K(s)$ does not sum to
$k$. With three strata at 0.34/0.33/0.33 and $k=12$ it yields 4/4/4, 5/4/4 or
4/4/3 depending on floating-point noise — making panel composition
irreproducible across machines. Hamilton's method allocates integer floors then
distributes remaining seats by largest fractional remainder, and sums exactly by
construction.

**On the minimum-slot guarantee.** Without it, a stratum holding 2% of cohort
mass receives $\lfloor 0.24 \rfloor = 0$ slots at $k=12$ and disappears. That
2% is the dissenting minority — the reviewers for whom the drug failed — and
they are the single most decision-relevant group in the panel.

**On the per-review cap.** Reviews vary from 3 to 10,787 characters. Uncapped, a
verbose reviewer can contribute five of twelve panel slots, and the generator
reads one person's experience as a pattern.

---

## 4. Computed statistics

No quantity in a PHAROS answer originates in the language model. Estimators are
chosen for regulatory recognisability.

| estimator | why this one |
|---|---|
| **Wilson score interval** | Wald yields inadmissible intervals on this corpus. For 3/31 reports, Wald gives (−0.6%, 20.1%) — a negative probability. Wilson gives (3.3%, 24.9%). |
| **Percentile bootstrap** for mean rating | Ratings are ordinal and sharply bimodal (68.1% at 1–2 or 9–10). A normal-theory interval on that shape assumes a distribution the data plainly lacks. |
| **PRR** with log-scale Gaussian interval | MHRA screening statistic. |
| **ROR** | EMA / Netherlands Pharmacovigilance Centre variant; remains valid under case-control sampling. |
| **χ² with continuity correction** | Independence test accompanying PRR under Evans' criteria. |
| **BCPNN Information Component** | WHO-UMC Bayesian shrinkage; does not over-react to a 2-of-2 cell as PRR does. |
| **Kruskal–Wallis** for head-to-head | Rank-based; a one-way ANOVA would assume a shape these distributions do not have. |

**Comparator selection.** Disproportionality defaults to *other drugs for the
same indication*, not the whole corpus. This controls confounding by indication:
comparing sedation reports for a sleep aid against the entire corpus flags every
sleep aid ever written about, because the indication rather than the drug is
doing the work.

**Continuity correction.** Haldane–Anscombe (+0.5 to every cell), so a zero
background cell yields a wide but finite interval rather than an infinity that
propagates into the answer.

**Reporting floor.** Below $a = 3$ exposed reports, no disproportionality
statistic is emitted at all. Below that threshold every estimator is noise, and
emitting one with a wide interval invites a reader to over-read it.

**Unit of analysis.** The *review*, never the evidence unit. A review mentioning
nausea in three units contributes one nausea report. Failure to collapse here is
the most common way analyses of this corpus inflate their own denominators.

**Standing caveat.** Every disproportionality block carries, in the data
structure itself, the statement that it is computed over self-selected patient
reviews rather than spontaneous adverse-event reports and is not a regulatory
signal. A reader cannot see the PRR without seeing that.

---

## 5. Generation and verification

### 5.1 Graph

`triage → retrieve → compute → synthesize → verify → (repair) → finalize`, with
refusal and abstention as terminal branches. Refusal precedes retrieval;
statistics precede generation; verification follows generation and can return
control. All loops are bounded.

### 5.2 Verification

Two layers. The deterministic layer runs first and cannot be overruled by the
model layer.

**Structural checks** (decidable, no model required):

| check | rejects |
|---|---|
| citation presence | uncited assertions |
| citation validity | identifiers absent from retrieved evidence |
| **disguised quantity** | *most, many, commonly, rarely, the majority* without a `[STAT-]` citation |
| numeric grounding | any numeral not traceable to a computed statistic |
| advice detection | recommendations, dosing instructions, directives |

The **disguised quantity** check is the novel element. *"Most reviewers report
insomnia"* contains no numeral, cites a genuine evidence unit, and satisfies any
entailment test, because some reviewers did report insomnia. It is nonetheless a
quantitative claim, and a reader receives "most" as a measurement. Such terms are
admitted only when backed by a computed statistic.

**Quotation exemption.** Quoted spans are stripped before quantity checks. A
faithful direct quotation — *One reviewer writes "I had several bad nights"* —
must not be penalised for the reviewer's word "several", which the system did not
assert. Without the exemption the verifier punishes exactly the behaviour it
should reward.

**Entailment layer** (model-based, when a live provider is configured) checks
whether cited material actually entails the claim rather than merely relating to
it. A verifier outage marks claims as structurally-checked-only and says so;
it never silently promotes unchecked claims to verified.

**Repair** is subtractive and bounded at one round: failed claims are dropped,
not rescued. A dropped claim costs the reader a sentence; a hedged ungrounded
claim costs them the ability to trust the rest.

### 5.3 Safety triage

Rules-first, because a refusal contingent on a model call fails **open** when the
API is unreachable, the key is absent, or the response is malformed. On a system
fielding medication questions, failing open is not an acceptable default. The
model layer runs second and may escalate severity but never reduce it.

The system refuses individualised clinical requests as a **design position**, not
a capability limit. It could produce a fluent, well-cited answer to *"should I
stop my sertraline"*; that is the hazard, since the reviews describe other people
and fluency reads as authority.

---

## 6. Evaluation protocol

### 6.1 Gold-set construction

Relevance judgements are **derived from structured metadata**, not from human
annotation (which does not scale to the 300 queries needed to separate
configurations differing by 0.02 nDCG) and not from LLM-as-judge.

LLM-as-judge was rejected on principle. This project argues that ungrounded model
output should not be trusted; validating that argument with ungrounded model
output would be incoherent. It would also make the leaderboard a measurement of
the judge, moving whenever the judge's version moved.

Queries are generated *from* (drug, condition, aspect) triples using multiple
templates per aspect. Graded relevance:

| grade | condition |
|---:|---|
| 3 | drug + indication + aspect all match |
| 2 | drug + indication match |
| 1 | drug matches |
| 0 | otherwise |

Cohorts below 30 units are excluded. A cohort smaller than the panel is retrieved
*exhaustively*, so its panel is perfectly representative regardless of retriever
behaviour — every fidelity metric reads 0 for reasons unrelated to the mechanism
under test. Sampling is round-robin over aspects so no single facet dominates.

**Stated limitation:** this measures whether retrieval finds units matching a
query's structured intent. That is a *necessary* condition for good retrieval,
not a sufficient one, and it is not a measure of answer quality.

### 6.2 Metrics and inference

nDCG@k uses linear gain $rel$ rather than $2^{rel}-1$. Exponential gain makes a
grade-3 unit worth seven grade-1 units, letting a configuration that finds one
perfect unit outscore one that assembles a well-rounded panel — a weighting
biased against the hypothesis under test.

Aggregates carry percentile bootstrap CIs (1,000 resamples). Configuration
comparisons use a **paired** bootstrap on per-query differences: all
configurations run on identical queries, and query-to-query variance dwarfs the
effects measured. An unpaired interval would be wide enough to swallow every
result in the table.

### 6.3 Reproducibility

The retrieval and fidelity suites require no language model, so the headline
table regenerates in CI. Every artifact records the config fingerprint that
produced it, and `pharos info` warns when a built index no longer matches the
active configuration.

The default encoder is truncated SVD over TF-IDF fitted on the corpus — a choice
worth defending rather than apologising for. It downloads nothing, is
bit-deterministic under a seed, and carries the domain's own vocabulary structure
rather than a general-web prior. On short, lexically dense, highly repetitive
clinical narrative, latent semantic indexing is a competitive baseline. A neural
bi-encoder is available behind an identical interface via one config change.

---

## 7. Results

Full tables: [RESULTS.md](RESULTS.md). Summary against `naive-baseline` (dense
top-*k*, no fusion, no MMR, no stratification), 300 queries, paired bootstrap:

| metric | PHAROS | baseline | Δ | 95% CI | *p* |
|---|---:|---:|---:|---|---:|
| VSD ↓ | 0.0082 | 0.0324 | −0.0242 | [−0.0281, −0.0208] | <0.001 |
| CRE ↓ | 0.377 | 0.667 | −0.290 | [−0.365, −0.221] | <0.001 |
| Coverage ↑ | 1.000 | 0.866 | +0.134 | [+0.111, +0.157] | <0.001 |
| nDCG@10 ↑ | 0.786 | 0.838 | −0.052 | [−0.059, −0.046] | <0.001 |
| Recall@10 ↑ | 0.232 | 0.324 | −0.092 | [−0.107, −0.077] | <0.001 |

**The critical control.** `uniform-strata` — equal quotas rather than
cohort-matched quotas — records CRE 1.301 against 0.662 for no stratification at
all: **twice as bad as doing nothing** (Δ = +0.924 vs full, [+0.834, +1.020],
*p* < 0.001). It nonetheless scores an identical 1.000 stratum coverage and is
statistically indistinguishable from the full system on every relevance metric.
Two conclusions follow. The benefit derives specifically from matching the
cohort's empirical distribution, not from rebalancing per se; and coverage alone
is an insufficient measure of panel representativeness, since a panel can contain
every outcome group in badly wrong proportions. Any account of these results
omitting this row is incomplete.

**Scaling.** VSD reduction is 78% / 75% / 70% / 81% across cohort-size bands
(1–12 / 13–30 / 31–80 / 81+ reviews; *n* = 75 / 184 / 39 / 2). Baseline stratum
coverage degrades **monotonically** with cohort size (0.931 → 0.853 → 0.812 →
0.667) while PHAROS holds at 1.000 throughout: on the largest cohorts a
conventional retriever omits a third of outcome groups from the panel entirely.
Larger cohorts give similarity search more freedom to select unrepresentatively.
The 81+ band contains 2 queries and is reported for completeness only.

**The trade.** Relevance loss is real and reported. The argument for accepting it
is that the two failure modes are not symmetric: a marginally less on-topic
answer is visible to the reader; a confidently unrepresentative one is not. This
is an argument, not a proof, and both columns are given so readers may disagree.

---

## 8. Threats to validity

**Construct.** Metadata-derived relevance is a proxy for usefulness. Aspect-
conditioned grades inherit the weak labeller's error rate. Neither measures
answer quality as a clinician would assess it.

**Internal.** The condition-repair table is curated. Repairs are constrained to
unique inverses yielding attested terms, and unrecoverable cases are nulled
rather than guessed, but the table is a judgement and is exposed for inspection
rather than hidden in code.

**External.** Results are reported for one corpus, one domain, and the default
LSA encoder. The mechanism should transfer to any corpus with a retrievable
outcome variable (product reviews, employer reviews, course evaluations), but
that is untested here. Neural-encoder results were not obtained in the
development environment for network reasons; the code path is implemented and
config-switchable.

**Statistical conclusion.** Bootstrap CIs assume exchangeability across queries.
Queries drawn from the same cohort are not fully independent; the paired design
mitigates but does not eliminate this.

**Provenance.** Reviews are self-selected, pseudonymous, and 2008–2017. No
statistic computed over them is a pharmacovigilance signal, and none should
inform clinical practice.

---

## 9. References

Bate, A., et al. (1998). A Bayesian neural network method for adverse drug
reaction signal generation. *European Journal of Clinical Pharmacology*, 54(4).

Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking
for reordering documents. *SIGIR '98*.

Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion
outperforms Condorcet and individual rank learning methods. *SIGIR '09*.

Evans, S. J. W., Waller, P. C., & Davis, S. (2001). Use of proportional reporting
ratios for signal generation from spontaneous adverse drug reaction reports.
*Pharmacoepidemiology and Drug Safety*, 10(6).

Gräßer, F., Kallumadi, S., Malberg, H., & Zaunseder, S. (2018). Aspect-based
sentiment analysis of drug reviews. *Digital Health '18*.

Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR
techniques. *ACM TOIS*, 20(4).

Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE
Transactions on Information Theory*, 37(1).

Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework:
BM25 and beyond. *Foundations and Trends in IR*, 3(4).

Rothman, K. J., Lanes, S., & Sacks, S. T. (2004). The reporting odds ratio and
its advantages over the proportional reporting ratio. *Pharmacoepidemiology and
Drug Safety*, 13(8).

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical
inference. *JASA*, 22(158).
