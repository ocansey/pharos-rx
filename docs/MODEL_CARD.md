# Model Card — PHAROS

Following Mitchell et al. (2019), *Model Cards for Model Reporting*.

## System details

| | |
|---|---|
| **Name** | PHAROS — Pharmacovigilance Agent for Retrieval-Ordered Synthesis |
| **Version** | 1.0.0 |
| **Type** | Retrieval-augmented synthesis system (LangChain / LangGraph) |
| **License** | MIT |
| **Components** | deterministic cleaner · clause segmenter · weak aspect labeller · hybrid retriever with stratified sampling · deterministic statistics engine · pluggable generator · claim verifier · rules-first safety triage |
| **Generator** | provider-agnostic: Anthropic, OpenAI, Ollama, or a deterministic offline mock. **No model is trained or fine-tuned by this project.** |
| **Encoder** | truncated SVD over TF-IDF fitted on the corpus (default), or any sentence-transformers bi-encoder |

PHAROS is a **system**, not a model. Its behaviour is dominated by its retrieval
and verification architecture rather than by any generator, which is the design
intent — swapping the generator should change prose quality, not what is claimed.

## Intended use

**Primary.** Summarising, with citations, what a population of patient reviewers
publicly wrote about a medication for an indication — outcome distributions,
commonly reported effects, and the range of experiences, always presented as
patient report rather than clinical fact.

**Intended users.** Researchers studying retrieval bias or patient-reported
outcomes; engineers evaluating RAG faithfulness architectures; educators
teaching evidence grounding and evaluation design.

**Intended context.** Exploratory, aggregate, research and educational use, by
users who understand that the underlying data is self-selected public opinion.

## Out-of-scope use

These are not edge cases; several are refused by construction.

- **Individual clinical decisions** — starting, stopping, switching, or dosing any
  medication. Refused at triage before retrieval occurs.
- **Diagnosis or symptom interpretation.** Refused.
- **Drug interaction checking.** Refused; not a drug-information resource.
- **Regulatory pharmacovigilance.** The disproportionality statistics use
  standard estimators (PRR, ROR, IC) over the wrong kind of data. They are not
  signals and must not enter a safety assessment.
- **Prescribing, formulary, or coverage decisions.**
- **Comparative effectiveness claims.** Review ratings are not trial endpoints;
  cohorts are not randomised and differ systematically in indication severity.
- **Any use implying the output is medical advice.**

## Factors

**Groups.** Reviews carry no demographic attributes — no age, sex, race,
geography, or socioeconomic data. Subgroup performance therefore **cannot be
measured**, and this is a real limitation rather than an omission. Drugs.com's
2008–2017 user base is unlikely to be representative of any general patient
population; conditions with stigma or low internet engagement are likely
under-represented.

**Conditions of use.** Performance depends on cohort size. Cohorts under ~10
reviews are flagged low-support; under 3 mentions no disproportionality statistic
is emitted. Very large cohorts show the largest retrieval skew in the baseline
and the largest benefit from stratification.

## Metrics

Retrieval quality (nDCG@k, Recall@k, MRR, Precision@k) plus two metrics
introduced here:

- **VSD** — Jensen–Shannon divergence between the retrieved panel's outcome
  distribution and the cohort's true distribution. Jensen–Shannon rather than
  Kullback–Leibler because KL is infinite when a stratum is empty in one
  distribution — exactly the worst-skewed panels.
- **CRE** — absolute error between panel mean rating and cohort mean rating, on
  the 1–10 scale, i.e. how many stars a reader of the panel alone is misled by.

Grounding: claim support rate, citation validity, numeric grounding rate.
Safety: pass rate across seven red-team hazard categories, **including benign
look-alikes that detect over-refusal**.

All aggregates carry bootstrap CIs; comparisons use a paired bootstrap.

## Evaluation data

300 queries generated from (drug, condition, aspect) triples over the corpus,
with relevance derived from structured metadata. Cohorts below 30 evidence units
excluded. Sampling round-robin over aspects.

**Not** human-annotated and **not** LLM-judged. LLM-as-judge was rejected because
this project argues ungrounded model output should not be trusted; validating
that argument with ungrounded model output would be incoherent, and would make
the evaluation a measurement of the judge.

## Training data

None. No component is trained on labelled data. The TF-IDF/SVD encoder is
*fitted* (unsupervised) on the evidence-unit corpus; the aspect labeller and
safety triage are rule-based and hand-authored.

## Quantitative analysis

| metric | PHAROS | naive RAG baseline | Δ | *p* |
|---|---:|---:|---:|---:|
| VSD ↓ | 0.0082 | 0.0324 | −75% | <0.001 |
| CRE ↓ | 0.377 | 0.667 | −43% | <0.001 |
| Stratum coverage ↑ | 1.000 | 0.866 | +15% | <0.001 |
| nDCG@10 ↑ | 0.786 | 0.838 | −6% | <0.001 |
| Recall@10 ↑ | 0.232 | 0.324 | −28% | <0.001 |

Red-team suite: **42/42**, including 8/8 benign look-alikes.

The relevance cost is reported, not buried. See `docs/METHODS.md` §7 for the
argument that it is worth paying for a summarisation task, and the counter-case.

## Ethical considerations

See [ETHICS.md](ETHICS.md). In brief: reviewers did not consent to this use;
de-identification is defence-in-depth over already-public pseudonymous text and
is not a re-identification guarantee; the system refuses individualised medical
questions as a design position rather than a capability limit; and fluent
citation-bearing prose confers unearned authority, which is why numeric claims
are removed from the generative path entirely.

## Caveats and recommendations

- Reproduce the numbers before citing them: `pharos evaluate` regenerates every
  table from the shipped code.
- Read `docs/RESULTS.md` §4 before quoting any single figure.
- Do not deploy for clinical use. Do not present output as medical advice.
- If reusing Stratified Evidence Sampling elsewhere, verify that the reference
  distribution is estimated over the **full** population and not over retrieved
  candidates. Getting that wrong reproduces the bias the method removes.

## Contact

Issues and questions via the repository issue tracker.
