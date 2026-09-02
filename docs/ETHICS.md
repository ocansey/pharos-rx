# Ethics Statement

This document records the ethical judgements built into PHAROS, including the
ones that cost the system capability. They are design positions, and they are
stated so a reader can disagree with them specifically rather than in general.

---

## 1. Consent

The 215,063 people whose words are in this corpus posted them publicly on
Drugs.com. They consented to publication on that site. **They did not consent to
inclusion in a research corpus, and they did not consent to being retrieved,
quoted, and aggregated by a language system.**

That gap is real and is not closed by the data being public, by the corpus being
widely used, or by the reviews being pseudonymous. It constrains what this
project does:

- Reviews are used **in aggregate**, to characterise populations, never to profile
  an individual reviewer.
- Quotations are short, clause-level, and always attached to a computed cohort
  context, so no single narrative is presented as representative.
- No attempt is made to link reviews to identities, to link reviews by the same
  author, or to enrich them with outside data.
- The corpus is **not redistributed**. `pharos fetch-data` retrieves it from the
  original publishers under their terms.

## 2. De-identification, and its limits

Mechanically detectable identifiers — email addresses, phone numbers, URLs,
long digit runs, US SSN patterns — are replaced with typed placeholders before
anything is indexed (57 rows in this corpus).

**This is defence in depth, not a re-identification guarantee, and it should not
be read as one.** Free-text clinical narrative resists de-identification: a rare
condition plus a distinctive drug plus an age plus a life event can identify a
person to someone who already knows them, and no regular expression detects that
combination. The residual risk is inherent to the source data and is not
eliminated by anything done here.

The honest framing: this scrubbing reduces the chance PHAROS *amplifies*
identifiers that are already public. It does not make the underlying corpus safe.

## 3. Refusal as a design position

PHAROS refuses an entire class of question **it is technically capable of
answering**.

Asked *"should I stop taking my sertraline?"*, the system could retrieve hundreds
of relevant reviews and assemble a fluent, well-cited, highly persuasive answer.
That capability is precisely the hazard:

- The reviews describe **other people**. The asker's history, comorbidities,
  concurrent medications, and reasons for the prescription are not in the corpus.
- **Fluency reads as authority.** A well-formatted answer with citations and
  confidence intervals carries more perceived credibility than an equivalent
  claim from a stranger, and the credibility is unearned.
- Abrupt discontinuation of several drug classes represented here — SSRIs,
  benzodiazepines, beta blockers, corticosteroids — causes real harm.

So triage refuses individualised requests before retrieval runs, and the refusal
explains *why* rather than simply declining, and points toward the aggregate
question the system can actually answer.

Two implementation choices follow from taking this seriously:

**Rules first.** A refusal contingent on a model call fails **open** when the API
is unreachable, the key is missing, or the response is malformed. On a system
fielding medication questions, failing open is not acceptable. The deterministic
layer decides first; the model layer may escalate severity but never reduce it.

**Over-refusal is measured too.** Eight of 42 red-team probes are *benign
look-alikes* — questions that mention a doctor, a prescription, or a dose but are
legitimate aggregate questions. A safety layer evaluated only on refusal rate
scores perfectly by refusing everything, and a system that refuses everything
helps nobody while appearing safe.

## 4. Crisis handling

Explicit indications of suicidal ideation, self-harm, overdose, or acute
emergency are routed to a dedicated response with real crisis resources, before
any other classification. This branch cannot be overridden downstream.

The response does not attempt counselling, does not assess risk, and does not ask
diagnostic questions. It says plainly that the system cannot help with this and
names resources. Where crisis language appears alongside an otherwise ordinary
question, crisis routing takes precedence.

## 5. Statistics that look more official than they are

PHAROS computes PRR, ROR, χ², and the WHO-UMC information component — the
standard regulatory pharmacovigilance toolkit — over **the wrong kind of data**.
These estimators were designed for spontaneous adverse-event reports submitted
through regulated channels. Patient reviews are self-selected public opinion.

Presenting a familiar estimator over unfamiliar data invites a reader to import
the estimator's usual authority. Three mitigations:

1. Every disproportionality block carries, **in the data structure itself**, the
   statement that it is computed over self-selected patient reviews and is not a
   regulatory signal. A reader cannot see the PRR without seeing that.
2. Below three exposed reports, no statistic is emitted at all — rather than
   emitting one with a very wide interval that invites over-reading.
3. The comparator defaults to other drugs for the same indication, so the most
   common spurious-signal mechanism (confounding by indication) is controlled
   rather than left to the reader to notice.

## 6. Selection bias, stated rather than corrected

The corpus over-represents dramatic outcomes in both directions: 68.1% of reviews
sit at 1–2 or 9–10. People write reviews when a medication is remarkable.

**Stratified Evidence Sampling does not fix this and does not claim to.** It
corrects the *additional* skew that retrieval imposes on top of the corpus's own —
the gap between what the corpus says and what the retriever showed the model.
Conflating the two would be a serious overclaim, and the distinction is stated in
the module docstring, the technical report, and the model card.

## 7. Ecological validity

Reviews span 2008–2017. Formulations change, generics enter, labels are updated,
and drugs are withdrawn. Nothing here reflects current prescribing practice or
current safety information. The system reports what people wrote, in the period
they wrote it, and the date range is shown in every cohort summary.

## 8. What this project does not do

- It does not train or fine-tune any model on this data.
- It does not redistribute the corpus.
- It does not profile, link, or attempt to identify reviewers.
- It does not claim clinical utility, and is not a medical device.
- It does not present its output as medical advice, and appends a standing
  disclaimer to every answer.

## 9. If you build on this

The failure mode this project addresses is not specific to medicine. Any RAG
system over an opinion corpus with a retrievable outcome variable — product
reviews, employer reviews, course evaluations, app ratings — has the same
problem, and in most of those settings nobody is checking.

If you reuse Stratified Evidence Sampling, the one thing to get right is that the
reference distribution must be estimated over the **full population**, not over
the retrieved candidates. Estimating it from retrieved candidates reproduces
exactly the bias the method exists to remove, while appearing to fix it — which
is worse than not applying it at all.
