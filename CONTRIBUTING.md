# Contributing to PHAROS

Thanks for looking. This document says what the project expects, so you can
judge before you invest time whether a contribution is likely to land.

PHAROS makes a specific empirical claim — that retrieval over an opinion corpus
systematically misrepresents the population it summarises, and that quota-based
sampling corrects it. Everything here follows from wanting that claim to stay
falsifiable.

## Setup

```bash
git clone https://github.com/ocansey/pharos-rx.git
cd pharos-rx
pip install -e ".[dev]"
pre-commit install
```

Then, to work against real data:

```bash
make build          # fetch, clean, segment, index — about 6 minutes
pharos info         # confirms what is built
```

The test suite needs none of that. It constructs a synthetic corpus in memory,
so `pytest` works on a fresh clone with no download and no API key.

## Before you open a pull request

```bash
make check          # ruff, mypy, pytest — the same three things CI runs
```

CI additionally runs the suite on Python 3.10, 3.11 and 3.12, and on **both
pandas 2.x and pandas 3.x**. That last one is not paranoia: the two majors
differ in ways that change behaviour rather than raise. `Series + "\x00" +
Series` silently drops the NUL on pandas 2, which once made every cohort lookup
return empty while passing every test on pandas 3. If you touch anything that
builds a key, groups, or joins, please run against both.

## What the project cares about

**Explain *why*, not *what*.** The code says what it does. Comments and
docstrings should say why it does it that way, and ideally what the obvious
alternative was and why it was rejected. `docs/METHODS.md` is the model.

**Numbers come from computation, never from a language model.** Any quantity
reaching a user must originate in `pharos/data/cohort.py` and carry a `STAT-`
identifier the verifier can check. A pull request that lets the generator
produce a count, a percentage, or a frequency word will be declined, however
well it reads.

**Claims must be measurable.** If you add a mechanism, add the ablation that
isolates it and the metric that would show it failing. `configs/ablations/`
holds one overlay per mechanism, and `pharos evaluate` regenerates
`docs/RESULTS.md` from the shipped code.

**Report the cost.** The results table shows the naive baseline beating PHAROS
on nDCG and recall, because it does. Contributions that improve one metric at
the expense of another are welcome; contributions that report only the improved
one are not.

**Determinism is a feature.** Seed everything. Do not use Python's built-in
`hash` for anything persisted — it is salted per process, and it will make the
corpus differ between runs. `blake2b` is used throughout for this reason.

## Tests

Every behavioural change needs a test that fails without it. Beyond that:

- **Name the behaviour, not the function.** `test_minority_stratum_is_never_rounded_out`
  tells a reader what the system guarantees; `test_allocate_2` does not.
- **Test the invariant, not the implementation.** A test that pins an internal
  data structure blocks refactoring without protecting anything.
- **Fix a bug, add the regression test.** See `TestCohortIndexKeys` for the
  shape — it documents the failure it prevents, so the next person understands
  why the constraint exists.
- Mark anything slow with `@pytest.mark.slow` so `-m "not slow"` stays fast.

## Especially welcome

- **A neural-encoder evaluation.** The `sentence-transformers` path is
  implemented and config-switchable but was never benchmarked — the development
  environment had no access to model weights. Running
  `configs/ablations/full-corpus.yaml` and reporting the numbers would close a
  stated gap in `docs/METHODS.md` §8.
- **Testing the mechanism on another corpus.** Stratified Evidence Sampling
  should apply to any corpus with a retrievable outcome variable — product
  reviews, employer reviews, course evaluations. Nothing here demonstrates that.
- **A redundancy metric.** MMR and the per-review cap are currently
  unjustified by the results table, because no metric measures the redundancy
  they exist to prevent. A metric that does would either vindicate them or
  argue for their removal, and either outcome is an improvement.
- **Human annotation of a gold-set sample.** Relevance is derived from metadata,
  which is a necessary but not sufficient proxy. A few hundred hand-judged
  queries would bound the error.
- **Aspect-labeller agreement study.** The weak labeller's error rate is
  currently asserted rather than measured.

## Not in scope

- **Clinical or diagnostic features.** The refusal of individualised medical
  questions is a design position, documented in `docs/ETHICS.md`, not a gap to
  be filled.
- **Presenting the disproportionality statistics as pharmacovigilance signals.**
  They are computed over self-selected reviews, not spontaneous adverse-event
  reports. The caveat lives in the data structures on purpose.
- **Removing the standing disclaimer** from generated answers.
- **Redistributing the corpus.** It stays with its publishers under their terms.

## Reporting a bug

Include: what you ran, what happened, what you expected, and your Python,
pandas, and PHAROS versions (`pharos info` prints the last one plus the config
fingerprint). If it involves the corpus, `pharos info` also reports whether the
built index matches your active configuration — a mismatch there explains a
surprising number of surprises.

## Licence

Contributions are accepted under the MIT licence, matching the project.
