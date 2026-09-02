"""Command-line interface.

pharos fetch-data       download the corpus and verify its checksums
pharos audit-labels     reproduce the condition-label forensics
pharos build-corpus     clean, subsample, segment, label
pharos build-index      fit the encoder and build the indices
pharos ask              answer one question through the full graph
pharos retrieve         inspect a panel without generating
pharos evaluate         run the ablation study and write docs/RESULTS.md
pharos redteam          run the safety suite
pharos info             show what is built and what it was built from
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pharos import __version__
from pharos.config import load_config

app = typer.Typer(
    name="pharos",
    help="PHAROS — cohort-grounded retrieval over patient drug reviews. A lighthouse, not a prescription.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

CONFIG_OPT = typer.Option(
    None, "--config", "-c", help="YAML config layered over configs/default.yaml"
)


def _progress(message: str) -> None:
    console.print(f"  [dim]·[/dim] {message}")


def _load_stack(cfg, need_index: bool = True):
    """Load corpus, statistics, index and retriever, with actionable errors."""
    from pharos.data.cohort import CohortStatistics
    from pharos.index.build import load_corpus, load_index
    from pharos.retrieval.retriever import PharosRetriever

    try:
        reviews, units, meta = load_corpus(cfg)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    stats = CohortStatistics(
        reviews, units, min_support=cfg.data.min_cohort_support, seed=cfg.data.seed
    )
    if not need_index:
        return reviews, units, meta, stats, None, None

    try:
        index = load_index(cfg)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    return reviews, units, meta, stats, index, PharosRetriever(index, stats, cfg)


# --------------------------------------------------------------------------- #
@app.command("fetch-data")
def fetch_data(
    config: str | None = CONFIG_OPT,
    force: bool = typer.Option(False, help="Re-download even if the files are present"),
) -> None:
    """Download the Drugs.com review corpus and verify its integrity."""
    from pharos.data.acquire import fetch_corpus

    cfg = load_config(config)
    fetch_corpus(cfg, force=force, progress=_progress)


@app.command("audit-labels")
def audit_labels(
    config: str | None = CONFIG_OPT,
    top: int = typer.Option(30, help="How many affected labels to display"),
    out: str | None = typer.Option(None, help="Write the full audit to this CSV"),
) -> None:
    """Reproduce the condition-label forensics on the raw corpus."""
    from pharos.data.clean import ConditionRepairer, load_raw

    cfg = load_config(config)
    raw = load_raw(cfg)
    audit = ConditionRepairer().audit(raw["condition"])

    affected = audit[audit["status"] != "ok"]
    total_rows = int(audit["n"].sum())
    affected_rows = int(affected["n"].sum())

    console.print(
        Panel(
            f"[bold]{len(audit):,}[/bold] distinct labels over [bold]{total_rows:,}[/bold] rows\n"
            f"[bold]{len(affected):,}[/bold] labels affected, covering "
            f"[bold]{affected_rows:,}[/bold] rows "
            f"([bold]{100 * affected_rows / total_rows:.1f}%[/bold] of the corpus)",
            title="Condition-label audit",
        )
    )

    table = Table(show_header=True, header_style="bold")
    for column in ("raw label", "rows", "status", "repaired to"):
        table.add_column(column)
    for row in affected.head(top).itertuples(index=False):
        table.add_row(repr(row.raw), f"{row.n:,}", row.status, str(row.repaired))
    console.print(table)

    by_status = affected.groupby("status")["n"].sum().sort_values(ascending=False)
    console.print("\n[bold]rows by status[/bold]")
    for status, n in by_status.items():
        console.print(f"  {status:<18} {n:>8,}")

    if out:
        audit.to_csv(out, index=False)
        console.print(f"\n[green]full audit written to {out}[/green]")


@app.command("build-corpus")
def build_corpus_cmd(config: str | None = CONFIG_OPT) -> None:
    """Clean, subsample, segment, and label the corpus."""
    from pharos.index.build import build_corpus, save_corpus

    cfg = load_config(config)
    console.print(f"[bold]building corpus[/bold]  config fingerprint {cfg.fingerprint()}")
    reviews, units, meta = build_corpus(cfg, progress=_progress)
    save_corpus(reviews, units, meta, cfg)

    table = Table(show_header=False, box=None)
    for key in (
        "rows_in",
        "rows_out",
        "entities_unescaped",
        "condition_repaired_leading",
        "condition_repaired_trailing",
        "condition_artifact",
        "exact_duplicates",
        "near_duplicates",
        "reviews_after_subsample",
        "evidence_units",
        "distinct_drugs",
        "distinct_conditions",
        "build_seconds",
    ):
        if key in meta:
            value = meta[key]
            table.add_row(
                key.replace("_", " "), f"{value:,}" if isinstance(value, int) else str(value)
            )
    console.print(Panel(table, title="corpus report"))
    console.print(f"[green]written to {cfg.paths.processed_dir}[/green]")


@app.command("build-index")
def build_index_cmd(config: str | None = CONFIG_OPT) -> None:
    """Fit the encoder and build the dense, lexical, and metadata indices."""
    from pharos.index.build import build_index, load_corpus

    cfg = load_config(config)
    _reviews, units, _meta = load_corpus(cfg)
    index = build_index(units, cfg, progress=_progress)
    index.save(cfg.paths.index_dir)

    table = Table(show_header=False, box=None)
    for key, value in index.build_info.items():
        table.add_row(
            str(key).replace("_", " "), f"{value:,}" if isinstance(value, int) else str(value)
        )
    console.print(Panel(table, title="index"))
    console.print(f"[green]written to {cfg.paths.index_dir}[/green]")


# --------------------------------------------------------------------------- #
@app.command()
def ask(
    question: str = typer.Argument(..., help="A question about what reviewers reported"),
    config: str | None = CONFIG_OPT,
    drug: str | None = typer.Option(None, help="Constrain to this drug"),
    condition: str | None = typer.Option(None, help="Constrain to this indication"),
    show_trace: bool = typer.Option(False, "--trace", help="Print the node-by-node trace"),
    show_evidence: bool = typer.Option(False, "--evidence", help="Print the evidence panel"),
    json_out: bool = typer.Option(False, "--json", help="Emit the full final state as JSON"),
) -> None:
    """Answer a question through the full graph."""
    from pharos.agent.graph import PharosAgent
    from pharos.llm.factory import provider_is_live

    cfg = load_config(config)
    _r, _u, _m, stats, _index, retriever = _load_stack(cfg)

    if not provider_is_live(cfg.llm):
        console.print(
            "[yellow]Running with the deterministic mock generator. Prose will be "
            "mechanical; every other stage is real. Set PHAROS_LLM__PROVIDER and the "
            "matching API key to generate for real.[/yellow]\n"
        )

    agent = PharosAgent(retriever, stats, cfg)
    state = agent.ask(question, drug_name=drug, condition=condition)

    if json_out:
        console.print_json(json.dumps(state, default=str))
        return

    console.print(Panel(state.get("answer", ""), title=question, border_style="cyan"))

    if show_evidence:
        console.print(Panel(state.get("statistics_text", ""), title="computed statistics"))
        console.print(Panel(state.get("panel_text", ""), title="evidence panel"))

    if show_trace:
        table = Table(show_header=True, header_style="bold")
        table.add_column("node")
        table.add_column("detail")
        for event in state.get("trace", []):
            node = event.get("node", "")
            detail = ", ".join(f"{k}={v}" for k, v in event.items() if k != "node")
            table.add_row(node, detail)
        console.print(table)

        claims = state.get("claims", [])
        if claims:
            ctable = Table(show_header=True, header_style="bold")
            ctable.add_column("verdict")
            ctable.add_column("claim")
            ctable.add_column("reason")
            for claim in claims:
                style = "green" if claim["verdict"] == "SUPPORTED" else "red"
                ctable.add_row(
                    f"[{style}]{claim['verdict']}[/{style}]",
                    claim["text"][:80],
                    claim.get("reason", ""),
                )
            console.print(ctable)


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Query text"),
    config: str | None = CONFIG_OPT,
    k: int | None = typer.Option(None, help="Panel size"),
    no_stratify: bool = typer.Option(False, help="Disable Stratified Evidence Sampling"),
) -> None:
    """Inspect a retrieval panel without generating an answer."""
    cfg = load_config(config)
    _r, _u, _m, _stats, _index, retriever = _load_stack(cfg)

    result = retriever.retrieve(query, k=k, stratify=None if not no_stratify else False)

    console.print(
        Panel(
            f"plan: {result.plan.describe()}\n"
            f"candidates: {result.n_candidates}   panel: {len(result.units)}   "
            f"stratified: {result.stratified}\n"
            f"cohort mean rating: "
            f"{result.cohort_mean_rating:.2f}"
            if result.cohort_mean_rating
            else "cohort mean: n/a",
            title=query,
        )
    )
    console.print(
        f"[bold]valence skew (JSD)[/bold] {result.valence_skew:.4f}    "
        f"[bold]rating error[/bold] {result.rating_error:.3f}"
    )
    if result.allocation:
        console.print(f"[dim]allocation {json.dumps(result.allocation)}[/dim]")

    table = Table(show_header=True, header_style="bold")
    for column in ("unit", "rating", "stratum", "drug", "text"):
        table.add_column(column)
    for ru in result.units:
        u = ru.unit
        table.add_row(u.unit_id, f"{u.rating:.0f}", u.stratum.value, u.drug_name, u.text[:70] + "…")
    console.print(table)


# --------------------------------------------------------------------------- #
@app.command()
def evaluate(
    config: str | None = CONFIG_OPT,
    n_queries: int | None = typer.Option(None, help="Override the gold-set size"),
    ablations: str | None = typer.Option(
        None, help="Comma-separated ablation names; default runs all"
    ),
    write_results: bool = typer.Option(True, help="Write docs/RESULTS.md and artifacts/"),
) -> None:
    """Run the ablation study and regenerate the results tables."""
    from pharos.eval.goldset import build_goldset
    from pharos.eval.report import write_results_markdown
    from pharos.eval.run import (
        compare_to_baseline,
        corpus_fidelity_report,
        fidelity_by_cohort_size,
        run_ablations,
        save_results,
        suite_results_to_payload,
    )

    cfg = load_config(config)
    if n_queries:
        cfg.eval.n_queries = n_queries

    _reviews, _units, corpus_meta, stats, index, _retriever = _load_stack(cfg)

    console.print(f"[bold]building gold set[/bold] (target {cfg.eval.n_queries} queries)")
    goldset = build_goldset(index.units, n_queries=cfg.eval.n_queries, seed=cfg.eval.seed)
    console.print(f"  {len(goldset)} queries with sufficient support")
    if not goldset:
        console.print("[red]gold set is empty — the corpus is too small to evaluate[/red]")
        raise typer.Exit(1)

    selected = [s.strip() for s in ablations.split(",")] if ablations else None
    results = run_ablations(index, stats, cfg, goldset, selected=selected, progress=_progress)
    deltas = compare_to_baseline(results, n_boot=cfg.eval.bootstrap_n, seed=cfg.eval.seed)
    fidelity = corpus_fidelity_report(stats, goldset)
    size_bands = fidelity_by_cohort_size(results)

    # --- console table ------------------------------------------------- #
    from pharos.eval.run import HEADLINE_METRICS

    table = Table(show_header=True, header_style="bold")
    table.add_column("configuration")
    for metric in HEADLINE_METRICS:
        table.add_column(metric)
    for result in results:
        table.add_row(
            result.name,
            *[
                f"{result.metrics[m][0]:.4f}" if m in result.metrics else "—"
                for m in HEADLINE_METRICS
            ],
        )
    console.print(table)

    if write_results:
        payload = suite_results_to_payload(
            results,
            deltas,
            extra={
                "corpus": corpus_meta,
                "index": index.build_info,
                "fidelity": fidelity,
                "fidelity_by_cohort_size": size_bands,
                "goldset": [q.to_dict() for q in goldset[:50]],
                "n_goldset": len(goldset),
            },
        )
        artifacts = Path(cfg.paths.artifacts_dir)
        save_results(payload, artifacts / "results.json")
        path = write_results_markdown(
            results, deltas, fidelity, corpus_meta, index.build_info, cfg, len(goldset), size_bands
        )
        console.print(f"[green]wrote {artifacts / 'results.json'} and {path}[/green]")


@app.command()
def redteam(
    config: str | None = CONFIG_OPT,
    write_results: bool = typer.Option(True, help="Write artifacts/redteam.json"),
) -> None:
    """Run the red-team safety suite through the full graph."""
    from pharos.agent.graph import PharosAgent
    from pharos.eval.run import evaluate_safety, save_results

    cfg = load_config(config)
    _r, _u, _m, stats, _index, retriever = _load_stack(cfg)
    agent = PharosAgent(retriever, stats, cfg)

    report = evaluate_safety(agent)

    table = Table(show_header=True, header_style="bold")
    for column in ("category", "passed", "n", "rate", "failures"):
        table.add_column(column)
    for category, bucket in report["by_category"].items():
        style = "green" if bucket["rate"] == 1.0 else "red"
        table.add_row(
            category,
            str(bucket["passed"]),
            str(bucket["n"]),
            f"[{style}]{bucket['rate']:.2%}[/{style}]",
            ", ".join(bucket["failures"]) or "—",
        )
    console.print(table)
    console.print(
        f"\n[bold]overall {report['n_passed']}/{report['n_probes']} "
        f"({report['overall_rate']:.2%})[/bold]"
    )

    if write_results:
        path = Path(cfg.paths.artifacts_dir) / "redteam.json"
        save_results(report, path)
        console.print(f"[green]wrote {path}[/green]")

    if report["overall_rate"] < 1.0:
        raise typer.Exit(1)


@app.command()
def info(config: str | None = CONFIG_OPT) -> None:
    """Show what is built, and what it was built from."""
    cfg = load_config(config)
    console.print(f"[bold]PHAROS {__version__}[/bold]   config fingerprint {cfg.fingerprint()}")

    table = Table(show_header=True, header_style="bold")
    for column in ("artifact", "path", "state"):
        table.add_column(column)

    raw_ok = (cfg.paths.raw_dir / "drugsComTrain_raw.tsv").exists()
    corpus_ok = (cfg.paths.processed_dir / "units.parquet").exists()
    index_ok = (cfg.paths.index_dir / "vectors.npy").exists()
    for name, path, ok in (
        ("raw corpus", cfg.paths.raw_dir, raw_ok),
        ("processed corpus", cfg.paths.processed_dir, corpus_ok),
        ("index", cfg.paths.index_dir, index_ok),
    ):
        table.add_row(name, str(path), "[green]present[/green]" if ok else "[red]missing[/red]")
    console.print(table)

    report_path = cfg.paths.processed_dir / "corpus_report.json"
    if report_path.exists():
        meta = json.loads(report_path.read_text())
        console.print(
            f"\ncorpus: [bold]{meta.get('reviews_after_subsample', 0):,}[/bold] reviews, "
            f"[bold]{meta.get('evidence_units', 0):,}[/bold] evidence units, "
            f"[bold]{meta.get('distinct_drugs', 0):,}[/bold] drugs, "
            f"[bold]{meta.get('distinct_conditions', 0):,}[/bold] conditions"
        )

    info_path = cfg.paths.index_dir / "build_info.json"
    if info_path.exists():
        build = json.loads(info_path.read_text())
        console.print(
            f"index:  encoder [bold]{build.get('encoder')}[/bold], "
            f"dim [bold]{build.get('embedding_dim')}[/bold], "
            f"BM25 vocabulary [bold]{build.get('bm25_vocabulary', 0):,}[/bold]"
        )
        if build.get("config_fingerprint") != cfg.fingerprint():
            console.print(
                "[yellow]the built index was produced by a different configuration; "
                "rebuild before trusting any evaluation numbers[/yellow]"
            )

    console.print(f"\ngenerator: [bold]{cfg.llm.provider}[/bold]")


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
