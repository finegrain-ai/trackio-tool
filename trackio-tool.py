#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click>=8.3.1",
#     "pandas",
#     "pyarrow",
#     "huggingface-hub",
#     "matplotlib",
# ]
# ///
"""Analyze Trackio data from a local file or HuggingFace dataset.

Supports both parquet and SQLite (.db) files.

Usage:
    uv run utils/trackio-tool.py analyze my-project.parquet
    uv run utils/trackio-tool.py analyze my-project.db
    uv run utils/trackio-tool.py analyze hf://my-org/my-dataset/my-project.parquet
    uv run utils/trackio-tool.py plot my-project.parquet
    uv run utils/trackio-tool.py plot a.db,b.db --run calm-river-a3f2,b:bright-dawn-b1c7
"""

import json
import sqlite3
from pathlib import Path
from typing import cast

import click
import pandas as pd


def load_sqlite(path: Path) -> pd.DataFrame:
    """Load a Trackio SQLite database into a DataFrame."""
    con = sqlite3.connect(path)
    rows = con.execute("SELECT id, timestamp, run_name, step, metrics FROM metrics").fetchall()
    con.close()

    records = []
    for row_id, timestamp, run_name, step, metrics_json in rows:
        record = {"id": row_id, "timestamp": timestamp, "run_name": run_name, "step": step}
        record.update(json.loads(metrics_json))
        records.append(record)

    return pd.DataFrame(records)


def load_data(path: Path) -> pd.DataFrame:
    """Load Trackio data from a parquet or SQLite file."""
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".db":
        return load_sqlite(path)
    raise click.ClickException(f"Unsupported file type: {ext} (expected .parquet or .db)")


def resolve_path(data_path: str) -> tuple[str, Path]:
    """Resolve data path to (project_name, local_path).

    Supports local paths (./data/my-project.parquet, ./data/my-project.db)
    and HF paths (hf://owner/dataset/my-project.parquet).
    """
    if data_path.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        rest = data_path[len("hf://") :]
        parts = rest.split("/", 2)
        if len(parts) < 3:
            raise click.ClickException(f"Invalid HF path: {data_path} (expected hf://owner/dataset/file)")
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2]

        local_path = Path(hf_hub_download(repo_id, filename, repo_type="dataset"))
        project = Path(filename).stem
        return project, local_path

    local_path = Path(data_path)
    project = local_path.stem
    return project, local_path


def _load(data_path: str) -> tuple[str, pd.DataFrame]:
    """Resolve and load a data file, returning (project_name, dataframe)."""
    project, local_path = resolve_path(data_path)
    if not local_path.exists():
        raise click.ClickException(f"Not found: {local_path}")
    return project, load_data(local_path)


@click.group()
def cli():
    """Analyze Trackio data."""


@cli.command()
@click.argument("data_path")
def analyze(data_path: str):
    """Print summary of all runs in the project."""
    project, df = _load(data_path)

    metric_cols = [c for c in df.columns if c not in ("id", "timestamp", "run_name", "step")]

    print(f"Project: {project}")
    print(f"Total rows: {len(df)}")
    print(f"Metrics: {', '.join(metric_cols)}")
    print()

    # Per-run summary
    runs = (
        df.groupby("run_name")
        .agg(
            rows=("step", "count"),
            step_min=("step", "min"),
            step_max=("step", "max"),
            first=("timestamp", "min"),
            last=("timestamp", "max"),
        )
        .sort_values("rows", ascending=False)
    )  # type: ignore[call-overload]

    for name, row in runs.iterrows():
        first = str(row["first"])[:16].replace("T", " ")
        last = str(row["last"])[:16].replace("T", " ")
        run_df = df[df["run_name"] == name]
        tracked = [c for c in metric_cols if cast(pd.Series, run_df[c]).notna().any()]
        print(f"{name}")
        print(f"  Rows: {row['rows']}  Steps: {row['step_min']} – {row['step_max']}")
        print(f"  First: {first}  Last: {last}")
        print(f"  Columns: {', '.join(tracked)}")
        print()


def _load_multi(data_paths: str) -> tuple[str, pd.DataFrame]:
    """Load one or more comma-separated data files into a single DataFrame.

    Adds a ``_display`` column with the label to use in legends:
    - bare run_name when unique across all projects
    - project:run_name when a run_name appears in multiple projects
    """
    paths = [p.strip() for p in data_paths.split(",")]
    frames: list[pd.DataFrame] = []
    projects: list[str] = []
    for p in paths:
        project, df = _load(p)
        df = df.copy()
        df["_project"] = project
        frames.append(df)
        projects.append(project)

    combined = pd.concat(frames, ignore_index=True)

    # Determine which run_names are ambiguous (appear in >1 project)
    run_projects = combined.groupby("run_name")["_project"].apply(set)
    ambiguous_runs = {name for name, projs in run_projects.items() if len(projs) > 1}

    combined["_display"] = combined.apply(
        lambda r: f"{r['_project']}:{r['run_name']}" if r["run_name"] in ambiguous_runs else r["run_name"],
        axis=1,
    )

    title = ", ".join(dict.fromkeys(projects))  # unique, ordered
    return title, combined


def _resolve_run_filter(
    runs_arg: str,
    df: pd.DataFrame,
) -> list[str]:
    """Resolve a comma-separated --run filter to a list of _display names.

    Accepts both ``run-name`` and ``project:run-name`` syntax.
    Errors if a bare ``run-name`` matches runs in multiple projects.
    """
    display_names = set(df["_display"].unique())
    # Build lookup: bare run_name -> list of _display names
    bare_to_display: dict[str, list[str]] = {}
    for d in display_names:
        bare = d.split(":", 1)[1] if ":" in d else d
        bare_to_display.setdefault(bare, []).append(d)

    result: list[str] = []
    for token in (t.strip() for t in runs_arg.split(",")):
        if token in display_names:
            # Exact match on a _display name (qualified or bare-and-unique)
            result.append(token)
        elif token in bare_to_display:
            matches = bare_to_display[token]
            if len(matches) > 1:
                raise click.ClickException(
                    f"Ambiguous run name '{token}' exists in multiple projects: "
                    + ", ".join(sorted(matches))
                    + ". Use project:run-name to disambiguate."
                )
            result.append(matches[0])
        else:
            raise click.ClickException(f"Run '{token}' not found")
    return result


@cli.command()
@click.argument("data_paths")
@click.option("--run", "runs", default=None, help="Comma-separated run names (default: all)")
@click.option("-o", "--output", default="plot.webp", help="Output file (default: plot.webp)")
def plot(data_paths: str, runs: str | None, output: str):
    """Plot metrics for runs, stacked in a column.

    DATA_PATHS is one or more data files, comma-separated
    (e.g. a.db,b.db).
    """
    import matplotlib.pyplot as plt

    title, df = _load_multi(data_paths)
    if runs:
        run_names = _resolve_run_filter(runs, df)
    else:
        run_names = list(dict.fromkeys(df["_display"]))  # unique, ordered

    meta_cols = {"id", "timestamp", "run_name", "step", "_project", "_display"}
    metric_cols = [c for c in df.columns if c not in meta_cols]
    # Only plot numeric metrics (skip image columns etc.)
    numeric_metrics = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_metrics:
        raise click.ClickException("No numeric metrics found")

    fig, axes = plt.subplots(len(numeric_metrics), 1, figsize=(12, 4 * len(numeric_metrics)))
    if len(numeric_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, numeric_metrics, strict=True):
        for run_name in run_names:
            run_df = df[df["_display"] == run_name]
            assert isinstance(run_df, pd.DataFrame)
            run_df = run_df.sort_values("step")
            if run_df.empty:
                print(f"Warning: run '{run_name}' not found")
                continue
            series = run_df[["step", metric]].dropna()
            if series.empty:
                continue
            ax.plot(series["step"], series[metric], linewidth=1, label=run_name)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        if len(run_names) > 1:
            ax.legend()

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")


if __name__ == "__main__":
    cli()
