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
    ./trackio-tool.py analyze my-project.parquet
    ./trackio-tool.py analyze my-project.db
    ./trackio-tool.py analyze hf://my-org/my-dataset/my-project.parquet
    ./trackio-tool.py plot my-project.parquet
    ./trackio-tool.py plot a.db,b.db --run calm-river-a3f2,b:bright-dawn-b1c7
    ./trackio-tool.py merge --from bar.parquet --into foo.db
    ./trackio-tool.py merge --from hf://my-org/my-dataset/bar.parquet --into foo.db
"""

import json
import math
import shutil
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
    """Work with Trackio data files."""


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


def resolve_path_for_download(data_path: str, filename: str) -> Path | None:
    """Download a specific filename from the same HF repo as data_path.

    Returns the local path on success, None if the file doesn't exist.
    Only meaningful for hf:// paths; for local paths, resolves relative to the
    same directory.
    """
    if data_path.startswith("hf://"):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        rest = data_path[len("hf://"):]
        parts = rest.split("/", 2)
        repo_id = f"{parts[0]}/{parts[1]}"
        try:
            return Path(hf_hub_download(repo_id, filename, repo_type="dataset"))
        except EntryNotFoundError:
            return None

    local = Path(data_path).parent / filename
    return local if local.exists() else None


# SQL for creating tables in the target DB when they don't exist yet.
_CREATE_METRICS = """
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    run_name TEXT,
    step INTEGER,
    metrics TEXT
)
"""
_CREATE_SYSTEM_METRICS = """
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    run_name TEXT,
    metrics TEXT
)
"""
_CREATE_CONFIGS = """
CREATE TABLE IF NOT EXISTS configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT UNIQUE,
    config TEXT,
    created_at TEXT
)
"""


def _ensure_tables(con: sqlite3.Connection) -> None:
    """Create the three trackio tables if they don't already exist."""
    con.execute(_CREATE_METRICS)
    con.execute(_CREATE_SYSTEM_METRICS)
    con.execute(_CREATE_CONFIGS)
    con.commit()


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def get_run_names_from_db(path: Path) -> set[str]:
    """Return the union of all run_name values across all three tables."""
    con = sqlite3.connect(path)
    names: set[str] = set()
    for table in ("metrics", "configs", "system_metrics"):
        if _table_exists(con, table):
            rows = con.execute(f"SELECT DISTINCT run_name FROM {table}").fetchall()  # noqa: S608
            names.update(r[0] for r in rows)
    con.close()
    return names


def get_run_names_from_parquet(path: Path, data_path: str) -> set[str]:
    """Collect run_name values from a parquet file and its companions."""
    names: set[str] = set()
    df = pd.read_parquet(path, columns=["run_name"])
    names.update(df["run_name"].unique())

    stem = path.stem
    for suffix in ("_system.parquet", "_configs.parquet"):
        companion = resolve_path_for_download(data_path, stem + suffix)
        if companion is not None:
            cdf = pd.read_parquet(companion, columns=["run_name"])
            names.update(cdf["run_name"].unique())
    return names


def _serialize_special_floats(value: object) -> object:
    """Convert inf/nan floats to JSON-safe string representations."""
    if isinstance(value, float):
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if math.isnan(value):
            return "NaN"
    return value


def _df_to_json_col(df: pd.DataFrame, meta_cols: list[str]) -> list[str]:
    """Pack non-meta columns of a DataFrame into a list of JSON strings."""
    data_cols = [c for c in df.columns if c not in meta_cols]
    result: list[str] = []
    for _, row in df.iterrows():
        d = {c: _serialize_special_floats(row[c]) for c in data_cols if pd.notna(row[c])}
        result.append(json.dumps(d))
    return result


def _merge_from_db(source: Path, target_con: sqlite3.Connection) -> dict[str, int]:
    """Copy rows from source DB into target_con. Returns row counts per table."""
    src = sqlite3.connect(source)
    counts: dict[str, int] = {}

    # metrics
    if _table_exists(src, "metrics"):
        rows = src.execute("SELECT timestamp, run_name, step, metrics FROM metrics").fetchall()
        target_con.executemany(
            "INSERT INTO metrics (timestamp, run_name, step, metrics) VALUES (?, ?, ?, ?)",
            rows,
        )
        counts["metrics"] = len(rows)

    # system_metrics
    if _table_exists(src, "system_metrics"):
        rows = src.execute("SELECT timestamp, run_name, metrics FROM system_metrics").fetchall()
        target_con.executemany(
            "INSERT INTO system_metrics (timestamp, run_name, metrics) VALUES (?, ?, ?)",
            rows,
        )
        counts["system_metrics"] = len(rows)

    # configs
    if _table_exists(src, "configs"):
        rows = src.execute("SELECT run_name, config, created_at FROM configs").fetchall()
        target_con.executemany(
            "INSERT INTO configs (run_name, config, created_at) VALUES (?, ?, ?)",
            rows,
        )
        counts["configs"] = len(rows)

    target_con.commit()
    src.close()
    return counts


def _merge_from_parquet(
    source: Path, data_path: str, target_con: sqlite3.Connection
) -> dict[str, int]:
    """Import parquet (+ companions) into target_con. Returns row counts per table."""
    counts: dict[str, int] = {}
    stem = source.stem

    # Main metrics parquet
    df = pd.read_parquet(source)
    if not df.empty:
        meta = ["id", "timestamp", "run_name", "step"]
        json_col = _df_to_json_col(df, meta)
        rows = list(zip(df["timestamp"], df["run_name"], df["step"], json_col))
        target_con.executemany(
            "INSERT INTO metrics (timestamp, run_name, step, metrics) VALUES (?, ?, ?, ?)",
            rows,
        )
        counts["metrics"] = len(rows)

    # System metrics companion
    sys_path = resolve_path_for_download(data_path, stem + "_system.parquet")
    if sys_path is not None:
        sdf = pd.read_parquet(sys_path)
        if not sdf.empty:
            meta = ["id", "timestamp", "run_name"]
            json_col = _df_to_json_col(sdf, meta)
            rows = list(zip(sdf["timestamp"], sdf["run_name"], json_col))
            target_con.executemany(
                "INSERT INTO system_metrics (timestamp, run_name, metrics) VALUES (?, ?, ?)",
                rows,
            )
            counts["system_metrics"] = len(rows)
    else:
        click.echo("Warning: no companion _system.parquet found; system_metrics not merged.")

    # Configs companion
    cfg_path = resolve_path_for_download(data_path, stem + "_configs.parquet")
    if cfg_path is not None:
        cdf = pd.read_parquet(cfg_path)
        if not cdf.empty:
            meta = ["id", "run_name", "created_at"]
            json_col = _df_to_json_col(cdf, meta)
            created = cdf["created_at"] if "created_at" in cdf.columns else [None] * len(cdf)
            rows = list(zip(cdf["run_name"], json_col, created))
            target_con.executemany(
                "INSERT INTO configs (run_name, config, created_at) VALUES (?, ?, ?)",
                rows,
            )
            counts["configs"] = len(rows)
    else:
        click.echo("Warning: no companion _configs.parquet found; configs not merged.")

    target_con.commit()
    return counts


@cli.command()
@click.option("--from", "from_path", required=True, help="Source file (local .db/.parquet or hf://...)")
@click.option("--into", "into_path", required=True, help="Target local .db file")
@click.option("--media/--no-media", default=True, help="Copy media directories (default: --media)")
def merge(from_path: str, into_path: str, media: bool):
    """Merge runs from one project file into another .db file."""
    # Validate target
    target = Path(into_path)
    if target.suffix.lower() != ".db":
        raise click.ClickException(f"--into must be a .db file, got: {into_path}")
    if not target.exists():
        raise click.ClickException(f"Target file not found: {into_path}")

    # Resolve source
    source_project, source_path = resolve_path(from_path)
    ext = source_path.suffix.lower()
    if ext not in (".db", ".parquet"):
        raise click.ClickException(f"--from must be a .db or .parquet file, got: {from_path}")
    if not source_path.exists():
        raise click.ClickException(f"Source file not found: {source_path}")

    # Collect run names and check for conflicts
    target_runs = get_run_names_from_db(target)
    if ext == ".db":
        source_runs = get_run_names_from_db(source_path)
    else:
        source_runs = get_run_names_from_parquet(source_path, from_path)

    overlap = target_runs & source_runs
    if overlap:
        names = ", ".join(sorted(overlap))
        raise click.ClickException(f"Conflicting run names in source and target: {names}")

    if not source_runs:
        raise click.ClickException("Source contains no runs.")

    # Ensure target tables exist
    target_con = sqlite3.connect(target)
    _ensure_tables(target_con)

    # Merge
    if ext == ".db":
        counts = _merge_from_db(source_path, target_con)
    else:
        counts = _merge_from_parquet(source_path, from_path, target_con)

    target_con.close()

    # Copy media directories
    media_copied = 0
    if media:
        target_project = target.stem
        target_media_base = target.parent / "media" / target_project

        if from_path.startswith("hf://"):
            from huggingface_hub import snapshot_download

            rest = from_path[len("hf://"):]
            parts = rest.split("/", 2)
            repo_id = f"{parts[0]}/{parts[1]}"
            for run_name in source_runs:
                pattern = f"media/{source_project}/{run_name}/**"
                try:
                    snap_dir = Path(snapshot_download(
                        repo_id, repo_type="dataset", allow_patterns=[pattern],
                    ))
                    src_run_dir = snap_dir / "media" / source_project / run_name
                    if src_run_dir.is_dir() and any(src_run_dir.iterdir()):
                        dst_run_dir = target_media_base / run_name
                        shutil.copytree(src_run_dir, dst_run_dir, dirs_exist_ok=True)
                        media_copied += 1
                except Exception:
                    pass  # media may not exist in the repo
        else:
            source_media_base = source_path.parent / "media" / source_project
            if source_media_base.is_dir():
                for run_name in source_runs:
                    src_run_dir = source_media_base / run_name
                    if src_run_dir.is_dir():
                        dst_run_dir = target_media_base / run_name
                        shutil.copytree(src_run_dir, dst_run_dir, dirs_exist_ok=True)
                        media_copied += 1

    # Summary
    click.echo(f"Merged {len(source_runs)} run(s) into {into_path}:")
    for table, n in sorted(counts.items()):
        click.echo(f"  {table}: {n} rows")
    if media and media_copied:
        click.echo(f"  media: {media_copied} run(s) copied")


if __name__ == "__main__":
    cli()
