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
"""Analyze Trackio data from a local file, HuggingFace dataset, Modal volume, or SSH server.

Supports both parquet and SQLite (.db) files.

Usage:
    ./trackio-tool.py analyze my-project.parquet
    ./trackio-tool.py analyze my-project.db
    ./trackio-tool.py analyze hf://my-org/my-dataset/my-project.parquet
    ./trackio-tool.py analyze modal://my-volume/trackio/my-project.db
    ./trackio-tool.py analyze modal://my-volume@dev/trackio/my-project.parquet
    ./trackio-tool.py analyze ssh://my-server/data/trackio/my-project.db
    ./trackio-tool.py analyze ssh://user@my-server/~/trackio/my-project.db
    ./trackio-tool.py plot my-project.parquet
    ./trackio-tool.py plot a.db,b.db --run calm-river-a3f2,b:bright-dawn-b1c7
    ./trackio-tool.py merge --from bar.parquet --into foo.db
    ./trackio-tool.py merge --from hf://my-org/my-dataset/bar.parquet --into foo.db
    ./trackio-tool.py merge --from modal://my-volume/trackio/bar.parquet --into foo.db
    ./trackio-tool.py merge --from ssh://my-server/data/trackio/bar.db --into foo.db
    ./trackio-tool.py drop my-project.db --run calm-river-a3f2
"""

import contextlib
import io
import json
import math
import shutil
import sqlite3
import stat
import tempfile
from enum import IntEnum
from pathlib import Path
from typing import cast

import click
import pandas as pd
from huggingface_hub import RepoFile, hf_hub_download, list_repo_tree, snapshot_download
from huggingface_hub.errors import EntryNotFoundError


def load_sqlite(path: Path) -> pd.DataFrame:
    """Load a Trackio SQLite database into a DataFrame."""
    with contextlib.closing(sqlite3.connect(path)) as con:
        rows = con.execute("SELECT id, timestamp, run_name, step, metrics FROM metrics").fetchall()

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


def _modal_entry_is_file(entry) -> bool:
    """Check if a Modal FileEntry is a regular file (not a directory/symlink/etc)."""
    return isinstance(entry.type, IntEnum) and entry.type.value == 1


def _parse_modal_url(url: str) -> tuple[str, str | None, str]:
    """Parse modal://<volume-name>[@<env>]/path into (volume_name, env, remote_path)."""
    rest = url[len("modal://") :]
    slash_idx = rest.find("/")
    if slash_idx < 0:
        raise click.ClickException(f"Invalid modal path: {url} (expected modal://volume/path)")
    host = rest[:slash_idx]
    remote_path = rest[slash_idx + 1 :]
    if not remote_path:
        raise click.ClickException(f"Invalid modal path: {url} (no file path after volume name)")
    if "@" in host:
        volume_name, env = host.split("@", 1)
    else:
        volume_name, env = host, None
    return volume_name, env, remote_path


_modal_tmpdir: str | None = None


def _get_modal_tmpdir() -> str:
    """Lazily create a shared temp directory for Modal downloads."""
    global _modal_tmpdir
    if _modal_tmpdir is None:
        _modal_tmpdir = tempfile.mkdtemp(prefix="trackio-modal-")
    return _modal_tmpdir


def _modal_volume(volume_name: str, env: str | None):
    """Return a modal.Volume handle, with a helpful error if modal is not installed."""
    try:
        import modal
    except ModuleNotFoundError as err:
        raise click.ClickException("modal:// paths require the modal package. Use: uv run --with modal ...") from err
    kwargs = {}
    if env is not None:
        kwargs["environment_name"] = env
    return modal.Volume.from_name(volume_name, **kwargs)


def _download_modal_file(volume_name: str, env: str | None, remote_path: str) -> Path:
    """Download a single file from a Modal volume into the temp dir.

    For .db files, also downloads the -wal and -shm companions so that
    SQLite WAL-mode data is not lost.
    """
    volume = _modal_volume(volume_name, env)
    tmpdir = Path(_get_modal_tmpdir())

    def _fetch(rpath: str) -> Path:
        buf = io.BytesIO()
        volume.read_file_into_fileobj(rpath, buf)
        buf.seek(0)
        local = tmpdir / Path(rpath).name
        local.write_bytes(buf.read())
        return local

    local_path = _fetch(remote_path)

    if local_path.suffix.lower() == ".db":
        for wal_suffix in ("-wal", "-shm"):
            try:
                _fetch(remote_path + wal_suffix)
            except FileNotFoundError:
                pass  # WAL/SHM files may not exist

    return local_path


def _parse_ssh_url(url: str) -> tuple[str, str]:
    """Parse ssh://[user@]host/path into (host, remote_path).

    The first ``/`` after the host starts the absolute remote path.
    Paths starting with ``/~/`` are treated as relative to the user's home
    directory (the leading ``/~/`` is stripped so SFTP resolves from $HOME).
    """
    rest = url[len("ssh://") :]
    slash_idx = rest.find("/")
    if slash_idx < 0:
        raise click.ClickException(f"Invalid SSH path: {url} (expected ssh://[user@]host/path)")
    host = rest[:slash_idx]
    remote_path = rest[slash_idx:]  # keep leading /
    if not host:
        raise click.ClickException(f"Invalid SSH path: {url} (missing host)")
    # /~/foo → foo (relative to home); SFTP resolves relative paths from $HOME
    if remote_path.startswith("/~/"):
        remote_path = remote_path[3:]
    if not remote_path or remote_path == "/":
        raise click.ClickException(f"Invalid SSH path: {url} (no file path after host)")
    return host, remote_path


_ssh_tmpdir: str | None = None


def _get_ssh_tmpdir() -> str:
    """Lazily create a shared temp directory for SSH downloads."""
    global _ssh_tmpdir
    if _ssh_tmpdir is None:
        _ssh_tmpdir = tempfile.mkdtemp(prefix="trackio-ssh-")
    return _ssh_tmpdir


def _ssh_connect(host: str):
    """Return a connected paramiko.SSHClient for the given host.

    Parses ``user@host`` if present and uses SSH agent / default keys for auth.
    """
    try:
        import paramiko
    except ModuleNotFoundError as err:
        raise click.ClickException(
            "ssh:// paths require the paramiko package. Use: uv run --with paramiko ..."
        ) from err
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy())
    if "@" in host:
        user, hostname = host.split("@", 1)
        client.connect(hostname, username=user)
    else:
        client.connect(host)
    return client


def _download_ssh_file(host: str, remote_path: str) -> Path:
    """Download a single file from an SSH host into the temp dir.

    For .db files, also downloads the -wal and -shm companions so that
    SQLite WAL-mode data is not lost.
    """
    with _ssh_connect(host) as client, client.open_sftp() as sftp:
        tmpdir = Path(_get_ssh_tmpdir())

        def _fetch(rpath: str) -> Path:
            local = tmpdir / Path(rpath).name
            sftp.get(rpath, str(local))
            return local

        local_path = _fetch(remote_path)

        if local_path.suffix.lower() == ".db":
            for wal_suffix in ("-wal", "-shm"):
                try:
                    _fetch(remote_path + wal_suffix)
                except FileNotFoundError:
                    pass  # WAL/SHM files may not exist

        return local_path


def resolve_path(data_path: str) -> tuple[str, Path]:
    """Resolve data path to (project_name, local_path).

    Supports local paths (./data/my-project.parquet, ./data/my-project.db),
    HF paths (hf://owner/dataset/my-project.parquet),
    Modal paths (modal://volume[@env]/path/to/file),
    and SSH paths (ssh://[user@]host/path/to/file).
    """
    if data_path.startswith("hf://"):
        rest = data_path[len("hf://") :]
        parts = rest.split("/", 2)
        if len(parts) < 3:
            raise click.ClickException(f"Invalid HF path: {data_path} (expected hf://owner/dataset/file)")
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2]

        local_path = Path(hf_hub_download(repo_id, filename, repo_type="dataset"))
        project = Path(filename).stem
        return project, local_path

    if data_path.startswith("modal://"):
        volume_name, env, remote_path = _parse_modal_url(data_path)
        local_path = _download_modal_file(volume_name, env, remote_path)
        project = Path(remote_path).stem
        return project, local_path

    if data_path.startswith("ssh://"):
        host, remote_path = _parse_ssh_url(data_path)
        local_path = _download_ssh_file(host, remote_path)
        project = Path(remote_path).stem
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


def _count_media_files(data_path: str, project: str, run_name: str) -> int:
    """Count media files for a run. Returns 0 if media dir doesn't exist."""
    if data_path.startswith("modal://"):
        volume_name, env, remote_path = _parse_modal_url(data_path)
        volume = _modal_volume(volume_name, env)
        remote_dir = str(Path(remote_path).parent)
        media_prefix = f"{remote_dir}/media/{project}/{run_name}"
        import modal.exception

        try:
            entries = list(volume.iterdir(media_prefix, recursive=True))
            return sum(1 for e in entries if _modal_entry_is_file(e))
        except modal.exception.NotFoundError:
            return 0

    if data_path.startswith("hf://"):
        rest = data_path[len("hf://") :]
        parts = rest.split("/", 2)
        repo_id = f"{parts[0]}/{parts[1]}"
        prefix = f"media/{project}/{run_name}"
        try:
            return sum(
                1
                for f in list_repo_tree(repo_id, path_in_repo=prefix, repo_type="dataset", recursive=True)
                if isinstance(f, RepoFile)
            )
        except EntryNotFoundError:
            return 0

    if data_path.startswith("ssh://"):
        host, remote_path = _parse_ssh_url(data_path)
        remote_dir = str(Path(remote_path).parent)
        media_prefix = f"{remote_dir}/media/{project}/{run_name}"
        with _ssh_connect(host) as client, client.open_sftp() as sftp:
            try:
                count = 0
                dirs = [media_prefix]
                while dirs:
                    current = dirs.pop()
                    for attr in sftp.listdir_attr(current):
                        child = f"{current}/{attr.filename}"
                        if stat.S_ISDIR(attr.st_mode or 0):
                            dirs.append(child)
                        elif stat.S_ISREG(attr.st_mode or 0):
                            count += 1
                return count
            except FileNotFoundError:
                return 0

    media_dir = Path(data_path).parent / "media" / project / run_name
    if media_dir.is_dir():
        return sum(1 for f in media_dir.rglob("*") if f.is_file())
    return 0


@cli.command()
@click.argument("data_path")
@click.option("--media/--no-media", default=True, help="Include media file counts (default: --media)")
def analyze(data_path: str, media: bool):
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
        if media:
            n = _count_media_files(data_path, project, str(name))
            print(f"  Media: {n} file(s)")
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
    """Download a specific filename from the same remote directory as data_path.

    Returns the local path on success, None if the file doesn't exist.
    Supports hf://, modal://, ssh://, and local paths.
    """
    if data_path.startswith("hf://"):
        rest = data_path[len("hf://") :]
        parts = rest.split("/", 2)
        repo_id = f"{parts[0]}/{parts[1]}"
        try:
            return Path(hf_hub_download(repo_id, filename, repo_type="dataset"))
        except EntryNotFoundError:
            return None

    if data_path.startswith("modal://"):
        volume_name, env, remote_path = _parse_modal_url(data_path)
        remote_dir = str(Path(remote_path).parent)
        companion_remote = f"{remote_dir}/{filename}" if remote_dir != "." else filename
        try:
            return _download_modal_file(volume_name, env, companion_remote)
        except FileNotFoundError:
            return None

    if data_path.startswith("ssh://"):
        host, remote_path = _parse_ssh_url(data_path)
        remote_dir = str(Path(remote_path).parent)
        companion_remote = f"{remote_dir}/{filename}" if remote_dir != "." else filename
        try:
            return _download_ssh_file(host, companion_remote)
        except FileNotFoundError:
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
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def get_run_names_from_db(path: Path) -> set[str]:
    """Return the union of all run_name values across all three tables."""
    with contextlib.closing(sqlite3.connect(path)) as con:
        names: set[str] = set()
        for table in ("metrics", "configs", "system_metrics"):
            if _table_exists(con, table):
                rows = con.execute(f"SELECT DISTINCT run_name FROM {table}").fetchall()  # noqa: S608
                names.update(r[0] for r in rows)
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
        d = {c: _serialize_special_floats(row[c]) for c in data_cols if bool(pd.notna(row[c]))}
        result.append(json.dumps(d))
    return result


def _merge_from_db(
    source: Path,
    target_con: sqlite3.Connection,
    run_filter: set[str] | None = None,
) -> dict[str, int]:
    """Copy rows from source DB into target_con. Returns row counts per table."""
    counts: dict[str, int] = {}

    def _where(col: str = "run_name") -> tuple[str, list[str]]:
        if run_filter is None:
            return "", []
        placeholders = ",".join("?" for _ in run_filter)
        return f" WHERE {col} IN ({placeholders})", sorted(run_filter)

    clause, params = _where()

    with contextlib.closing(sqlite3.connect(source)) as src:
        # metrics
        if _table_exists(src, "metrics"):
            rows = src.execute(
                f"SELECT timestamp, run_name, step, metrics FROM metrics{clause}",
                params,
            ).fetchall()
            target_con.executemany(
                "INSERT INTO metrics (timestamp, run_name, step, metrics) VALUES (?, ?, ?, ?)",
                rows,
            )
            counts["metrics"] = len(rows)

        # system_metrics
        if _table_exists(src, "system_metrics"):
            rows = src.execute(
                f"SELECT timestamp, run_name, metrics FROM system_metrics{clause}",
                params,
            ).fetchall()
            target_con.executemany(
                "INSERT INTO system_metrics (timestamp, run_name, metrics) VALUES (?, ?, ?)",
                rows,
            )
            counts["system_metrics"] = len(rows)

        # configs
        if _table_exists(src, "configs"):
            rows = src.execute(
                f"SELECT run_name, config, created_at FROM configs{clause}",
                params,
            ).fetchall()
            target_con.executemany(
                "INSERT INTO configs (run_name, config, created_at) VALUES (?, ?, ?)",
                rows,
            )
            counts["configs"] = len(rows)

    target_con.commit()
    return counts


def _merge_from_parquet(
    source: Path,
    data_path: str,
    target_con: sqlite3.Connection,
    run_filter: set[str] | None = None,
) -> dict[str, int]:
    """Import parquet (+ companions) into target_con. Returns row counts per table."""
    counts: dict[str, int] = {}
    stem = source.stem

    # Main metrics parquet
    df = pd.read_parquet(source)
    if run_filter is not None:
        df = df[df["run_name"].isin(list(run_filter))]
        assert isinstance(df, pd.DataFrame)
    if not df.empty:
        meta = ["id", "timestamp", "run_name", "step"]
        json_col = _df_to_json_col(df, meta)
        rows = list(zip(df["timestamp"], df["run_name"], df["step"], json_col, strict=True))
        target_con.executemany(
            "INSERT INTO metrics (timestamp, run_name, step, metrics) VALUES (?, ?, ?, ?)",
            rows,
        )
        counts["metrics"] = len(rows)

    # System metrics companion
    sys_path = resolve_path_for_download(data_path, stem + "_system.parquet")
    if sys_path is not None:
        sdf = pd.read_parquet(sys_path)
        if run_filter is not None:
            sdf = sdf[sdf["run_name"].isin(list(run_filter))]
            assert isinstance(sdf, pd.DataFrame)
        if not sdf.empty:
            meta = ["id", "timestamp", "run_name"]
            json_col = _df_to_json_col(sdf, meta)
            rows = list(zip(sdf["timestamp"], sdf["run_name"], json_col, strict=True))
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
        if run_filter is not None:
            cdf = cdf[cdf["run_name"].isin(list(run_filter))]
            assert isinstance(cdf, pd.DataFrame)
        if not cdf.empty:
            meta = ["id", "run_name", "created_at"]
            json_col = _df_to_json_col(cdf, meta)
            created = cdf["created_at"] if "created_at" in cdf.columns else [None] * len(cdf)
            rows = list(zip(cdf["run_name"], json_col, created, strict=True))
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
@click.option("--run", "runs", default=None, help="Comma-separated run names to import (default: all)")
@click.option("--media/--no-media", default=True, help="Copy media directories (default: --media)")
@click.option("--bootstrap/--no-bootstrap", default=False, help="Create target .db if it doesn't exist")
def merge(from_path: str, into_path: str, runs: str | None, media: bool, bootstrap: bool):
    """Merge runs from one project file into another .db file."""
    # Validate target
    target = Path(into_path)
    if target.suffix.lower() != ".db":
        raise click.ClickException(f"--into must be a .db file, got: {into_path}")
    if not target.exists() and not bootstrap:
        raise click.ClickException(
            f"Target database {into_path} does not exist. Use --bootstrap to create it from --from data."
        )

    # Resolve source
    source_project, source_path = resolve_path(from_path)
    ext = source_path.suffix.lower()
    if ext not in (".db", ".parquet"):
        raise click.ClickException(f"--from must be a .db or .parquet file, got: {from_path}")
    if not source_path.exists():
        raise click.ClickException(f"Source file not found: {from_path}")

    # Collect run names and apply --run filter
    if ext == ".db":
        all_source_runs = get_run_names_from_db(source_path)
    else:
        all_source_runs = get_run_names_from_parquet(source_path, from_path)

    if runs is not None:
        requested = {r.strip() for r in runs.split(",")}
        unknown = requested - all_source_runs
        if unknown:
            raise click.ClickException(f"Run(s) not found in source: {', '.join(sorted(unknown))}")
        source_runs = requested
    else:
        source_runs = all_source_runs

    if not source_runs:
        raise click.ClickException("Source contains no runs.")

    # Check for conflicts with target
    if target.exists():
        target_runs = get_run_names_from_db(target)
        overlap = target_runs & source_runs
        if overlap:
            names = ", ".join(sorted(overlap))
            raise click.ClickException(f"Conflicting run names in source and target: {names}")

    # Ensure target tables exist
    with contextlib.closing(sqlite3.connect(target)) as target_con:
        _ensure_tables(target_con)

        # Merge
        run_filter = source_runs if runs is not None else None
        if ext == ".db":
            counts = _merge_from_db(source_path, target_con, run_filter)
        else:
            counts = _merge_from_parquet(source_path, from_path, target_con, run_filter)

    # Copy media directories
    media_files_copied = 0
    if media:
        target_project = target.stem
        target_media_base = target.parent / "media" / target_project

        if from_path.startswith("hf://"):
            rest = from_path[len("hf://") :]
            parts = rest.split("/", 2)
            repo_id = f"{parts[0]}/{parts[1]}"
            for run_name in source_runs:
                pattern = f"media/{source_project}/{run_name}/**"
                try:
                    snap_dir = Path(
                        snapshot_download(
                            repo_id,
                            repo_type="dataset",
                            allow_patterns=[pattern],
                        )
                    )
                    src_run_dir = snap_dir / "media" / source_project / run_name
                    if src_run_dir.is_dir() and any(src_run_dir.iterdir()):
                        dst_run_dir = target_media_base / run_name
                        shutil.copytree(src_run_dir, dst_run_dir, dirs_exist_ok=True)
                        media_files_copied += sum(1 for f in src_run_dir.rglob("*") if f.is_file())
                except EntryNotFoundError:
                    pass  # media may not exist in the repo
        elif from_path.startswith("modal://"):
            volume_name, env, remote_path = _parse_modal_url(from_path)
            volume = _modal_volume(volume_name, env)
            remote_dir = str(Path(remote_path).parent)
            import modal.exception

            for run_name in source_runs:
                media_prefix = f"{remote_dir}/media/{source_project}/{run_name}"
                try:
                    entries = list(volume.iterdir(media_prefix, recursive=True))
                except modal.exception.NotFoundError:
                    continue  # media may not exist on the volume
                files = [e for e in entries if _modal_entry_is_file(e)]
                if not files:
                    continue
                for i, entry in enumerate(files, 1):
                    click.echo(f"\r  media/{run_name}: {i}/{len(files)} files", nl=False)
                    rel = Path(entry.path).relative_to(media_prefix)
                    dst = target_media_base / run_name / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    buf = io.BytesIO()
                    volume.read_file_into_fileobj(entry.path, buf)
                    buf.seek(0)
                    dst.write_bytes(buf.read())
                click.echo()
                media_files_copied += len(files)
        elif from_path.startswith("ssh://"):
            host, remote_path = _parse_ssh_url(from_path)
            remote_dir = str(Path(remote_path).parent)
            with _ssh_connect(host) as client, client.open_sftp() as sftp:
                for run_name in source_runs:
                    media_prefix = f"{remote_dir}/media/{source_project}/{run_name}"
                    try:
                        file_paths: list[str] = []
                        dirs = [media_prefix]
                        while dirs:
                            current = dirs.pop()
                            for attr in sftp.listdir_attr(current):
                                child = f"{current}/{attr.filename}"
                                if stat.S_ISDIR(attr.st_mode or 0):
                                    dirs.append(child)
                                elif stat.S_ISREG(attr.st_mode or 0):
                                    file_paths.append(child)
                    except FileNotFoundError:
                        continue  # media may not exist on remote
                    if not file_paths:
                        continue
                    for i, fpath in enumerate(file_paths, 1):
                        click.echo(f"\r  media/{run_name}: {i}/{len(file_paths)} files", nl=False)
                        rel = Path(fpath).relative_to(media_prefix)
                        dst = target_media_base / run_name / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        sftp.get(fpath, str(dst))
                    click.echo()
                    media_files_copied += len(file_paths)
        else:
            source_media_base = source_path.parent / "media" / source_project
            if source_media_base.is_dir():
                for run_name in source_runs:
                    src_run_dir = source_media_base / run_name
                    if src_run_dir.is_dir():
                        dst_run_dir = target_media_base / run_name
                        shutil.copytree(src_run_dir, dst_run_dir, dirs_exist_ok=True)
                        media_files_copied += sum(1 for f in src_run_dir.rglob("*") if f.is_file())

    # Summary
    click.echo(f"Merged {len(source_runs)} run(s) into {into_path}:")
    for table, n in sorted(counts.items()):
        click.echo(f"  {table}: {n} rows")
    if media and media_files_copied:
        click.echo(f"  media: {media_files_copied} file(s) copied")


@cli.command()
@click.argument("data_path")
@click.option("--run", "runs", required=True, help="Comma-separated run names to drop")
@click.option("--media/--no-media", default=True, help="Also delete media directories (default: --media)")
def drop(data_path: str, runs: str, media: bool):
    """Remove specific runs from a local .db file."""
    target = Path(data_path)
    if target.suffix.lower() != ".db":
        raise click.ClickException(f"Expected a .db file, got: {data_path}")
    if not target.exists():
        raise click.ClickException(f"File not found: {data_path}")

    requested = {r.strip() for r in runs.split(",")}
    existing = get_run_names_from_db(target)
    unknown = requested - existing
    if unknown:
        raise click.ClickException(f"Run(s) not found in database: {', '.join(sorted(unknown))}")

    with contextlib.closing(sqlite3.connect(target)) as con:
        placeholders = ",".join("?" for _ in requested)
        params = sorted(requested)
        counts: dict[str, int] = {}
        for table in ("metrics", "configs", "system_metrics"):
            if _table_exists(con, table):
                cur = con.execute(
                    f"DELETE FROM {table} WHERE run_name IN ({placeholders})",
                    params,  # noqa: S608
                )
                if cur.rowcount:
                    counts[table] = cur.rowcount
        con.commit()

    # Delete media directories
    media_files_deleted = 0
    media_runs_deleted = 0
    if media:
        project = target.stem
        media_base = target.parent / "media" / project
        if media_base.is_dir():
            for run_name in requested:
                run_dir = media_base / run_name
                if run_dir.is_dir():
                    media_files_deleted += sum(1 for f in run_dir.rglob("*") if f.is_file())
                    media_runs_deleted += 1
                    shutil.rmtree(run_dir)

    # Summary
    click.echo(f"Dropped {len(requested)} run(s) from {data_path}:")
    for table, n in sorted(counts.items()):
        click.echo(f"  {table}: {n} rows deleted")
    if media and media_files_deleted:
        click.echo(f"  media: {media_files_deleted} file(s) deleted in {media_runs_deleted} run(s)")


if __name__ == "__main__":
    cli()
