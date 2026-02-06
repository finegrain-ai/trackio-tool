# trackio-tool

CLI utilities for inspecting and plotting [Trackio](https://github.com/gradio-app/trackio) training data.

Supports both parquet and SQLite (`.db`) files, including files hosted on HuggingFace, [Modal](https://modal.com/) volumes, or SSH servers.

## Requirements

No install needed except [uv](https://docs.astral.sh/uv/). The script is a [uv inline script](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies) and manages its own dependencies.

## Commands

### analyze

Print a summary of all runs in a project: row counts, step ranges, timestamps, tracked metrics, and media file counts.

```bash
# Local parquet file
./trackio-tool.py analyze my-project.parquet

# Local SQLite file
./trackio-tool.py analyze my-project.db

# HuggingFace dataset
./trackio-tool.py analyze hf://my-org/my-dataset/my-project.parquet

# Modal volume
uv run --with modal ./trackio-tool.py analyze modal://my-volume/trackio/my-project.db

# Modal volume with explicit environment
uv run --with modal ./trackio-tool.py analyze modal://my-volume@dev/trackio/my-project.parquet

# SSH server (absolute path)
uv run --with paramiko ./trackio-tool.py analyze ssh://my-server/data/trackio/my-project.db

# SSH server (path relative to home directory)
uv run --with paramiko ./trackio-tool.py analyze ssh://user@my-server/~/trackio/my-project.db
```

Example output:

```
Project: my-project
Total rows: 673
Metrics: loss/mymodel, eval/mymodel_psnr, eval/mymodel_lpips

dainty-meadow-878c
  Rows: 587  Steps: 1 – 33187
  First: 2026-02-05 12:46  Last: 2026-02-05 16:01
  Columns: loss/mymodel, eval/mymodel_psnr, eval/mymodel_lpips
  Media: 12 file(s)
```

Use `--no-media` to skip counting media files (useful for remote sources where listing is slow).

### plot

Plot numeric metrics for one or more data files. Produces a stacked column of charts, one per metric.

```bash
# Single file, all runs
./trackio-tool.py plot my-project.db

# Filter specific runs
./trackio-tool.py plot my-project.db --run dainty-meadow-878c,autumn-willow-d122

# Custom output path (default: plot.webp)
./trackio-tool.py plot my-project.db -o training.png
```

#### Comparing across projects

Pass multiple data files as a comma-separated list to plot runs from different projects together:

```bash
./trackio-tool.py plot project-a.db,project-b.db
```

When run names are unique across projects, they appear as-is in the legend. When the same run name exists in multiple projects, it is automatically qualified as `project:run-name`.

You can filter runs with `--run` using either syntax:

```bash
# Bare run name (must be unambiguous across all loaded projects)
./trackio-tool.py plot a.db,b.db --run calm-river-a3f2

# Qualified name (required when a run name exists in multiple projects)
./trackio-tool.py plot a.db,b.db --run a:calm-river-a3f2,b:bright-dawn-b1c7
```

If a bare run name is ambiguous, the tool errors with a message listing the matching projects.

### merge

Combine runs from one project file into an existing `.db` file. Supports `.db`-to-`.db`, `.parquet`-to-`.db`, HuggingFace, and Modal sources.

```bash
# Merge from another local DB
./trackio-tool.py merge --from project-b.db --into project-a.db

# Merge from a parquet file (imports companion _system.parquet and _configs.parquet too)
./trackio-tool.py merge --from project-b.parquet --into project-a.db

# Merge from a HuggingFace dataset
./trackio-tool.py merge --from hf://my-org/my-dataset/project-b.parquet --into project-a.db

# Merge from a Modal volume
uv run --with modal ./trackio-tool.py merge --from modal://my-volume/trackio/project-b.db --into project-a.db

# Merge from an SSH server
uv run --with paramiko ./trackio-tool.py merge --from ssh://my-server/data/trackio/project-b.db --into project-a.db

# Import only specific runs
./trackio-tool.py merge --from project-b.db --into project-a.db --run calm-river-a3f2,bright-dawn-b1c7

# Skip media file copying
./trackio-tool.py merge --no-media --from project-b.db --into project-a.db
```

The command:

- Errors if any run names overlap between source and target.
- Copies all three tables (`metrics`, `configs`, `system_metrics`), creating missing tables as needed.
- Use `--run` to import only specific runs (comma-separated). By default all runs are imported.
- By default (`--media`) copies media directories from `media/<project>/<run>/` next to the source into the corresponding location next to the target. Use `--no-media` to skip this. The `--run` filter applies to media as well.
- For HF sources, downloads companion parquet files and media from the same dataset repo.
- For Modal sources, downloads files from the volume. Requires `modal` (`uv run --with modal`).
- For SSH sources, downloads files via SFTP. Requires `paramiko` (`uv run --with paramiko`).

### Modal volumes

To store Trackio data on a Modal volume, set the `TRACKIO_DIR` environment variable in your Modal app to a path on the volume (e.g. `/vol/trackio`). Trackio will write its data files there, and you can then access them with:

```bash
uv run --with modal ./trackio-tool.py analyze modal://my-volume/trackio/my-project.db
```

### SSH servers

Access Trackio data on any SSH-accessible machine using `ssh://[user@]host/path` URLs. The path after the host is absolute on the remote machine. Use `/~/` to specify a path relative to the user's home directory. Authentication uses your SSH agent and default keys (same as regular `ssh` usage).

```bash
# Absolute path
uv run --with paramiko ./trackio-tool.py analyze ssh://my-server/data/trackio/my-project.db

# Relative to home directory, e.g. default trackio location
uv run --with paramiko ./trackio-tool.py analyze ssh://user@my-server/~/.cache/huggingface/trackio/my-project.db
```

### drop

Remove specific runs from a local `.db` file.

```bash
# Drop a single run
./trackio-tool.py drop my-project.db --run calm-river-a3f2

# Drop multiple runs
./trackio-tool.py drop my-project.db --run calm-river-a3f2,bright-dawn-b1c7

# Drop runs but keep their media directories
./trackio-tool.py drop my-project.db --run calm-river-a3f2 --no-media
```

The command deletes matching rows from all three tables (`metrics`, `configs`, `system_metrics`) and by default (`--media`) also removes the corresponding `media/<project>/<run>/` directories and reports the number of files deleted. Use `--no-media` to keep media on disk.

## Development

Coding agents are used to assist with the development of this tool (in other words it is mostly vibe-coded).

Linting, formatting and type checking:

```bash
uv sync --all-extras
uv run ruff check
uv run ruff format
uv run pyright
```
