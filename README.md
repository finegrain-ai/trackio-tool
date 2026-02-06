# trackio-utils

CLI utilities for inspecting and plotting [Trackio](https://github.com/gradio-app/trackio) training data.

Supports both parquet and SQLite (`.db`) files, including files hosted on HuggingFace.

## Requirements

No install needed except [uv](https://docs.astral.sh/uv/). The script is a [uv inline script](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies) and manages its own dependencies.

## Commands

### analyze

Print a summary of all runs in a project: row counts, step ranges, timestamps, and tracked metrics.

```bash
# Local parquet file
./trackio-tool.py analyze my-project.parquet

# Local SQLite file
./trackio-tool.py analyze my-project.db

# HuggingFace dataset
./trackio-tool.py analyze hf://my-org/my-dataset/my-project.parquet
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
```

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

## Development

Coding agents are used to assist with the development of this tool.

Linting, formatting and type checking:

```bash
uv sync --all-extras
uv run ruff check
uv run ruff format
uv run pyright
```
