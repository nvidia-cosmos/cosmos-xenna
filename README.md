# Cosmos-xenna

TODO: Fill this in more.

## Introduction

## Installing

```bash
pip install cosmos-xenna[gpu]
```

## Ray cluster requirements

Cosmos-xenna needs a few env vars to be set before starting Ray clusters. These are set by Xenna when we
start clusters locally, but, if using an already existing cluster, they will need to be set in the processes
initializing the cluster.

``` bash
# Needed to Xenna control over setting CUDA env vars. Without this, Ray will overwrite the
# env vars we set.
RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="0"
# Needed to get debug info from as many actors as possible. By default, Ray only allows 10k
# actors to be listed. However, on large clusters, we may have more than 10k actors.
RAY_MAX_LIMIT_FROM_API_SERVER=40000
RAY_MAX_LIMIT_FROM_DATA_SOURCE=40000
```

## Development

### Setup development environment

We use UV for developement. To get started, [install UV](https://docs.astral.sh/uv/#installation), and
run `uv sync` in this directory.

This will create a virtual environment at `.venv` based on the current lock file and will include all of
the dependencies from core, dev, gpu and examples.

### Running commands

Use UV to run all commands. For example. To run the example pipeline, use:

``` bash
uv run examples/simple_vlm_inference.py 
```

This will auto-sync dependencies if needed and execute the command in the uv-managed virtualenv.

### Vscode integration

We provide recommended extensions and default settings for yotta via the .vscode/ folder. With these
settings, vscode should automatically format your code and raise linting/typing issues. Vscode will
try to fix some minor linting issues on save.

### Linting

We use Ruff and PyRight for static analysis. Using the default vscode settings and recomended extensions,
these should auto-run in vscode. They can be run manually with:

``` bash
uv run run_presubmit.py default
```

### Adding dependencies

To add packages to the core dependencies, use `uv add some-package-name`

To add packages to dev use `uv add --dev some-package-name`

To add packages to other groups use `uv add --group some-group some-package-name`
