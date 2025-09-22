# Cosmos-xenna

## Introduction

Cosmos-xenna is a Python library for building and running distributed data pipelines using Ray. It
has a heavy focus on pipelines which are a series of inference steps using AI models. For example, a
pipeline which downloads an image, runs a VLM on it to produce a caption, and then runs an embedding model
to produce a text embedding and uploads the resulting data.

Cosmos-xenna simplifies the development of distributed AI pipelines by providing:

- A simple interface
- Autoscaling/autobalancing of stages
- Stateful actors which allow the user to load/download weights before running processing
- Independent allocation of NVDEC/NVENC hardware and "main" GPU compute

## Installing

```bash
pip install cosmos-xenna[gpu]
```

## Quick Start

For detailed examples, check out the `examples/` directory.

## Ray cluster requirements

Cosmos-xenna needs a few environment variables to be set before starting Ray clusters. These are set by Xenna when we
start clusters locally, but if using an already existing cluster, they will need to be set in the processes
initializing the cluster.

```bash
# Needed to give Xenna control over setting CUDA environment variables. Without this, Ray will overwrite the
# environment variables we set.
RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="0"
# Needed to get debug info from as many actors as possible. By default, Ray only allows 10k
# actors to be listed. However, on large clusters, we may have more than 10k actors.
RAY_MAX_LIMIT_FROM_API_SERVER=40000
RAY_MAX_LIMIT_FROM_DATA_SOURCE=40000
```

## Development

### Setup development environment

We use UV for development. To get started, [install UV](https://docs.astral.sh/uv/#installation), and
run `uv sync` in this directory.

This will create a virtual environment at `.venv` based on the current lock file and will include all
of the dependencies from core, dev, GPU, and examples.

### Running commands

Use UV to run all commands. For example, to run the example pipeline, use:

```bash
uv run examples/simple_vlm_inference.py 
```

This will auto-sync dependencies if needed and execute the command in the UV-managed virtualenv.

### VSCode integration

We provide recommended extensions and default settings for yotta via the .vscode/ folder. With these
settings, VSCode should automatically format your code and raise linting/typing issues. VSCode will
try to fix some minor linting issues on save.

### Linting

We use Ruff and PyRight for static analysis. Using the default VSCode settings and recommended extensions,
these should auto-run in VSCode. They can be run manually with:

```bash
./run_presubmit.py default
```

### Adding dependencies

To add packages to the core dependencies, use `uv add some-package-name`

To add packages to dev use `uv add --dev some-package-name`

To add packages to other groups use `uv add --group some-group some-package-name`

## License and Contact

This project will download and install additional third-party open source software projects. Review the
license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).
