import subprocess

import typer

app = typer.Typer(help="Development tasks CLI")


def _run_command(command: str) -> None:
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"Command failed: {e}", err=True)
        raise typer.Exit(code=1)  # noqa: B904


def _lint() -> None:
    _run_command("ruff check --fix-only .")
    _run_command("ruff format")
    _run_command("ruff check --fix-only .")
    _run_command("pyright .")
    _run_command("ruff format --check .")
    _run_command("ruff check --preview .")


def _unit_test() -> None:
    _run_command("pytest")


@app.command()
def lint() -> None:
    """Run linting tasks"""
    _lint()


@app.command()
def unit_test() -> None:
    """Run tests"""
    _unit_test()


@app.command()
def test() -> None:
    """Run tests"""
    _unit_test()


@app.command()
def all() -> None:
    """Run all tasks"""
    _lint()
    _unit_test()


@app.command()
def default() -> None:
    """Development tasks CLI"""
    _lint()
    _unit_test()


if __name__ == "__main__":
    app()
