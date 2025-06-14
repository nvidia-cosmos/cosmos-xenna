# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer()

HEADER_TEMPLATE = """# SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""  # noqa: E501


def _get_current_year() -> int:
    return datetime.datetime.now().year


def _get_expected_header() -> str:
    return HEADER_TEMPLATE.format(year=_get_current_year())


def _find_python_files() -> list[Path]:
    python_files = []
    try:
        # -z uses null bytes as delimiters, robust for filenames with spaces etc.
        # We specify '.' to ensure it operates on the current directory downwards.
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "."],
            capture_output=True,
            text=False,  # Get bytes for stdout
            check=True,
            cwd=Path(".").resolve(),  # Ensure it runs in the context of the current script's dir or specified target
        )
        # Decode assuming UTF-8, split by null character, filter out empty strings
        files = [f for f in result.stdout.decode("utf-8").split("\0") if f]
        for file_path_str in files:
            if file_path_str.endswith(".py"):
                # git ls-files returns paths relative to the repo root if run from repo root,
                # or relative to cwd if cwd is a subdirectory. Since we run with '.', paths
                # will be relative to the current directory.
                python_files.append(Path(file_path_str))
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error running git ls-files: {e}", err=True)
        sys.exit(1)
    except FileNotFoundError:  # git command not found
        typer.echo("Error: 'git' command not found. Is git installed and in PATH?", err=True)
        sys.exit(1)
    return python_files


def _check_file_header(file_path: Path, expected_header: str) -> bool:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            header_lines = [next(f) for _ in range(len(expected_header.splitlines()))]
            actual_header = "".join(header_lines)
            return actual_header.strip() == expected_header.strip()
    except StopIteration:  # File is shorter than header
        return False
    except FileNotFoundError:
        typer.echo(f"File not found: {file_path}", err=True)
        return False


def _update_file_header(file_path: Path, expected_header: str) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    shebang = ""
    content_starts_at = 0
    if lines and lines[0].startswith("#!"):
        shebang = lines[0]
        content_starts_at = 1

    # Find the end of the existing header (contiguous lines starting with '#')
    header_end_index = content_starts_at
    for i in range(content_starts_at, len(lines)):
        if lines[i].startswith("#"):
            header_end_index = i + 1
        else:
            # First non-comment line marks the end of any potential header block
            break

    # The actual content of the file, after any shebang and existing header
    remaining_lines = lines[header_end_index:]

    new_content_parts = []
    if shebang:
        new_content_parts.append(shebang)
    new_content_parts.append(expected_header)
    if remaining_lines:
        new_content_parts.append("\n")
    new_content_parts.extend(remaining_lines)

    new_content = "".join(new_content_parts).strip()

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    typer.echo(f"Updated header for: {file_path}")


def _check_headers() -> None:
    """Check if all Python files have the correct license header."""
    expected_header = _get_expected_header()
    python_files = _find_python_files()
    all_ok = True
    if not python_files:
        typer.echo("No Python files found.")
        raise typer.Exit(code=0)

    for file_path in python_files:
        if not _check_file_header(file_path, expected_header):
            typer.echo(f"Header missing or incorrect in: {file_path}", err=True)
            all_ok = False

    if all_ok:
        typer.echo("All Python files have correct headers.")
    else:
        raise typer.Exit(code=1)


def _fix_headers() -> None:
    """Update or add license headers to all Python files."""
    expected_header = _get_expected_header()
    python_files = _find_python_files()

    if not python_files:
        typer.echo("No Python files found.")
        raise typer.Exit(code=0)

    for file_path in python_files:
        if not _check_file_header(file_path, expected_header):
            _update_file_header(file_path, expected_header)
    typer.echo("Header update process complete.")


def _run_command(command: str) -> None:
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"Command failed: {e}", err=True)
        raise typer.Exit(code=1)  # noqa: B904


def _lint(fix: bool = True) -> None:
    if fix:
        _fix_headers()
    _check_headers()
    if fix:
        _run_command("ruff check --fix-only .")
    _run_command("ruff format")
    if fix:
        _run_command("ruff check --fix-only .")
    _run_command("pyright .")
    _run_command("ruff format --check .")
    _run_command("ruff check --preview .")


def _unit_test() -> None:
    _run_command("pytest")


@app.command()
def lint(fix: bool = True) -> None:
    """Run linting tasks"""
    _lint(fix)


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
