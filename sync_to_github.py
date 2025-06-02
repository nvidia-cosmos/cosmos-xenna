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

import os
import subprocess
import tempfile
from datetime import datetime
from typing import Optional

# Configuration
GITHUB_REPO = "nvidia-cosmos/cosmos-xenna"

# Add patterns here for files/folders that should be ignored during sync
IGNORE_PATTERNS = [
    ".git",
    ".venv",
    "sync_from_i4.py",
    "sync_to_github.py",
    "INTERNAL_README.md",
]


def run_command(command: list[str], cwd: Optional[str] = None) -> None:
    """Runs a shell command, prints its output. Halts on error with a stack trace."""
    print(f"Executing: {' '.join(command)}")
    subprocess.check_call(command, text=True, cwd=cwd)


def main() -> None:
    # Get the root directory of the current Git repository
    try:
        repo_root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
        )
        repo_root = repo_root_result.stdout.strip()
        print(f"Current repository root: {repo_root}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Not a git repository or git command not found. {e}")
        raise

    with tempfile.TemporaryDirectory() as temp_clone_dir:
        print(f"Created temporary directory: {temp_clone_dir}")

        # Debug: Check authentication and repository access
        print("Checking GitHub authentication...")
        try:
            auth_result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, check=True)
            print("Authentication status:")
            print(auth_result.stdout)
        except subprocess.CalledProcessError as e:
            print("Authentication check failed:")
            print(e.stderr)
            raise

        print(f"\nChecking repository access for {GITHUB_REPO}...")
        try:
            repo_result = subprocess.run(
                ["gh", "repo", "view", GITHUB_REPO], capture_output=True, text=True, check=True
            )
            print("Repository info:")
            print(repo_result.stdout)
        except subprocess.CalledProcessError as e:
            print("Repository check failed:")
            print(e.stderr)
            raise

        # 1. Clone the GitHub repository into the temporary directory using gh
        print(f"\nCloning {GITHUB_REPO} into {temp_clone_dir}...")
        run_command(["gh", "repo", "clone", GITHUB_REPO, temp_clone_dir, "--", "--depth", "1", "--branch", "main"])

        # Get the commit SHA from the current repository
        current_commit_sha_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, cwd=repo_root
        )
        current_commit_sha = current_commit_sha_result.stdout.strip()
        print(f"Current commit SHA: {current_commit_sha}")

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Set branch name and commit message
        new_branch_name = f"sync/github-{current_commit_sha}-{timestamp}"
        commit_message = f"Sync to GitHub from commit {current_commit_sha}"

        print(f"New branch name will be: {new_branch_name}")
        print(f"Commit message will be: {commit_message}")

        # 2. Create and switch to a new branch in the GitHub repository
        print(f"Creating and switching to new branch: {new_branch_name}")
        try:
            run_command(["git", "checkout", "-B", new_branch_name], cwd=temp_clone_dir)
        except subprocess.CalledProcessError:
            print(f"Branch {new_branch_name} might already exist. Attempting to check it out.")
            run_command(["git", "checkout", new_branch_name], cwd=temp_clone_dir)

        # 3. Copy files using rsync, excluding specified patterns
        print(f"Copying files from {repo_root} to {temp_clone_dir}...")

        # First, get list of files that git tracks
        git_files_result = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True, cwd=repo_root
        )
        git_files = git_files_result.stdout.strip().split("\n")

        # Create a temporary file with the list of files to copy
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as f:
            for file in git_files:
                if not any(file.startswith(pattern) for pattern in IGNORE_PATTERNS):
                    f.write(f"{file}\n")
            files_list_path = f.name

        # Use rsync with the files list
        rsync_command = ["rsync", "-av", "--files-from", files_list_path, "--delete", f"{repo_root}/", temp_clone_dir]

        try:
            run_command(rsync_command)
        finally:
            # Clean up the temporary file
            os.unlink(files_list_path)

        # 4. Add and commit the changes
        print("Adding changes to git...")
        run_command(["git", "add", "."], cwd=temp_clone_dir)

        print(f"Committing changes with message: {commit_message}...")
        status_result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=temp_clone_dir, check=False
        )
        if not status_result.stdout.strip():
            print("No changes to commit.")
        else:
            run_command(["git", "commit", "-m", commit_message], cwd=temp_clone_dir)

        # 5. Push the changes to GitHub
        print(f"Pushing branch {new_branch_name} to GitHub...")
        try:
            run_command(["git", "push", "-u", "origin", new_branch_name], cwd=temp_clone_dir)
            print(f"Successfully pushed branch {new_branch_name} to GitHub.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to push to GitHub: {e}")
            print("Please ensure you have the necessary permissions and try again.")

    print("\nScript finished.")
    print(f"Branch '{new_branch_name}' has been pushed to GitHub.")
    print("Please create a Pull Request manually on GitHub if needed.")


if __name__ == "__main__":
    main()
