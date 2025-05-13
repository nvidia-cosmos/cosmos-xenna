import subprocess
import shutil
import os
import tempfile
from datetime import datetime

# Configuration
SOURCE_REPO_URL = "ssh://git@gitlab-master.nvidia.com:12051/dir/imaginaire4.git"
SOURCE_PATH_TO_COPY = "packages/cosmos-xenna"

# Add patterns here (relative to the destination directory, e.g., 'cosmos-xenna/') 
# for files/folders that should NOT be deleted from the destination by rsync's --delete.
PRESERVE_PATTERNS_IN_DESTINATION = [
    'sync_from_i4.py',
    '.gitlab-ci.yml',
]

def run_command(command, cwd=None):
    """Runs a shell command, prints its output. Halts on error with a stack trace."""
    print(f"Executing: {' '.join(command)}")
    subprocess.check_call(command, text=True, cwd=cwd)

def main():
    # Get the root directory of the current Git repository
    try:
        repo_root_result = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True)
        repo_root = repo_root_result.stdout.strip()
        print(f"Current repository root: {repo_root}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Not a git repository or git command not found. {e}")
        # This initial check is critical, so we exit if it fails.
        raise

    with tempfile.TemporaryDirectory() as temp_clone_dir:
        print(f"Created temporary directory: {temp_clone_dir}")

        # 1. Clone the source repository (main branch) into the temporary directory
        print(f"Cloning {SOURCE_REPO_URL} into {temp_clone_dir}...")
        run_command(["git", "clone", "--depth", "1", "--branch", "main", SOURCE_REPO_URL, temp_clone_dir])

        # Get the commit SHA from the cloned repository (imaginaire4)
        source_commit_sha_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=temp_clone_dir
        )
        source_commit_sha = source_commit_sha_result.stdout.strip()
        print(f"Source commit SHA from imaginaire4/main: {source_commit_sha}")

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Dynamically set branch name, commit message, and MR title
        new_branch_name = f"sync/i4-{SOURCE_PATH_TO_COPY.replace('packages/','')}-{source_commit_sha}-{timestamp}"
        commit_message = f"Sync {SOURCE_PATH_TO_COPY} from imaginaire4/main at {source_commit_sha}"
        mr_title = f"Sync {SOURCE_PATH_TO_COPY} from imaginaire4/main@{source_commit_sha} ({timestamp})"

        print(f"New branch name will be: {new_branch_name}")
        print(f"Commit message will be: {commit_message}")
        print(f"MR title will be: {mr_title}")

        # 2. Create and switch to a new branch in the current repository
        print(f"Creating and switching to new branch: {new_branch_name} in {repo_root}")
        try:
            run_command(["git", "checkout", "-B", new_branch_name], cwd=repo_root)
        except subprocess.CalledProcessError:
            print(f"Branch {new_branch_name} might already exist. Attempting to check it out.")
            run_command(["git", "checkout", new_branch_name], cwd=repo_root)
        print(f"Switched to branch {new_branch_name}.")

        # 3. Copy the specified directory using rsync, excluding README_INTERNAL.md
        source_rsync_path = os.path.join(temp_clone_dir, SOURCE_PATH_TO_COPY) + os.sep
        target_copy_destination_in_repo = repo_root
        
        if os.path.exists(target_copy_destination_in_repo):
            print(f"Removing existing target directory in repo: {target_copy_destination_in_repo}")
            if os.path.isdir(target_copy_destination_in_repo):
                shutil.rmtree(target_copy_destination_in_repo)
            else:
                os.remove(target_copy_destination_in_repo)
        os.makedirs(target_copy_destination_in_repo, exist_ok=True)

        print(f"Copying files from {source_rsync_path} to {target_copy_destination_in_repo}...")
        rsync_command = [
            "rsync", "-av", "--delete", 
            "--exclude", "README_INTERNAL.md", # Exclude from source
        ]
        # Add exclude patterns for files to preserve in the destination
        for pattern in PRESERVE_PATTERNS_IN_DESTINATION:
            rsync_command.extend(["--exclude", pattern])
        
        rsync_command.extend([source_rsync_path, target_copy_destination_in_repo])

        run_command(rsync_command)

        # 4. Add, commit the changes in the current repository
        print("Adding changes to git...")
        run_command(["git", "add", "."], cwd=repo_root)

        print(f"Committing changes with message: {commit_message}...")
        # Check if there are any changes to commit before attempting to commit
        status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=repo_root)
        if not status_result.stdout.strip():
            print("No changes to commit.")
        else:
            run_command(["git", "commit", "-m", commit_message], cwd=repo_root)

        # 5. Push the new branch and attempt to create an MR
        print(f"Pushing branch {new_branch_name} to origin and attempting MR creation...")
        push_command_with_mr = [
            "git", "push", "-u", "origin", new_branch_name,
            "-o", "merge_request.create",
            "-o", f"merge_request.title='{mr_title}'",
            "-o", "merge_request.target=main" 
        ]
        try:
            run_command(push_command_with_mr, cwd=repo_root)
            print(f"Successfully pushed branch {new_branch_name} and requested MR creation.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to push with MR creation options: {e}")
            print(f"Attempting a simple push for branch {new_branch_name} without MR creation options...")
            run_command(["git", "push", "-u", "origin", new_branch_name], cwd=repo_root)
            print(f"Successfully pushed branch {new_branch_name}. Please create MR manually if needed.")

    print("\nScript finished.")
    print(f"Branch '{new_branch_name}' processed with changes from '{SOURCE_PATH_TO_COPY}'.")
    print("Please check GitLab to confirm the Merge Request status.")

if __name__ == "__main__":
    main()
