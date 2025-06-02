# Internal Xenna readme

## Introduction

Today, cosmos-xenna lives in three places:

1. As a self-contained [sub-package in imaginaire4](https://gitlab-master.nvidia.com/dir/imaginaire4/-/tree/main/packages/cosmos-xenna?ref_type=heads).
2. This [internal gitlab repo](https://gitlab-master.nvidia.com/dir/nvidia-cosmos/cosmos-xenna)
3. A soon to be public [repo on github](https://github.com/nvidia-cosmos/cosmos-xenna)

We do this because xenna has a very strong coupling to i4 pipeline code.

## Syncing code

We have written script for syncing code around the various repos.

### Sync from i4 to gitlab

1. Checkout and pull main
2. Run the syncing script `uv run sync_from_i4.py`
3. This will create an MR
  [in the repo](https://gitlab-master.nvidia.com/dir/nvidia-cosmos/cosmos-xenna/-/merge_requests). Get it merged.

### Sync from gitlab to github

1. Checkout and pull main
2. Run the syncing script `uv run sync_to_github.py`
3. This will create a Pull request in the [github repo](https://github.com/nvidia-cosmos/cosmos-xenna/pulls).
   Get it merged.
