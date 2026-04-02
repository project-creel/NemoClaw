#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sync this fork with NVIDIA/NemoClaw upstream.
#
# Usage:
#   ./scripts/sync-upstream.sh          # sync main branch
#   ./scripts/sync-upstream.sh v1.2.0   # sync a specific tag
#
# Prerequisites:
#   git remote add upstream git@github.com:NVIDIA/NemoClaw.git
#
# This script fetches upstream, fast-forward merges into the local main
# branch, and pushes to origin. If the merge is not a fast-forward (i.e.
# we have commits on main that upstream doesn't), it stops and asks you
# to rebase or merge manually.
#
# After syncing, review the diff against our hosted-claw branch:
#   git log hosted-claw..main --oneline
#
# When our hosted-claw changes land upstream, this fork's CI can be
# dropped in favor of NVIDIA's published images.

set -euo pipefail

TARGET="${1:-main}"

if ! git remote get-url upstream &>/dev/null; then
  echo "Error: 'upstream' remote not configured." >&2
  echo "Run: git remote add upstream git@github.com:NVIDIA/NemoClaw.git" >&2
  exit 1
fi

echo "Fetching upstream..."
git fetch upstream

if [[ "$TARGET" == main ]]; then
  CURRENT_BRANCH=$(git branch --show-current)
  if [[ "$CURRENT_BRANCH" != "main" ]]; then
    echo "Switching to main..."
    git checkout main
  fi

  echo "Merging upstream/main (fast-forward only)..."
  if ! git merge --ff-only upstream/main; then
    echo "" >&2
    echo "Fast-forward merge failed — main has diverged from upstream." >&2
    echo "Options:" >&2
    echo "  git rebase upstream/main   # rebase our commits on top" >&2
    echo "  git merge upstream/main    # create a merge commit" >&2
    exit 1
  fi

  echo "Pushing main to origin..."
  git push origin main

  if [[ "$CURRENT_BRANCH" != "main" ]]; then
    echo "Switching back to $CURRENT_BRANCH..."
    git checkout "$CURRENT_BRANCH"
  fi
else
  echo "Fetching tag $TARGET..."
  git fetch upstream tag "$TARGET"
  echo "Tag $TARGET fetched. To inspect: git log $TARGET --oneline -10"
fi

echo "Done."
