#!/usr/bin/env bash
set -e

echo "Checking working tree..."
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: You have uncommitted changes. Commit or stash them first."
  exit 1
fi

echo "Fetching remotes..."
git fetch origin
git fetch upstream

echo "Updating local main from upstream/main..."
git switch main
git merge --ff-only upstream/main

echo "Pushing main to your fork..."
git push origin main

echo "Rebasing custom-kernels on updated main..."
git switch custom-kernels
git rebase main

echo "Pushing rebased custom-kernels to your fork..."
git push --force-with-lease origin custom-kernels

echo "Done."
git status --short --branch