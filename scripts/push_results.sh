#!/usr/bin/env bash
# push_results.sh — push code + results + figures to GitHub from your MacBook
# ============================================================================
# Usage:
#   ./push_results.sh                 # commit everything + push
#   ./push_results.sh "my message"    # custom commit message
#
# It will:
#   1. Make sure you are on the working branch
#   2. Stage all changes (code, results/, *.png, *.csv)
#   3. Commit (skips if nothing changed)
#   4. Push with up to 4 retries on network errors (2s,4s,8s,16s backoff)
# ============================================================================
set -u

BRANCH="claude/general-session-gviGa"
MSG="${1:-Update results, figures, and analysis}"

# --- ensure we are in a git repo ---
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: not inside a git repository. cd into ~/AL-ML first."
  exit 1
fi

# --- switch to / create the working branch ---
current=$(git rev-parse --abbrev-ref HEAD)
if [ "$current" != "$BRANCH" ]; then
  echo "Switching from '$current' to '$BRANCH'..."
  git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH"
fi

# --- stage everything ---
git add -A
git add -f results/ 2>/dev/null || true   # in case results/ is gitignored

# --- commit (skip cleanly if nothing to commit) ---
if git diff --cached --quiet; then
  echo "No new changes to commit."
else
  git commit -m "$MSG"
  echo "Committed: $MSG"
fi

# --- push with exponential backoff ---
delays=(2 4 8 16)
attempt=0
until git push -u origin "$BRANCH"; do
  if [ "$attempt" -ge "${#delays[@]}" ]; then
    echo "ERROR: push failed after $((attempt)) retries."
    echo "If this is an auth error (403), check your GitHub credentials / token."
    exit 1
  fi
  d=${delays[$attempt]}
  echo "Push failed — retrying in ${d}s..."
  sleep "$d"
  attempt=$((attempt + 1))
done

echo "Done. Pushed '$BRANCH' to origin."
