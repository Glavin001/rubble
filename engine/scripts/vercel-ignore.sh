#!/usr/bin/env bash
# Vercel's Ignored Build Step (vercel.json ignoreCommand): exit 0 skips the build, 1 builds.
# It runs when a queued build starts.
#
# 1. Skip a commit that is no longer the head of its branch. Vercel builds one deployment
#    at a time, so building superseded commits only delays the newest. If the head cannot
#    be read, build.
# 2. Skip when nothing the site is built from changed since the previous deployment.

ref="${VERCEL_GIT_COMMIT_REF:-}"
sha="${VERCEL_GIT_COMMIT_SHA:-}"
if [[ -n "$ref" && -n "$sha" ]]; then
  head="$(timeout 20 git ls-remote origin "refs/heads/$ref" 2>/dev/null | cut -f1)"
  if [[ -n "$head" && "$head" != "$sha" ]]; then
    echo "Skipping ${sha:0:7}: $ref is now at ${head:0:7}"
    exit 0
  fi
fi

prev="${VERCEL_GIT_PREVIOUS_SHA:-HEAD^}"
git cat-file -e "$prev^{commit}" 2>/dev/null || exit 1
git diff --quiet "$prev" HEAD -- engine vercel.json ':(exclude)engine/docs' ':(exclude)engine/stress-oracle' ':(exclude)engine/*.md'
