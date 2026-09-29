#!/usr/bin/env bash
# Copies the audit code and its results into the release worktree.
# (The sandbox that produced them cannot write under .claude/.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(dirname "$HERE")/.claude/worktrees/release-hardening"
DST="$WT/experiments/adversarial_review_audit"
mkdir -p "$DST/results" "$DST/sagemaker"
cp "$HERE"/experiments/adversarial_review_audit/{README.md,audit.py,classify.py,analyze.py,test_classify.py} "$DST/"
cp "$HERE"/sagemaker/{run_job.sh,driver.py,verify_digests.py} "$DST/sagemaker/"
cp -R "$HERE/final/aws" "$DST/results/aws"
cp -R "$HERE/final/mac_pilot" "$DST/results/mac_pilot"
mkdir -p "$DST/results/aws/env"
cp "$HERE"/aws_results/*/env_*.json "$DST/results/aws/env/" 2>/dev/null || true
cp "$HERE/RESULTS.md" "$DST/results/RESULTS.md"
echo "installed into $DST"
