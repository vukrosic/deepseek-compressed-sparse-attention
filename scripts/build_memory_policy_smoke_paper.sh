#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../docs/research/reports"
latexmk -pdf -interaction=nonstopmode -halt-on-error memory_policy_smoke_20260525.tex
