#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../docs/research/reports"
latexmk -pdf -interaction=nonstopmode -halt-on-error csa_mini_paper_20260524.tex
