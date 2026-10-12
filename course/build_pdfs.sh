#!/usr/bin/env bash
# Compile all slide decks, the roadmap handout and the exam (+ answer key) into course/pdf/.
# Needs a TeX Live with beamer/metropolis and the exam class (e.g. apt install texlive-latex-extra latexmk).
# Numbers come from slides/figures/numbers.tex (python write_numbers.py after python make_reference.py).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p pdf build
for f in slides/*.tex; do
  (cd slides && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build "$(basename "$f")" >/dev/null)
  cp "build/$(basename "${f%.tex}").pdf" pdf/
done
(cd exam && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build final_exam.tex >/dev/null)
(cd exam && pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build \
   -jobname=final_exam_solutions "\def\withsolutions{}\input{final_exam}" >/dev/null && \
   pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build \
   -jobname=final_exam_solutions "\def\withsolutions{}\input{final_exam}" >/dev/null)
cp build/final_exam.pdf build/final_exam_solutions.pdf pdf/
ls -1 pdf
