Two motivation slides for the FathomNet project (first-semester MSc biology), in the
pureminimalistic style of "Deep Learning for Maritime Vision Applications".

  Slide 1  How can a computer tell species apart? - what a CNN is, using one real sea star
           (Mediaster aequalis): the real ResNet-18 feature maps after each block and the
           real probabilities of the classifier (72 % M. aequalis, 23 % Asteroidea, 5 % Hippasteria).
  Slide 2  The FathomNet 2025 challenge: the task, the tree-distance score with a concrete
           example (0 / 3 / 4 / 8 / 12), the public winning models we can analyse, what we
           already measured, and open questions about the challenge itself.

Files
  fathomnet_motivation.pdf     the two compiled slides
  fathomnet_motivation.tex     standalone deck (same colours/boxes as main.tex)
  fathomnet_frames.tex         just the two frames -> \input{fathomnet_frames} in main.tex
  img/                         crops, feature maps (FathomNet / MBARI, CC BY-NC-ND 4.0)
  make_figures.py              regenerates img/ from the course data (run from course/)
  beamer*pureminimalistic.sty  theme files (from the original zip)

Build: pdflatex fathomnet_motivation.tex
To use in main.tex: copy img/ next to main.tex, add the \newtcolorbox{ideabox} definition and
\usetikzlibrary{calc, decorations.pathreplacing} to the preamble, then \input{fathomnet_frames}.

Sources of the numbers: the ~98 % -> 57-61 % drop, "almost half of the mistakes in another phylum"
and the model table come from the TorbenGl/underwaterclassification project (README, LOGBOOK);
the leaderboard scores 1.45 / 1.83 are from the model cards as recorded there. Distances use the
7-rank tree (kingdom ... species), as in the competition rules.
