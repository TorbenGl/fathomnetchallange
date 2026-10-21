Two motivation slides for the FathomNet project, in the pureminimalistic style of
"Deep Learning for Maritime Vision Applications".

  fathomnet_motivation.pdf   the two compiled slides
  fathomnet_motivation.tex   standalone deck (preamble = same colours/boxes as main.tex)
  fathomnet_frames.tex       just the two frames -> \input{fathomnet_frames} in main.tex
  img/animal1..8.jpg         FathomNet crops (FathomNet / MBARI, CC BY-NC-ND 4.0)
  beamer*pureminimalistic.sty  theme files (from the original zip)

Build: pdflatex fathomnet_motivation.tex
To use in main.tex: copy img/ next to main.tex, add the \newtcolorbox{ideabox} definition
from fathomnet_motivation.tex to the preamble, then \input{fathomnet_frames}.
