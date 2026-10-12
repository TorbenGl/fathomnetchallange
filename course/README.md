# Can a computer identify deep-sea animals?
**A 6-session classifier project for biology students · FathomNet 2025 · no programming or GPU needed**

Students learn what a classifier is by building and comparing a few of them on 23,698 expert-labelled crops of 79
deep-sea animals, then find out when such a classifier can be trusted. The project ends with a mini-challenge
(poster) and a written final exam.

| You want … | Open |
|---|---|
| the slides for the kick-off, including the **roadmap** | [`pdf/00_kickoff_roadmap.pdf`](pdf/00_kickoff_roadmap.pdf) |
| a 1-page roadmap handout for students | [`pdf/roadmap_handout.pdf`](pdf/roadmap_handout.pdf) |
| the lecture slides for sessions 2–5 | [`pdf/01_…`](pdf/01_what_is_a_classifier.pdf) · [`02_…`](pdf/02_features_and_cnns.pdf) · [`03_…`](pdf/03_testing_honestly.pdf) · [`04_…`](pdf/04_challenge_and_recap.pdf) |
| the final exam / the answer key | [`pdf/final_exam.pdf`](pdf/final_exam.pdf) · [`pdf/final_exam_solutions.pdf`](pdf/final_exam_solutions.pdf) |
| the student notebooks (one per session) | `lab0_data.ipynb` … `lab4_challenge.ipynb` (shipped with executed outputs) |

## The roadmap

| # | Session (slides) | Lab notebook | Students hand in |
|---|---|---|---|
| 1 | Kick-off & roadmap (`00`) | `lab0_data` – set up, download, meet the animals, choose a group | group + labels + one figure; Q1–Q5 |
| 2 | What is a classifier? (`01`) | `lab1_first_classifier` – a threshold by hand, nearest centroid, kNN, a decision tree as identification key, overfitting, baseline | Q answers, their dichotomous key |
| 3 | How a computer sees an animal (`02`) | `lab2_features_and_zoo` – pixels vs. colour characters vs. CNN features; 7 classifiers × 3 feature sets | results table, Q answers |
| 4 | Testing honestly (`03`) | `lab3_honest_evaluation` – metrics, confusion matrix, tree-of-life mistakes, random vs. image vs. **dive** split | confusion matrix, split chart, half page |
| 5 | Mini-challenge (`04`) | `lab4_challenge` – best macro-F1 on unseen dives, validation set, optional CNN fine-tuning | leaderboard entry + poster |
| 6 | Poster session & final exam | – | 90-min written exam (100 points) |

Suggested grading: lab notebooks 30 %, poster 30 %, exam 40 % (in the kick-off slides; change as you like).

**The classifier ladder** students climb: majority baseline → a rule by hand → nearest centroid / kNN / decision
tree → better features (colour statistics, CNN embeddings) → random forest / logistic regression / neural network →
(optional) fine-tuning a whole CNN.

**The story the course tells:** features matter more than the classifier; training accuracy measures memory; and
a classifier that scores well on a random split loses much of it on dives it has never seen – the same effect the
prize-winning FathomNet models show (≈ 98 % on training images, 57–61 % on new dives).

## Reference results (measured in this repository)

Full dataset (23,664 crops after removing duplicate boxes), 79 labels, 75/25 split, measured on 4 CPU cores.
All numbers in the slides and exam come from this run (`results/reference_results.json`).

| Test accuracy, 79 labels, random split | pixels (768) | colour stats (30) | CNN ResNet-18 (512) |
|---|---|---|---|
| majority baseline | 1.3 % | 1.3 % | 1.3 % |
| nearest centroid | 10.7 % | 15.9 % | 37.9 % |
| 5-nearest neighbours | 23.0 % | 33.6 % | 53.4 % |
| decision tree | 15.4 % | 27.5 % | 20.5 % |
| random forest | 37.2 % | 48.4 % | 56.1 % |
| logistic regression | 39.7 % | 32.4 % | 63.7 % |
| neural network (MLP) | 46.3 % | 46.1 % | **64.6 %** |

| CNN features, mean of 3 splits | logistic regression | 5-nearest neighbours |
|---|---|---|
| random split (97 % of test crops from a dive also in training) | 63.5 % | 53.5 % |
| image split | 57.5 % | 45.5 % |
| **dive split** (new expeditions) | **49.6 %** | **35.5 %** |

On the dive split, 55 % of the mistakes of logistic regression are in a different phylum; the phylum is right for
72.6 % of the crops (80.0 % on the random split). Recall falls with the rank of the label: species labels 67 %,
class labels 44 %, the phylum label *Porifera* 27 %. With colour statistics, nearest centroid predicts phyla *worse*
than the majority baseline (26.4 % vs 30.4 %) – a nice discussion point in Lab 1.

## Quick start

**Students on Google Colab:** upload a notebook (*File → Upload notebook*), run the first cell. It clones this
repository, installs `scikit-learn` and `fathomnet`, and downloads the crops (~15 min the first time). If Google
Drive is mounted, a zip of the crops is stored in `MyDrive/fathomnet2025_compact.zip` and restored in seconds next time.
`setup(per_category=100)` instead of `setup()` gives a quick balanced subset (~7,900 crops).

> The repository is private. Either make it (or a copy) public, or give students a zip: from the repository root
> `zip -r fathomnetchallange.zip . -x "fathomnet2025_compact/*" "fgvc-comp-2025/*" "course/features_cache/*" ".git/*"`;
> students drag `fathomnetchallange.zip` into Colab's file panel before running the first cell.
> The first cell clones the default branch; until this branch is merged, use the zip.

**Locally:**
```bash
pip install -r course/requirements.txt jupyter
python prepare_fathomnet2025.py              # once: 23,698 crops, ~15 min, 250 MB (exit code 1 = the one known broken box)
cd course && jupyter lab lab0_data.ipynb
```

**Command line** (same experiments as the notebooks):
```bash
cd course
python run_lab.py --target phylum --features colour_stats --classifiers knn decision_tree
python run_lab.py --features cnn --classifiers all --split random dive --out results/my_runs.csv
python tests/test_lab.py                     # offline self-test, synthetic data
```

## Files

```
course/
├── README.md                this page (tutor guide)
├── lab0_data.ipynb … lab4_challenge.ipynb   student notebooks, executed (generated by build_labs.py)
├── classifier_lab/          small helper library - short, meant to be read by curious students
│   ├── data.py              load crops + taxonomy, dive of each image, targets, random/image/dive splits
│   ├── features.py          pixels, 30 colour/texture characters, ImageNet-CNN embeddings (cached)
│   ├── models.py            the classifier zoo (scikit-learn), with biology analogies
│   ├── evaluate.py          scores, per-label recall, confusion matrix, tree-of-life mistake severity
│   └── finetune.py          optional: fine-tune ResNet-18 end to end (GPU)
├── setup_env.py             first-cell helper: Colab/local, download or restore the dataset
├── run_lab.py               command-line experiments
├── make_reference.py        tutor: reference run -> results/ + slides/figures/
├── write_numbers.py         tutor: results -> slides/figures/numbers.tex (numbers used in slides + exam)
├── build_labs.py            tutor: regenerates the notebooks (edit text there, not in the .ipynb)
├── build_pdfs.sh            tutor: compiles all slides, the handout and the exam into pdf/
├── slides/                  LaTeX (beamer, metropolis) - 5 decks + roadmap handout + shared style
├── exam/final_exam.tex      exam class; one source for exam and answer key
├── pdf/                     the compiled PDFs
├── results/                 reference results (CSV/JSON)
└── tests/test_lab.py        offline self-test
```

## Rebuilding everything

```bash
pip install -r requirements.txt nbformat nbconvert ipykernel
python make_reference.py && python write_numbers.py      # ~25 min on 4 CPU cores (CNN features 13 min, then cached)
python build_labs.py && for f in lab*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$f"; done
./build_pdfs.sh                                          # needs texlive-latex-extra, latexmk
```

## Tutor notes

* **Timing.** CNN features for all crops: 13 min on 4 CPU cores (once; cached in `features_cache/`); a Colab GPU
  should take a few minutes (not measured). Measured run times of the executed notebooks on 4 CPU cores with cached features: Lab 0 and Lab 1 under
  30 s, Lab 2 ≈ 8.5 min (the zoo; random forests are the slowest), Lab 3 ≈ 4.5 min, Lab 4 ≈ 1.5 min.
* **Clean data.** 30 boxes appear more than once on the same image (`../KNOWN_ISSUES.md`). `load_dataset()` keeps one
  copy of the 27 with a consistent label and drops the 3 with *conflicting* labels: 34 rows removed, 23,664 crops remain.
* **Dives.** "Dive" = the folder of the image URL (e.g. `Doc Ricketts/images/0621`): 1,465 dives; the 10 largest
  hold 21 % of all crops. The dive split uses scikit-learn's `StratifiedGroupKFold`, so label proportions stay similar.
* **Exam.** 6 questions, 100 points; computations (Q2, Q3, Q5) use small invented numbers, Q4 uses the reference
  run. The answer key is the same source compiled with `\def\withsolutions{}` (see `build_pdfs.sh`).
* **Relation to the CNN-taxonomy project** (`TorbenGl/underwaterclassification`): that project analyses the
  *published winning models* on a held-out set. This course is the gentler entry point; deck 04 points to it as a follow-up.

## Not tested / limitations

* **Google Colab and GPUs were not available** while building this. The Colab branch of the first cell and
  `setup_env.py` (Drive zip) are untested; please do a 10-minute dry run on Colab before the first session.
* **`finetune.py` (Lab 4, Level 5)** was only smoke-tested on CPU (3 labels, 2 epochs, 112 px: runs, 96 % test accuracy on 23 crops). A full GPU run has not been done; times in the notebook are estimates.
* The notebooks were executed end to end on the full dataset on CPU (see outputs in the `.ipynb` files).
* Labels are FathomNet expert labels; some "mistakes" are label errors. Images: CC BY-NC-ND 4.0, annotations
  CC BY-ND 4.0 – acknowledge FathomNet and MBARI.
