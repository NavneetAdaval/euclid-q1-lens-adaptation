# What a target-domain training signal buys a sim-to-real lens finder on Euclid Q1, from a balanced to a 1:10 target — code, stored results and identifier lists

This repository holds the code, the stored test scores, the pre-registrations and the
identifier lists behind the paper *What a target-domain training signal buys a sim-to-real lens finder on Euclid Q1, from a balanced to a 1:10 target*, by Navneet Kumar Adaval
(arXiv: forthcoming). The repository is at https://github.com/NavneetAdaval/euclid-q1-lens-adaptation; it is archived on
Zenodo at doi:[10.5281/zenodo.24285258](https://doi.org/10.5281/zenodo.24285258).

Every table, number macro and figure in the paper regenerates from this repository with one
command, except Figure 2 and the lens gallery, which show Euclid cutouts. The repository
contains **no Euclid pixels**: no cutouts, image shards, simulated lens images, feature caches
or model weights.

Licence: **GPL-3.0** (`LICENSE`). The Euclid data and the Discovery Engine catalogues keep their
own licences (see *Data*).

## Layout

The layout is the project's own, unchanged, because the hashed pre-registrations and the
integrity inventory record paths of the form `runs/…`, `scripts/…` and `data/splits/…`.

| Path | What it holds |
|---|---|
| `scripts/` | all project code: catalogue query and cutout download, splits, the arc-on-real simulator (`generate_arc_on_real.py`), the domain-adaptation losses and runners, the Kaggle bundle builders, the pre-registration writers, the analyses, `integrity.py`, and the tests (`test_*.py`) |
| `paper/` | the paper source (`main.tex`, `sections/`, `refs.bib`, `plainnat_doi_eprint.bst`), its generated `tables/`, `figures/` and `data/`, and the generators in `paper/scripts/` |
| `runs/` | stored per-object test scores (`logits/step_*.npz`, `scores/*.npz`, each with its `ids.npz`), run records (`run.json`, `run_config.json`, `results.json`, curve CSVs), and the hashed pre-registration and budget files |
| `kaggle/coral_r4/runs/` | the CORAL × nb3 run records and scores |
| `data/splits/` | `splits.json` (the split definitions), `dilute10_pool.json` (the 1:10 target pool) and `d1_test_neg_remove.json` (the negatives removed from the test pool) |
| `data/ids/` | readable identifier lists: `lens_positive_ids.csv` (the 489 lens positives) and `deflector_ids.csv` (the 19,167-galaxy deflector sample), exact Euclid Q1 `object_id`s |
| `backup/rewrite_v1_20261010_182859/` | three pre-rewrite copies of paper files that the review-round hash check compares against (`make_review_macros.py`) |
| `public_inputs/` | pixel-free extracts of the inputs that are not redistributed in full (`MANIFEST.json` lists source, licence, rows and sha256 of each); the lens-positive identifiers are in `lens_ids.npz` and the deflector sample in `deflectors.npz` |
| `audit/` | the test-pool exclusion rule (`audit/decontam/rule.json`) and the post hoc analyses (`audit/review_r1/`, `audit/review_r1b/`, `audit/rewrite_v5/`) |
| `evidence/`, `kaggle_prefold/` | the pre-fold copies of the DANN pre-registration and budget |
| `public_inputs/record_extract.json` | everything the build needs from the author's lab log, which is not distributed: the protected digests, the numbers with their log line, and verdicts (written by `scripts/write_record_extract.py`, read through `scripts/record_extract.py`) |
| `KAGGLE_*.md` | the notebook cells each Kaggle run was launched with |

## Rebuilding every table, figure and number

From the repository root, with Python 3 and the packages in `requirements-build.txt`
(numpy, pandas, scipy, scikit-learn, matplotlib, and a CPU build of torch/torchvision, which
two of the imported modules need):

```bash
python scripts/integrity.py --verify
python paper/scripts/make_all.py
```

`integrity.py --verify` checks every protected pre-registration, budget and pool file
against the digest recorded in `public_inputs/record_extract.json`. `make_all.py` regenerates every table, number
macro and figure, then runs `paper/scripts/crosscheck.py`, which re-derives each number
independently (2822 checks) and reports any mismatch. Nothing trains and no GPU is needed.
The inputs that are not redistributed in full are read from `public_inputs/`; the two tables
that read a checkpoint (`make_a5_restart.py`, `make_m_batch.py`) and the figures that need
cutouts (`fig_f2_cutouts.py`, `fig_fy_lens_gallery.py`) keep their committed outputs.
Figure PDFs carry a fixed creation date (`SOURCE_DATE_EPOCH=0`), so a rebuild is
byte-identical.

To compile the paper: `cd paper && tectonic -X compile main.tex` (or pdfLaTeX + BibTeX).

## Inputs that are withheld, and how to check your own copies

The post hoc review analyses (`audit/review_r1/`, `audit/review_r1b/`) record the sha256 of every
file they read, and `paper/scripts/make_review_macros.py` re-checks those hashes before it uses
any of their results. 156 of those inputs are not in this repository, for licence or size
reasons, and are listed in `paper/data/not_redistributed.json`, each with the hash its review
file recorded:

| Category | Files | What they are |
|---|---:|---|
| `euclid_pixels` | 85 | Euclid Q1 cutout shards of the unlabelled pool, and the Discovery Engine lens cutouts |
| `simulated_pixels` | 60 | the simulated lenses, painted onto Euclid cutouts |
| `deflector_pixels` | 1 | the Euclid cutouts of the deflector sample |
| `catalogue_csv` | 8 | the stored MER query output and the Discovery Engine catalogues and lens-model tables |
| `superseded_draft` | 2 | an earlier arXiv draft archive and an earlier release report |

The check is explicit, not silent: a present input must match its recorded hash, and an absent one
must be listed in `not_redistributed.json` with that same hash, or the build stops. It prints
`N inputs verified, M not redistributed`.

**To obtain the files:** the Euclid Q1 cutouts come from the ESA Euclid Science Archive
(`scripts/01_query_catalog.py`, `scripts/02_download_cutouts.py`); the Discovery Engine catalogue
and lens cutouts from its publication's Zenodo records (v0.0.3, doi:10.5281/zenodo.15025832;
v0.0.1, doi:10.5281/zenodo.15003117; `scripts/build_lens_cutouts.py`); and the simulated shards
are regenerated by `scripts/generate_arc_on_real.py` from its fixed seed. Byte-identical
regeneration of the simulated shards is not guaranteed (the GalSim version and worker count were
not recorded).

**To verify your copies,** lay them out like this repository and run

```bash
python scripts/verify_withheld_inputs.py PATH/TO/YOUR/COPY
```

It reports each entry as `match`, `differs` or `missing` and exits 0 only if all match. Files that
match can be copied in at the same paths, and the build then verifies them instead of accepting
them as withheld.

## Identifier lists

| File | Rows | sha256 |
|---|---:|---|
| `data/ids/lens_positive_ids.csv` | 489 | `a85e36167ef2a7d15642aefe5f83e7de4798732531269f5f035befe64fc6bb66` |
| `data/ids/deflector_ids.csv` | 19,167 | `7dc0e9cb0dbb37be6551b5d05ba59f59e6c23288566360c9d3f0c7fde5f38b66` |

Both hold one exact Euclid Q1 `object_id` per line (header `euclid_object_id`), written from
`public_inputs/lens_ids.npz` and `public_inputs/deflectors.npz`; every ID equals the rebuild from
its own RA/Dec, `id = sign(Dec) × (floor(RA·10⁷)·10⁹ + floor(|Dec|·10⁷))`.

## What needs the Euclid data and a GPU

Training, simulation and the cutout figures need the images, which are not redistributed:

- **Negatives:** Euclid Q1 MER catalogue and mosaics, from the ESA Euclid Science Archive
  (`scripts/01_query_catalog.py`, `scripts/02_download_cutouts.py`).
- **Lens positives:** the Discovery Engine cutout archives, v0.0.3,
  doi:[10.5281/zenodo.15025832](https://doi.org/10.5281/zenodo.15025832)
  (`scripts/build_lens_cutouts.py`).
- **Grade-X hard negatives:** identified through Discovery Engine catalogue v0.0.1,
  doi:[10.5281/zenodo.15003117](https://doi.org/10.5281/zenodo.15003117), cut from the Q1
  mosaics (`scripts/07_download_ml_fps.py`).
- **Simulated lenses:** `scripts/generate_arc_on_real.py` paints arcs onto the deflector sample
  (needs GalSim, its COSMOS catalogue and lenstronomy). Its seed is fixed, but an exact
  regeneration is not guaranteed: the GalSim version and the number of parallel workers were
  not recorded.
- **Training:** the runners in `scripts/` (`run_*.py`) need a CUDA GPU; most runs used a
  Kaggle T4, launched with the cells in `KAGGLE_*.md`. The encoder is
  `hf_hub:mwalmsley/zoobot-encoder-greyscale-convnext_nano`, loaded through `timm`.

**Library versions used in training were not recorded** (the paper's Appendix J says which
versions are known). `requirements-build.txt` pins the environment that rebuilds the paper's
numbers, not the training environment.

## Data

- **Euclid Q1:** European Space Agency, Euclid Q1 data,
  doi:[10.57780/esa-2853f3b](https://doi.org/10.57780/esa-2853f3b), CC BY-NC 3.0 IGO.
- **Discovery Engine** catalogues and cutouts (Euclid Collaboration: Walmsley et al.), CC-BY-4.0.

The extracts in `public_inputs/` hold only the columns and rows the build reads; the Discovery
Engine extracts are CC-BY-4.0, the MER and deflector extracts are Euclid Q1 data under
CC BY-NC 3.0 IGO. Images: ESA Euclid/Euclid Consortium/NASA/Q1-2025.

`scripts/lens_dataset.py`'s `zoobot_stretch` reproduces `fits_to_jpg_like_scaling` from the
**galaxy-datasets** package by Mike Walmsley (GPL-3.0), with a constant-image guard added.

## Provenance

- **The λ = 0 confirmatory folds' loss module is `evidence/kaggle_bundle_v14/da_losses.py`;** it
  differs from `scripts/da_losses.py` only inside `BNMLoss`.
- **Eight procedural claims are checked against the author's unpublished lab log** when the
  extract is made; the extract records the log's sha256, so the check can be repeated if the log
  is shared.
- **Section references (§7.x) in code comments and strings point to the author's internal lab
  log, which is not distributed;** every value the build needs from it is in
  `public_inputs/record_extract.json`.
- **The balanced FreeMatch fold-1 run used an earlier version of
  `scripts/run_unfreeze_freematch.py`** than the one released: its `run_config.json` lacks the
  three step-zero tolerance keys the released runner always writes, and records the exact
  step-zero tolerance that preceded the gate change. That version is not recovered.
- **For four 1:10 arms (DANN, BNM and the CORAL and FreeMatch frozen stages) the exact versions
  of `dilute10_common.py`, `dilute10_batch_log.py` and the arm's wrapper that ran are lost;**
  the released files are later versions, and the run records hold the digests of the versions
  that ran. Baseline A, FreeMatch and CORAL's balanced frozen stage recorded no per-module
  digests, and BNM's balanced runs recorded only the files its pre-registration hashes.

## Known issues

- **The bare-deflector IDs in `data/splits/splits.json` are float-rounded** (3,575 of 3,601).
  Each maps to exactly one exact ID in `data/ids/deflector_ids.csv`
  (`int(float(exact)) == stored`).
- **`scripts/exact_ids.py` names the project's absolute local path** (`PROJECT = ...`). The
  build uses it only when the original image files are present; with the public extracts it
  goes through `_common._identities_public()` instead.
- **`scripts/generate_arc_on_real.py`'s docstring is stale**: it mentions an Euclid VIS PSF, a
  real HST/COSMOS source and Poisson noise on the arc; the code convolves with a Gaussian of
  FWHM 0.16″, draws parametric COSMOS profiles and adds no noise (`PROJECTv7.md` §7.35).
- **`scripts/lens_dataset.py`'s `LensTransform` docstring is stale**: it says "rot90", but the
  code rotates by a continuous random angle in ±90° (`RandomAffine`), as the paper's §3.2 states.
  Run records label the weak augmentation "+-3px translate"; it also includes the flips and the
  rotation.
- **The λ = 0 confirmatory folds' `run.json` field `fp16_skipped_cumulative` reads 0** because it
  was written before training; the skipped-step counts are in each fold's `steps.csv` and
  `checkpoints.csv` (`skipped_cumulative`), which the paper's Appendix J reads.
- **Eight checkpoints are included, and no others:** the frozen-stage heads of the 1:10 runs,
  `runs/dilute10/coral_frozen/fold*/checkpoints/step_8000.pt` (16.7 kB each) and
  `runs/dilute10/freematch_frozen/fold*/checkpoints/step_8000.pt` (1.66 MB each). They are small
  linear heads on the frozen encoder's features and contain no image data; they are protected
  artifacts, so `integrity.py --verify` needs them. No encoder weights are included.
- **Development material is not in this repository:** score files of development runs that the
  build does not read, audit records the build does not read, the project's working reports,
  compiled drafts, `logs/` and IDE files. They are classified for a later archive.
- **Several run records contain absolute local paths and the Kaggle account name.** Many of
  them are hashed (protected or self-certifying) and are kept byte for byte.

## Licence

GPL-3.0; see `LICENSE`.

## Citing

Please cite the paper (arXiv: forthcoming) and this archive (doi:[10.5281/zenodo.24285258](https://doi.org/10.5281/zenodo.24285258)); `CITATION.cff`
has both.
