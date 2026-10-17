<div align="center">

# EAUC: Eccentricity-Area Under the Curve

**Beyond RMSE and MAE: unmask hidden bias and unfairness in dyadic regression models (recommenders, drug response, trade, ...).**

[![arXiv](https://img.shields.io/badge/arXiv-2401.10690-b31b1b.svg)](https://doi.org/10.48550/arXiv.2401.10690)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Jorge Paz Ruza et al. · Universidade da Coruña, CITIC

<img src="assets/eccentricity_curve.png" alt="Eccentricity vs. error curves of four predictors on synthetic data: the dyad-average predictor is the most accurate near zero eccentricity but its error grows fastest as eccentricity increases" width="460">

<sub>Illustrative curves on synthetic data, produced by <code>examples/synthetic_example.py</code>. They are not results from the paper.</sub>

</div>

## Overview

Dyadic regression models predict a real value for a pair of entities, such as the rating a user gives an item. The paper shows that state-of-the-art models are biased towards the mean past value of the entities involved (**eccentricity bias**): they are precise when an observation is close to what the entities usually get, and much worse, even worse than random, when it is not. Global metrics such as RMSE and MAE do not see this. **EAUC** does: it integrates the prediction error over the eccentricity of the observations, so it rewards models that stay accurate on the unusual cases where fairness matters most.

This repository contains:

- **`eauc`, an installable library** with the metric, the *eccentricity vs. error* curve, the dataset-level KS uniformity statistic and the post-training bias corrections of the paper. It works on any `(user, item, value, prediction)` table, whatever the domain.
- **The evaluation framework of the paper**: the model wrappers and analysis scripts used to produce the experiments.

**Definitions.** With `mean_u` and `mean_i` the mean train value of user `u` and item `i`:

- Dyadic Mean Value: `DMV_ui = (mean_u + mean_i) / 2`
- Eccentricity of an observation: `Ecc_ui = |r_ui - DMV_ui|`
- EAUC: the area under the curve of absolute error vs. eccentricity over the test set, divided by `(max r - min r)²`. It lies in `[0, 1]`; **lower is better**.

## Installation

```bash
git clone https://github.com/Kominaru/EAUC.git
cd EAUC
pip install -e .          # numpy, pandas, scikit-learn, scipy
pip install -e ".[plot]"  # optionally, matplotlib for the plotting options
```

## Quickstart

Score any model from two CSV files (`user_id, item_id, rating` for the train set, plus `pred` for the test set):

```python
import pandas as pd
from eauc import eauc, eccentricity_error_curve, dataset_ks

train = pd.read_csv("train.csv")   # columns: user_id, item_id, rating
test = pd.read_csv("test.csv")     # columns: user_id, item_id, rating, pred

print(eauc(train, test))                          # 0.0 (best) .. 1.0 (worst)
print(eccentricity_error_curve(train, test))      # MAE per eccentricity bin
print(dataset_ks(pd.concat([train, test])))       # how prone the dataset is to induce the bias
```

Other column names are supported through `user_col`, `item_col`, `value_col` and `pred_col`, so the same calls work for drug-cell line responses, trade volumes or any other dyadic table.

Reduce eccentricity bias of an already-trained model with a post-hoc correction fitted on a small probe (see the paper, Section VII):

```python
from eauc import EccentricityCorrector, make_probe

train_rest, probe, test_rest = make_probe(train, test)          # probe = 20% of test + as many train rows
corrector = EccentricityCorrector("linear").fit(train, probe)   # or "random_forest"
corrected = corrector.transform(test_rest)                      # `pred` replaced by the corrected value
print(eauc(train, test_rest), "->", eauc(train, corrected))
```

A complete runnable example on synthetic data (no downloads): `python examples/synthetic_example.py`.

### Command line

```bash
eauc score   --train train.csv --test test.csv                 # EAUC, RMSE and MAE
eauc curve   --train train.csv --test test.csv --plot curve.png
eauc ks      --data ratings.csv                                # dataset D_KS statistic
eauc correct --train train.csv --test test.csv --out corrected.csv --strategy linear
```

### API notes

| Function | Purpose |
|---|---|
| `eauc(train, test, method="ordered", tails=1)` | The metric. `"ordered"` is Algorithm 1 of the paper; `"grouped"` and `"bins"` are the variants used in the plots. |
| `eccentricity_error_curve(train, test, bins=20)` | Mean absolute error per eccentricity bin. |
| `add_eccentricity(train, test)` | Adds `user_mean`, `item_mean`, `dmv`, `eccentricity` columns (rows with unseen users or items are dropped). |
| `dataset_ks(data)` | Entity-wise Kolmogorov-Smirnov uniformity statistic of the paper (Eq. 6). |
| `make_probe`, `EccentricityCorrector` | Probe extraction and the `linear` / `random_forest` corrections, with optional ML-RUS balancing. |

Means are always computed on the train set only, and inputs are never modified.

## Repository layout

| Path | Contents |
|---|---|
| `eauc/` | The Python package: metric (`metrics.py`), post-hoc corrections (`correction.py`), command line (`cli.py`) |
| `tests/` | Tests of the metric, the corrections and the command line (`pytest`) |
| `examples/synthetic_example.py` | End-to-end example on synthetic data |
| `models/` | Model wrappers used in the paper: MF, GC-MC, GLocal-K, Bayesian SVD++, MRMA and trivial baselines |
| `src/` | Analysis and figure scripts of the paper experiments (`avg_bias_auc_analysis.py`, `regression_corrector.py`, `dataset_complexity_measure.py`, `figure_scripts/`) |
| `download_data.py` | Downloads the public datasets (`python download_data.py ml-1m`) |
| `envs_requirements/` | Pinned environments of the model wrappers used in the paper |

## Reproducing the paper experiments

The scripts under `models/` and `src/` are the evaluation framework used for the paper. Each model wrapper saves `train_outputs_<n>.csv` and `test_outputs_<n>.csv` (columns `user_id, item_id, rating, pred`) under `outputs/<dataset>/<model>/`, and `src/avg_bias_auc_analysis.py` compares the models:

```bash
python download_data.py ml-1m
python models/MF/main.py                                   # see the script for its options
python -m src.avg_bias_auc_analysis --models MF average --dataset ml-1m --exec_i 1 --exec_n 1
```

The datasets of the paper are Movielens, Netflix Prize, GDSC1, CTRPv2, IMF DOTS 2023 and Kiva microloans; `download_data.py` fetches the publicly downloadable ones. Per-model hyperparameters are listed in Appendix A of the paper, and the environments used by each wrapper in `envs_requirements/`.

> The library reimplements the metric and the correction recipe of the paper in a reusable form. The legacy scripts in `src/` are kept as they were used for the paper (the `eauc` library now provides the metric used by `src/avg_bias_auc_analysis.py`). The paper's sigmoid and inverse-linear corrections are only available in `src/regression_corrector.py`.

## Results

Selected results from the paper (Netflix Prize, test set, MF model; see Table IV of the paper for the full table):

| Predictor | RMSE ↓ | MAE ↓ | EAUC ↓ |
|---|---|---|---|
| Random | 1.696 | 1.386 | 0.395 |
| Dyad Average | 0.964 | 0.781 | 0.435 |
| MF | 0.826 | 0.635 | 0.354 |
| MF + Linear correction + ML-RUS (Clipping) | 0.879 | 0.686 | 0.307 |
| MF + Random Forest correction | 1.008 | 0.809 | 0.276 |

The Dyad Average predictor has a far lower RMSE than the random one but a *higher* EAUC: RMSE hides the bias that EAUC reveals. The corrections trade some global error for a lower EAUC, i.e. less eccentricity bias.

## Citation

If you use EAUC or this code, please cite the paper (GitHub's "Cite this repository" button uses [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{paz2024beyond,
  title   = {Beyond {RMSE} and {MAE}: Introducing {EAUC} to unmask hidden bias and unfairness in dyadic regression models},
  author  = {Paz-Ruza, Jorge and Alonso-Betanzos, Amparo and Guijarro-Berdi{\~n}as, Bertha and Cancela, Brais and Eiras-Franco, Carlos},
  journal = {arXiv preprint arXiv:2401.10690},
  year    = {2024},
  doi     = {10.48550/arXiv.2401.10690},
  url     = {https://arxiv.org/abs/2401.10690}
}
```

## Authors

**Jorge Paz Ruza** [![ORCID](https://img.shields.io/badge/-ORCID-A6CE39?style=flat-square&logo=orcid&logoColor=white)](https://orcid.org/0000-0002-7869-1070) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jorge-paz-ruza-646141186/)<br>
**Amparo Alonso Betanzos** [![ORCID](https://img.shields.io/badge/-ORCID-A6CE39?style=flat-square&logo=orcid&logoColor=white)](https://orcid.org/0000-0003-0950-0012) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/amparo-alonso-80516211/)<br>
**Bertha Guijarro Berdiñas**<br>
**Brais Cancela** [![ORCID](https://img.shields.io/badge/-ORCID-A6CE39?style=flat-square&logo=orcid&logoColor=white)](https://orcid.org/0000-0002-2295-4142)<br>
**Carlos Eiras Franco** [![ORCID](https://img.shields.io/badge/-ORCID-A6CE39?style=flat-square&logo=orcid&logoColor=white)](https://orcid.org/0000-0001-6322-7593) [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/carloseiras/)

LIDIA Group, CITIC, Universidade da Coruña (A Coruña, Spain).

## License

Code released under the [MIT License](LICENSE). The wrappers in `models/` build on third-party implementations (for example GC-MC by Jiani Zhang, see `models/GCMC/README.md`), which keep their original licenses.

## Contact

Questions, issues and collaboration ideas are welcome: open a [GitHub issue](https://github.com/Kominaru/EAUC/issues) or write to Jorge Paz Ruza at [j.ruza@udc.es](mailto:j.ruza@udc.es).

## Acknowledgements

This work was funded by MICIU/AEI/10.13039/501100011033 and ESF+ (FPU21/05783), ERDF A way of making Europe (PID2019-109238GB-C22, PID2023-147404OB-I00), ERDF EU (PID2021-128045OA-I00), and the Xunta de Galicia (ED431C 2022/44, ED431G 2023/01) with European Union ERDF funds. CITIC is a Research Center accredited by the Galician University System.
