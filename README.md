# REFIT Energy Disaggregation
End-to-end household energy analytics using the REFIT dataset, covering raw data cleaning, washing machine usage analysis, and deep learning based appliance disaggregation (NILM).

The project inspects and cleans raw House 1 measurements, studies washing-machine demand across 19 homes, and compares two convolutional neural networks that estimate washer power from whole-home power.

**Current finding:** the saved models underestimate washer activity and have higher pooled mean absolute error than an always-off baseline.

## What is included

| File | Contents |
| --- | --- |
| [DATA.md](DATA.md) | Raw-data inspection, coverage, cleaning rules, missingness and label-quality concerns |
| [RESULTS.md](RESULTS.md) | Washer EDA, figures, validation, model metrics and failure analysis |
| [Recommendation.md](Recommendation.md) | One-page utility-facing recommendation and a measurable energy-saving pilot |
| [requirements.txt](requirements.txt) | Packages imported by the notebooks |

| Notebook | Purpose |
| --- | --- |
| [00_data_extraction.ipynb](notebooks/00_data_extraction.ipynb) | Dataset sources and archive extraction |
| [01_data_cleaning.ipynb](notebooks/01_data_cleaning.ipynb) | Inspect the two raw House 1 files, resolve duplicate timestamps and resample |
| [02_exploratory_analysis.ipynb](notebooks/02_exploratory_analysis.ipynb) | Coverage, cycle detection, hourly/weekday demand and comparisons across homes |
| [03_model_build.ipynb](notebooks/03_model_build.ipynb) | Seq2Point and Seq2Seq training, prediction, evaluation and model export |

```text
refit-energy-disaggregation/
├── notebooks/
├── data/                         # Download locally; raw/ and cleaned/ are gitignored
│   ├── raw/
│   │   ├── REFIT_RAW_081116.7z
│   │   ├── RAW_House1_Part1.csv
│   │   └── RAW_House1_Part2.csv
│   └── cleaned/
│       ├── CLEAN_REFIT_081116.7z
│       └── CLEAN_House*.csv
├── artifacts/
│   ├── refit_outputs/            # EDA figures, cycles, coverage and raw timestamp audit
│   └── model_results/            # Models, predictions, metrics and experiment metadata
├── requirements.txt
├── README.md
├── DATA.md
├── RESULTS.md
└── Recommendation.md
```

## Setup

Use a Python environment that supports TensorFlow, and a notebook runner such as VS Code with the Python and Jupyter extensions. Python 3.13.5 is used.

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install ipykernel
```

Select **REFIT Energy** as the notebook kernel. The additional `ipykernel` installation supports notebook execution; it is separate from the libraries used in the analysis.

For JupyterLab, optionally install and launch it from the same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install jupyterlab
.\.venv\Scripts\python.exe -m jupyterlab notebooks
```

## Obtain the data

Download the assignment's archives manually and place them at:

- `data/raw/REFIT_RAW_081116.7z`: [raw archive](https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z).
- `data/cleaned/CLEAN_REFIT_081116.7z`: [official cleaned archive](https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z).

The extraction notebook records unsuccessful HTTP downloading and uses local archives. Its raw extraction call is commented out; notebook 01 extracts House 1 directly from the raw archive. Notebook 00 extracts the cleaned archive.

REFIT provides active power in watts at a nominal eight-second sampling interval for 20 UK households. See the [dataset page](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/) and [official cleaned-data documentation](https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt) for provenance and appliance mappings. The dataset is distributed under CC BY 4.0; retain attribution when using it.

## Run the workflow

Run notebook cells from top to bottom in numerical notebook order. **The kernel working directory must be `notebooks/`**, because the current paths use `../data` and `../artifacts`. Check `Path.cwd()` before running if your editor launches kernels elsewhere.

1. **Notebook 00:** confirm the archive locations and extract the cleaned CSVs.
2. **Notebook 01:** extract and inspect raw House 1. Its CSV discovery currently takes the first two sorted CSVs under `data/raw`; keep that directory limited to the two source CSVs or explicitly select `RAW_House1_Part1.csv` and `RAW_House1_Part2.csv`. The cleaned-data export cell is commented out.
3. **Notebook 02:** use `CLEAN_DATA_DIR = "../data/cleaned"`. Change its existing `OUTPUT_DIR = Path("/refit_outputs")` to `OUTPUT_DIR = Path("../artifacts/refit_outputs")` to save results inside this repository.
4. **Notebook 03:** keep `DATA_DIR = Path("../data/cleaned")` and `OUTPUT_DIR = Path("../artifacts/model_results")`, then run the Seq2Point and Seq2Seq sections.

Notebook 03 currently checks that CSVs exist for all 19 EDA households before training on `MODEL_HOUSES = [1, 2, 3]`. Extract the complete cleaned archive even when training only those three homes. Its `AUTO_DOWNLOAD` configuration variable does not implement downloading.

Notebook 01 uses Windows-style raw-data paths. For other operating systems, change these to `Path("../data/raw")` and `Path("../data/raw/REFIT_RAW_081116.7z")`.

The custom raw-cleaning output is an independent inspection exercise. EDA and modeling read the **official cleaned data**, not notebook 01's output.

## Assumptions and preprocessing decisions

- Resample aggregate and washer measurements to one-minute arithmetic mean power. For complete one-minute readings, energy is `sum(power_W) / 60,000` kWh.
- Use the documented washer channel for each home. Combine House 4's two washer meters only for household washer demand; omit House 12 because it has no identified washer channel.
- In raw House 1, sort timestamps and combine conflicting duplicate timestamps using the maximum value per power channel and the last Unix timestamp. This preserves larger readings but can bias energy upward and combine values that were not measured together.
- The raw routine temporarily masks zeros and then replaces missing grouped values with zero. This loses the distinction between missing readings and appliance-off states. Forward filling later fills at most five missing minute slots; remaining gaps stay missing. See [DATA.md](DATA.md) for the implications.
- For official cleaned data, mask aggregate readings outside 0-25,000 W and individual washer readings outside 0-4,000 W. Recombine chunk sums and counts to obtain correct minute means. Add missing minute slots as NaN without new gap filling.
- EDA records issue flags but retains flagged minutes. Modeling rejects any window containing an issue flag, which greatly reduces active examples. These outputs therefore have different quality-selection rules.

## Modeling and validation

The two models share five Conv1D layers with filters `30, 30, 40, 50, 50`, kernel sizes `10, 8, 6, 5, 5`, and a 1,024-unit dense layer. Seq2Point predicts the washer's midpoint power from a 61-minute aggregate window. Seq2Seq predicts the corresponding 61-minute washer sequence and averages overlapping predictions.

The architecture follows ideas in [Zhang et al. (2018), Sequence-to-point learning with neural networks for nonintrusive load monitoring](https://arxiv.org/abs/1612.09106). This implementation adapts the window length, data resolution, normalization and evaluation to REFIT.

Default experiments use Houses 1, 2 and 3 from **2014-06-01 inclusive to 2014-11-01 exclusive**. Each household's timeline is split chronologically into 70% training, 15% validation and 15% test **before** generating windows. Windows cannot cross split boundaries. Aggregate normalization uses training data only; early stopping uses validation loss. Ground-truth targets are expressed in kW during training and predictions are converted back to W.

Training uses Adam, learning rate 0.0005, batch size 256 and at most 30 epochs. Seq2Point and Seq2Seq early-stopping patience is three and five epochs, respectively. Both active-sample weights are 1.0. The 120,000-window per-home cap was not reached in the saved training run, so its sampling routine did not rebalance the data.

Evaluation reports MAE, RMSE, activity precision/recall/F1 at 20 W, and relative total-energy error (SAE), with an always-off baseline. Compare the CNNs on their **shared test timestamps**. This tests future periods in known households; it does not establish performance on unseen households.

## Trade-offs and current limits

One-minute inputs simplify computation but blur short transitions. The 61-minute context is shorter than many detected washer events. Centered Seq2Point inference uses 30 minutes of future aggregate context, so it requires delayed or retrospective use. Seq2Seq also uses future context and overlap averaging can smooth peaks.

Quality filtering protects against questionable labels but removes most active windows in this run: only 234 of 18,684 paired active training midpoints remain. Low overall MAE is consequently dominated by off-state performance. The CNNs' pooled MAEs are 1.370 W and 1.395 W, compared with 1.150 W for the always-off baseline. Full interpretation is in [RESULTS.md](RESULTS.md).

The saved 15/30-minute figure illustrates signal smoothing only. Models were not retrained or evaluated using 15/30-minute aggregate inputs.

## Tools and Development Environment

1. **Google Colab:** Used for training and experimenting with the CNN-based appliance disaggregation models.
2. **Local Environment:** Used for local development, code execution, and testing.

## Possible improvements

1. Preserve raw missingness and imputation flags; replace maximum-based duplicate resolution with a justified rule, quantify outlier removals, and add matching before/after cleaning plots.
2. Audit how issue flags and contiguous-window requirements remove washer events. Establish trustworthy complete cycles without silently admitting inconsistent labels.
3. Address the remaining activity imbalance through cycle-aware sampling or weighted losses, using validation data to choose settings.
4. Test longer context windows and simpler event-based baselines before increasing model complexity.
5. Evaluate unseen homes and later seasons, report event/active-state errors, and estimate uncertainty across independent household/day blocks and multiple seeds.
6. Retrain and evaluate at actual utility meter resolutions, then validate on locally submetered Indian households.
7. Centralize paths and configuration, persist all data-quality masks, and lock the tested Python/package versions.
8. Try better model like Transform or LSTM models.

## References

- Zhang, C., Zhong, M., Wang, Z., Goddard, N., & Sutton, C. (2018). [Sequence-to-Point Learning with Neural Networks for Non-Intrusive Load Monitoring](https://arxiv.org/pdf/1612.09106). *Proceedings of the AAAI Conference on Artificial Intelligence*.

