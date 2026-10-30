# Sheaf Run Monitor
Interactive live Streamlit dashboard for `adavit/sheaf-mpnn-runs`.

- Filter: `state == finished`, `user == fs395` (creator username).
- Group by: `state, model, dataset, transport_type`.
- Count completed runs against **250** per combination; show remaining and progress.
- Include **every model family** belonging to fs395. Model name is read from config `model` with fallbacks for `model_name`, `model_type`, `architecture`, `arch`, `network`, and nested name/type fields.
- For every model/dataset/transport_type group, choose **one best run** using the highest available validation metric:
  - `minesweeper`, `tolokers`, `questions`: `mean_val_roc_auc`. Show that run's `mean_test_roc_auc` and `std_test_roc_auc`.
  - All other datasets: `mean_val_acc`. Show that run's `mean_test_acc` and `std_test_acc`.
- The progress table includes the selected run's scores and link. A separate leaderboard ranks combinations **within each dataset** by best validation performance. This prevents selecting runs based on test performance.
- If no run has a valid validation metric, keep its count but leave its best-score fields blank rather than making up a result.
- Refresh manually or every five minutes while the website is open.
- Hourly GitHub Actions email alerts for newly completed combinations.
- Stream W&B GraphQL pages in bounded batches (30 per page), keeping only one group summary and up to five run examples per combination, instead of caching thousands of full W&B Run objects. This reduces Streamlit Community Cloud memory pressure.
- The web app displays sample run IDs, not an exhaustive run-by-run table. Complete runs are available in W&B.

## Local launch
```bash
pip install -r requirements.txt
export WANDB_API_KEY='YOUR_API_KEY'
export DASHBOARD_PASSWORD='LONG_UNIQUE_PASSWORD'
streamlit run app.py
```

## Deployment
Host this private repository as a Streamlit application on a supported platform, specifying `app.py` as the entry point. Supply `WANDB_API_KEY` and `DASHBOARD_PASSWORD` as hosting secrets, not committed files. The W&B API endpoint defaults to `https://api.forge.coreweave.com`, configurable using `WANDB_BASE_URL`.

For email alerts add four **GitHub Actions repository secrets**: `WANDB_API_KEY`, `SMTP_USER`, `SMTP_PASSWORD`, and `ALERT_TO`. Gmail SMTP requires an app password (not your regular password). The workflow checks hourly, emails groups reaching 250 for the first time, and records them in `data/notified.json`. Never commit credentials.

The dashboard uses the existing Streamlit secrets; no credentials are stored in this repository. If Community Cloud previously disabled the app for memory overuse, use the app's **Manage app → Reboot app** control after the redeployment. GitHub Actions email notifications require their own separate secrets.

The compact GraphQL fetch uses a modern W&B Python SDK (`wandb>=0.27.1`). No authenticated live data has been checked by ChatGPT.

## Regression tests

Run `python -m unittest discover -s tests -v`. The private repository also includes a GitHub Actions test workflow checking metric selection, non-NSD models, missing scores, and creator filtering.


## Missing benchmark combinations

The dashboard's expected matrix is derived from actual per-family sweep YAMLs and the
`src/exp/registries/presets.py` model registry in the private source repository
[`pytorch-SheafNeuralNetworks-working`](https://github.com/alessioborgi/pytorch-SheafNeuralNetworks-working).
It no longer incorrectly assumes 4 restriction maps for every model.

| Model(s) | Logged W&B `transport_type` values | Expected per dataset |
| --- | --- | --- |
| `nsp`, `poly_nsd`, `bsnn` | diagonal, general, orthogonal, low_rank | 4 each |
| `id_sheaf` | diagonal (identity maps) | 1 |
| `conn_nsd` | orthogonal (feature-derived maps) | 1 |
| `joint_diff`, `rot_inv` | general | 1 each |
| `sheaf_attn` | res_sheafan | 1 |

**Sheaf attention distinction:** the preset registry also defines a `sheafan` variant,
but the shipped dataset-specific `sheaf_attn` sweep YAMLs use `res_sheafan` only.
The W&B `model` field is `sheaf_attn`; a user alias `sheaf_att` is accepted.

The three originally requested model families **`sheaf_deepset`**,
**`sheaf_set_transformer`**, and **`sheaf_gin`** do not occur in the inspected
source repository's sweep definitions or model registry. Until their actual
supported map sets are confirmed, they are excluded from **expected/missing**
totals (to avoid false missing experiments). Any runs already observed under
those names in W&B are shown separately as *unverified observations*.

Datasets: six Pei et al. (2020) benchmarks (`texas`, `wisconsin`,
`cornell`, `chameleon`, `squirrel`, `film`; Actor = Film)
and five Platonov et al. (2023) benchmarks
(`roman_empire`, `amazon_ratings`, `minesweeper`, `tolokers`, `questions`).

**Correct expected count:** 17 model-specific sweeps × 11 datasets =
**187 combinations**. Optionally include Platonov's
`chameleon_filtered` and `squirrel_filtered` as separate datasets:
17 × 13 = **221 combinations**. Previously the app incorrectly claimed
11 models × 11 datasets × 4 maps = 484 combinations.

Each row shows runs finished by W&B creator `fs395`, missing runs to the
dashboard's requested 250-run target, completion percentage and status.
The table is filterable and exportable, and uses only existing compact group
summaries (no extra per-run W&B retrieval).

**Note:** some shipped `joint_diff` and `rot_inv` sweeps configure
`n_trials: 100`. The dashboard retains your **250-run target**; this
is a user-defined monitoring threshold, not necessarily each sweep's
configured trial budget.

Run offline tests with `python -m unittest discover -s tests -v`.
