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

The Streamlit dashboard includes a separate **Missing benchmark combinations** table. It compares all expected model–dataset–restriction map configurations against the compact W&B snapshot for finished runs from creator `fs395`.

- Models (11): `nsp`, `poly_nsd`, `bsnn`, `joint_diff`, `sheaf_deepset`, `sheaf_set_transformer`, `sheaf_gin`, `id_sheaf`, `conn_nsd`, `sheaf_att`, `rot_inv`.
- Pei et al. (2020) datasets (6): `texas`, `wisconsin`, `cornell`, `chameleon`, `squirrel`, `film` (alias: Actor).
- Platonov et al. (2023) datasets (5): `roman_empire`, `amazon_ratings`, `minesweeper`, `tolokers`, `questions`.
- Restriction maps (4): `diagonal`, `general`, `orthogonal`, `low_rank` (normalized from the W&B `transport_type` configuration field).

**Total expected:** 11 × 11 × 4 = **484 combinations**. Optionally include `chameleon_filtered` and `squirrel_filtered` as separate cleaned dataset variants (**572 combinations**). They are **not** merged with the original Chameleon and Squirrel benchmarks.

Each row shows existing finished runs, how many remain to reach 250, completion percentage, and whether it is **Missing** (zero), **Incomplete** (1–249), or **Complete** (250+). The table supports model/dataset/map/status filters and a CSV download, while retaining only a small expected-coverage matrix in memory. `nsp` and `nsd` are **distinct** models: this intentionally uses the model name specified by the user.

Sources:
- Pei et al., *Geom-GCN*, ICLR 2020.
- Platonov et al., *A Critical Look at the Evaluation of GNNs Under Heterophily*, ICLR 2023.

Run offline unit tests with `python -m unittest discover -s tests -v`.
