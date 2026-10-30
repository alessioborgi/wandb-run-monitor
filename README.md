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
- Refresh manually or every two minutes while the website is open.
- Hourly GitHub Actions email alerts for newly completed combinations.

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

No authenticated W&B data has been checked yet. Hosting and secret configuration are required before the live dashboard and email checker become operational.

## Regression tests

Run `python -m unittest discover -s tests -v`. The private repository also includes a GitHub Actions test workflow checking metric selection, non-NSD models, missing scores, and creator filtering.
