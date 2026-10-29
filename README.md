# Sheaf Run Monitor
Interactive live Streamlit dashboard for `adavit/sheaf-mpnn-runs`.

- Filter: `state == finished`, `user == fs395` (creator username).
- Group by: `state, model, dataset, transport_type`.
- Count completed runs against **250** per combination; show remaining and progress.
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
