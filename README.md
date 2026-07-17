# YSocial Thesis Analytics Dashboard

An interactive Streamlit dashboard for analysing YSocial simulation databases.

## Online use

The deployed application automatically loads:

`data/Thesis_scaled_v1/database_server.db`

A different YSocial database can also be uploaded through the sidebar for the current browser session.

## Dashboard sections

- Interactions
- Friendship network
- Groups and communities
- Topics
- Textual content
- Comments
- Network × content analysis
- Experiment comparison

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_dashboard.py
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

The application reads the SQLite database for analysis and does not modify it.
