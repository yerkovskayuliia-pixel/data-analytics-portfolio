# Data Analytics Portfolio — Yuliia Yerkovska

Projects from my Data Analytics training: Python, Excel, Power BI and Tableau.

| Project | Tools | What it shows |
|---|---|---|
| [CRM Analytics for an Online IT School](python-crm-analysis/) | Python, pandas, statsmodels, Plotly | Data cleaning, EDA, unit economics, metric tree, A/B test design |

_More projects (Excel, Power BI, Tableau) are being added._

---

## Repository rules: no credentials on GitHub

Database logins, passwords, API keys and connection strings must never be committed.

- **Secrets live in `.env`**, which `.gitignore` excludes. `.env.example` shows the variable names without values.
- **Python** reads them from the environment:
  ```python
  import os
  from dotenv import load_dotenv   # pip install python-dotenv
  load_dotenv()
  conn = psycopg2.connect(host=os.getenv("DB_HOST"), dbname=os.getenv("DB_NAME"),
                          user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"))  # secret-scan: ignore
  ```
- **Pre-commit check.** `tools/check_secrets.py` runs before every commit. It scans code, notebooks and SQL, plus the inside of `.xlsx`, `.pbix`/`.pbit` and `.twb`/`.twbx` files (Power Query and Tableau connections). If it finds a password, key or `user:password@` URL, the commit is blocked.
- **Power BI / Tableau.** Credentials are kept in the desktop app, not in the file. Never use *Embed password* when publishing a Tableau workbook. Server names in connections are allowed, passwords are not.

After cloning, turn the check on once:
```bash
git config core.hooksPath .githooks
```
Check any folder manually before adding it:
```bash
python tools/check_secrets.py path/to/folder
```
