# %% [markdown]
# # 01 · CRM Data Cleaning and Preparation
#
# **Project:** CRM analysis of online programming school X (July 2023 – June 2024).
# **Goal of this notebook:** turn 4 raw CRM exports into clean, linked tables
# that support funnel, marketing-efficiency and unit-economics analysis.
#
# | Table | Contents | Key |
# |---|---|---|
# | Contacts | contacts (leads) | `Id` |
# | Calls | sales managers' calls | `CONTACTID` → Contacts |
# | Spend | daily ad spend | `Source`, `Campaign` → Deals |
# | Deals | potential deals | `Contact Name` → Contacts |
#
# All cleaning logic lives in the module [`src/cleaning.py`](../src/cleaning.py) so it can be
# reused and tested. This notebook checks the raw data, runs the ETL and validates the result.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent / "src"))
import cleaning as etl

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

# %% [markdown]
# ## 1. Raw data
#
# We read `.xlsx`, not `.csv`: in the CSV export the Cyrillic text in *Level of Deutsch* turned into `?`,
# and 19-digit IDs were rounded as floats (which breaks the joins). All IDs are read as strings.

# %%
raw = etl.load_raw()
overview = pd.DataFrame({
    name: {"rows": len(df), "columns": df.shape[1],
           "full duplicates": df.duplicated().sum(),
           "missing cells, %": round(df.isna().mean().mean() * 100, 1)}
    for name, df in raw.items()
}).T
overview

# %%
def missing_report(df):
    m = df.isna().sum()
    return (pd.DataFrame({"missing": m, "%": (m / len(df) * 100).round(1),
                          "dtype": df.dtypes.astype(str)})
            .query("missing > 0").sort_values("%", ascending=False))

missing_report(raw["Deals"])

# %% [markdown]
# **What Deals looks like before cleaning:**
# - `Payment Type`, `Months of study`, `Course duration`, `City`, `Level of Deutsch` are filled for a small share
#   of deals: managers fill them only at late funnel stages. This is how the process works, not an error.
#   A gap means "the deal never reached this stage", so these fields are **neither dropped nor mean-imputed**.
# - Amounts are stored as text in different formats: `1000`, `€ 3.500,00`, `$11.000,00`.
# - `SLA` mixes `time` (hh:mm:ss) and `timedelta` (more than one day).

# %%
raw["Deals"][["Initial Amount Paid", "Offer Total Amount", "SLA", "Level of Deutsch", "City"]].dropna().sample(8, random_state=7)

# %%
missing_report(raw["Calls"])

# %% [markdown]
# ## 2. Cleaning rules and why
#
# | # | Table | Problem | Fix | Reason |
# |---|---|---|---|---|
# | 1 | Deals | `Lost Reason = Duplicate` | drop | per the FAQ this is a technical duplicate lead; keeping it inflates leads and deflates conversion |
# | 2 | Deals | `Closing Date` < `Created Time` | → `NaT` | a deal cannot close before it is created (FAQ: data error, the analyst decides) |
# | 3 | Deals | `Initial Amount Paid` > `Offer Total Amount` | 2 rows with offer 1,200 on an 11-month course are a typo in the offer; elsewhere initial is capped at offer | the first payment cannot exceed the course price; conservative revenue estimate |
# | 4 | Deals | offer = €0 / 1 / 9 | flag `is_demo` | per the FAQ these are demo accesses for a token fee, not students |
# | 5 | Deals | `Payment Done` but `Months of study` empty | not counted as a student | FAQ: "if the month is not filled in, the student is not studying and the money must not be counted" |
# | 6 | Deals | amounts as text, SLA of mixed type | → float / seconds | calculations need numbers |
# | 7 | Deals | `Level of Deutsch` as free text (Cyrillic б1, в1, "B1 (until February)") | → CEFR `A0…C2` | Cyrillic б/в = B; take the current level (the first one mentioned) |
# | 8 | Deals | cities: `-`, street addresses, different spellings | → `city`, `country` | needed for geographic analysis |
# | 9 | Deals | `Content`, `Term` | drop | ad-level detail; campaign and source are kept |
# | 10 | Deals | categorical gaps | explicit labels `Unknown`, `Not rated`, `Not lost` | the gap itself is informative; dropping rows would distort the funnel |
# | 11 | Calls | `Dialled Number`, `Tag` 100% empty | drop | no information |
# | 12 | Calls | no duration | drop rows (83) | a call without a duration cannot be analysed |
# | 13 | Calls | `Outgoing Call Status` empty for inbound calls | → `Not applicable` | not missing, just not applicable |
# | 14 | Spend | 917 full duplicates | drop | all are zero rows and do not affect totals |
# | 15 | Spend | `Campaign` empty | → `(no campaign)` | the spend is real and must not be dropped |
#
# **Revenue.** Per the FAQ, payment is confirmed by the status `Payment Done`, and the first payment is `Initial Amount Paid`.
# The rest of the course is paid monthly, so the revenue received by the export date is:
#
# $$revenue = initial + (months\_of\_study - 1) \cdot \frac{offer - initial}{course\_duration - 1},\quad revenue \le offer$$
#
# For a one-off payment (`initial = offer`), revenue = `offer`.

# %%
clean, logs = etl.run_etl(save=True)
for name, lines in logs.items():
    print(f"\n=== {name}")
    for line in lines:
        print(" •", line)

# %% [markdown]
# ## 3. Validating the result

# %%
pd.DataFrame({
    name: {"rows before": len(raw[name]), "rows after": len(df),
           "columns": df.shape[1], "full duplicates": df.duplicated().sum()}
    for name, df in clean.items()
}).T

# %%
deals = clean["Deals"]
deals.dtypes.to_frame("dtype").T

# %%
checks = {
    "deal ids are unique": deals["id"].is_unique,
    "no closing_date < created_time": not (deals["closing_date"] < deals["created_time"].dt.normalize()).any(),
    "initial <= offer everywhere": not (deals["initial_amount_paid"] > deals["offer_total_amount"]).any(),
    "students have months_of_study": deals.loc[deals["is_student"], "months_of_study"].notna().all(),
    "students have a product": deals.loc[deals["is_student"], "product"].ne("Unknown").all(),
    "revenue only for students": (deals.loc[~deals["is_student"], "revenue"] == 0).all(),
    "deal contact_id found in Contacts, %": round(deals["contact_id"].isin(clean["Contacts"]["id"]).mean() * 100, 1),
    "call contact_id found in Contacts, %": round(clean["Calls"]["contact_id"].dropna().isin(clean["Contacts"]["id"]).mean() * 100, 1),
}
pd.Series(checks, name="check").to_frame()

# %%
deals["german_level"].value_counts().to_frame("deals")

# %% [markdown]
# ## 4. Summary
#
# - Deals after cleaning: **19,822** (1,771 technical duplicates removed), calls: **95,791**, spend rows: **19,862**.
# - Students (paid + study months filled in, not demo): **837** of 856 `Payment Done` deals.
# - The joins work: 99.7% of deals and 100% of calls with a contact find their contact.
# - Clean tables are saved to `data/processed/` in two formats: **CSV** (for Tableau/Sheets) and
#   **pickle** (for Python: keeps the `datetime`, `category` and `bool` types and loads instantly).
#
# ➡️ Next step: [02_eda_analysis.ipynb](02_eda_analysis.ipynb)
