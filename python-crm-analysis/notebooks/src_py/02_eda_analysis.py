# %% [markdown]
# # 02 · Exploratory Data Analysis (EDA)
#
# We work with the clean tables from [01_data_cleaning](01_data_cleaning.ipynb).
#
# **Definitions used throughout:**
# - **Lead**: a deal (potential deal) in the Deals table after technical duplicates are removed.
# - **Student (successful deal)**: `Payment Done` + `Months of study` filled in + not a demo access.
# - **C1 (conversion)** = students / leads.
# - **Revenue**: the amount actually received by the export date (see the formula in 01).
#
# Plan from the brief: descriptive statistics → time series → campaigns and sources → sales team →
# payments and products → geography → conclusions.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path.cwd().parent / "src"))
import cleaning as etl
import viz
from viz import PRIMARY, ACCENT, GOOD, MUTED, save, pct_axis, eur_axis

viz.set_style()
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
pd.set_option("display.width", 200)

data = etl.load_clean()
deals, calls, spend, contacts = data["Deals"], data["Calls"], data["Spend"], data["Contacts"]
students = deals[deals["is_student"]]
OBS = dict(observed=True)
print(f"Leads: {len(deals):,} | students: {len(students):,} | C1: {deals['is_student'].mean():.2%}")
print(f"Period: {deals['created_time'].min():%d.%m.%Y} – {deals['created_time'].max():%d.%m.%Y}")
print(f"Revenue: €{students['revenue'].sum():,.0f} | ad budget: €{spend['spend'].sum():,.0f}")

# %% [markdown]
# ## 1. Descriptive statistics
# ### 1.1 Numeric fields: mean, median, mode, range

# %%
def describe_num(s: pd.Series) -> pd.Series:
    s = s.dropna()
    mode = s.mode()
    return pd.Series({
        "n": len(s), "mean": s.mean(), "median": s.median(),
        "mode": mode.iloc[0] if len(mode) else np.nan,
        "min": s.min(), "max": s.max(), "range": s.max() - s.min(), "std": s.std(),
    })

numeric = {
    "Deals · offer_total_amount (students), €": students["offer_total_amount"],
    "Deals · initial_amount_paid (students), €": students["initial_amount_paid"],
    "Deals · revenue per student, €": students["revenue"],
    "Deals · months_of_study": students["months_of_study"],
    "Deals · course_duration, months": students["course_duration"],
    "Deals · deal_cycle_days (students)": students["deal_cycle_days"],
    "Deals · SLA, hours": deals["sla_sec"] / 3600,
    "Calls · duration_sec": calls["duration_sec"],
    "Spend · spend per row per day, €": spend["spend"],
    "Spend · clicks": spend["clicks"],
    "Spend · impressions": spend["impressions"],
}
stats = pd.DataFrame({k: describe_num(v) for k, v in numeric.items()}).T
stats

# %% [markdown]
# **Interpretation:**
# - **Course price** for students: the mode and median are €11,000 (the 11-month Digital Marketing and UX/UI courses).
#   The mean is lower (~€7.5k) because of cheap packages and the 6-month Web Developer course. The distribution is bimodal,
#   so the mean describes a "typical" student poorly; the median and the split by product are more accurate.
# - **First payment**: the mode is €1,000, the standard down payment for instalments.
# - **Deal cycle**: median 17 days, mean 33, with a long right tail (up to 305 days): some clients
#   take months to decide.
# - **SLA**: median ~5.5 h, mean ~31 h; some requests wait weeks for a reply. This is the first candidate growth lever.
# - **Calls**: median 8 seconds, mean 165; most calls do not get through, real conversations are rarer.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(students["offer_total_amount"], bins=20, ax=axes[0], color=PRIMARY)
axes[0].set_title("Course price (students)"); eur_axis(axes[0], "x"); axes[0].set_xlabel("")
sns.histplot(students["deal_cycle_days"].dropna(), bins=40, ax=axes[1], color=PRIMARY)
axes[1].axvline(students["deal_cycle_days"].median(), color=ACCENT, ls="--", lw=2, label="median")
axes[1].set_title("Lead → payment, days"); axes[1].legend(frameon=False); axes[1].set_xlabel("")
sns.histplot(np.log10(calls["duration_sec"] + 1), bins=40, ax=axes[2], color=PRIMARY)
axes[2].set_title("Call duration (log10 sec)"); axes[2].set_xlabel("")
axes[2].set_xticks([0, 1, 2, 3]); axes[2].set_xticklabels(["0", "10s", "100s", "1000s"])
plt.tight_layout(); save(fig, "02_distributions"); plt.show()

# %% [markdown]
# ### 1.2 Categorical fields: quality, stage, source, product

# %%
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
for ax, col in zip(axes.flat, ["quality", "stage", "source", "product"]):
    vc = deals[col].value_counts()
    vc = vc[vc > 0].sort_values()
    ax.barh(vc.index.astype(str), vc.values, color=PRIMARY, height=0.7)
    for i, v in enumerate(vc.values):
        ax.text(v, i, f" {v:,} ({v / len(deals):.0%})", va="center", fontsize=9, color=viz.TEXT_2)
    ax.set_title(f"Leads by {col}"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, vc.max() * 1.25)
plt.tight_layout(); save(fig, "02_categorical"); plt.show()

# %%
cat_conv = []
for col in ["quality", "source", "product", "education_type"]:
    g = deals.groupby(col, **OBS).agg(leads=("id", "size"), students=("is_student", "sum"))
    g["C1"] = g["students"] / g["leads"]
    g.index = pd.MultiIndex.from_product([[col], g.index.astype(str)])
    cat_conv.append(g)
pd.concat(cat_conv).style.format({"C1": "{:.1%}", "leads": "{:,}"}).background_gradient(subset="C1", cmap="Blues")

# %% [markdown]
# **Findings by category:**
# - **Stage:** 71% of leads end in `Lost`, 4.3% reach payment. The main loss reasons (below) are "doesn't answer" and
#   "stopped answering" (40% of lost deals, 51% together with invalid numbers): the problem is **reaching the lead and response speed**, not price.
# - **Quality:** class A converts in 34% of cases, B in 21%, C in 10%, D/E almost never. The manager's rating
#   predicts payment well, so the A+B share is a reliable *leading* metric for an A/B test.
# - **Product:** 82% of leads have no product (they never reached a consultation). Among the known ones the leaders are
#   Digital Marketing (≈55% of sales), UX/UI Design and Web Developer.

# %%
lost = deals.loc[deals["stage"].eq("Lost"), "lost_reason"].value_counts(normalize=True).head(10).sort_values()
fig, ax = plt.subplots(figsize=(10, 4.5))
colors = [ACCENT if r in ("Doesn't Answer", "Stopped Answering", "Invalid number") else PRIMARY for r in lost.index]
ax.barh(lost.index.astype(str), lost.values, color=colors, height=0.7)
pct_axis(ax, "x"); ax.set_title("Why leads are lost (share of Lost deals)")
ax.text(0.98, 0.05, "orange = never reached the client", transform=ax.transAxes, ha="right", color=viz.TEXT_2)
ax.grid(axis="y", visible=False)
save(fig, "02_lost_reasons"); plt.show()

# %% [markdown]
# ## 2. Time series analysis
# ### 2.1 Deal creation over time and its link to calls

# %%
monthly = deals.groupby("month").agg(leads=("id", "size"), students=("is_student", "sum"),
                                     revenue=("revenue", "sum"))
monthly["calls"] = calls.groupby(calls["call_start_time"].dt.to_period("M").dt.to_timestamp()).size()
monthly["spend"] = spend.groupby(spend["date"].dt.to_period("M").dt.to_timestamp())["spend"].sum()
monthly["C1"] = monthly["students"] / monthly["leads"]
monthly["calls_per_lead"] = monthly["calls"] / monthly["leads"]
monthly.style.format({"C1": "{:.1%}", "revenue": "€{:,.0f}", "spend": "€{:,.0f}", "calls_per_lead": "{:.1f}"})

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(monthly.index, monthly["leads"], marker="o", color=PRIMARY, label="Leads")
axes[0].plot(monthly.index, monthly["calls"] / 5, marker="o", color=ACCENT, label="Calls ÷ 5")
axes[0].set_title("Leads and sales calls per month"); axes[0].legend(frameon=False, loc="upper left")
axes[1].bar(monthly.index, monthly["C1"], width=20, color=PRIMARY)
pct_axis(axes[1], decimals=0); axes[1].set_title("Lead → student conversion (C1) by lead month")
for x, v in zip(monthly.index, monthly["C1"]):
    axes[1].text(x, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9, color=viz.TEXT_2)
plt.tight_layout(); save(fig, "02_time_series"); plt.show()
print(f"Monthly correlation of leads and calls: r = {monthly['leads'].corr(monthly['calls']):.2f}")

# %%
weekly = deals.set_index("created_time").resample("W")["id"].count().to_frame("leads")
weekly["calls"] = calls.set_index("call_start_time").resample("W")["id"].count()
weekly = weekly.dropna()
lagged = {f"lag {l} wk": weekly["leads"].corr(weekly["calls"].shift(-l)) for l in range(0, 4)}
pd.Series(lagged, name="corr(leads, calls)").to_frame()

# %% [markdown]
# **Findings:**
# - Lead volume grew from ~600 to 2,000–2,900 per month (peak in April 2024, matching the budget peak).
# - Calls follow the lead flow almost in sync (r ≈ 0.94 monthly, the weekly maximum is at lag zero):
#   the sales team scales with marketing, but **conversion falls**: 5–6% in August 2023 – January 2024 →
#   2.6–3.4% in April–May 2024. More traffic does not turn into proportionally more sales.
# - The C1 drop in May–June 2024 is partly because those deals have not "matured" yet (median cycle 17 days,
#   the tail is months long), so the last 1–2 months should be compared with care.

# %% [markdown]
# ### 2.2 Closing time and duration from creation to closing

# %%
closed = deals.dropna(subset=["closing_date"])
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
bins = [-1, 0, 3, 7, 14, 30, 60, 90, 400]
labels = ["same day", "1-3", "4-7", "8-14", "15-30", "31-60", "61-90", "90+"]
for status, color, lab in [(False, MUTED, "Lost / other"), (True, PRIMARY, "Students")]:
    part = closed[closed["is_student"] == status]
    share = pd.cut(part["deal_cycle_days"], bins, labels=labels).value_counts(normalize=True).sort_index()
    axes[0].plot(share.index.astype(str), share.values, marker="o", color=color, label=lab)
pct_axis(axes[0]); axes[0].set_title("Days from lead to closing"); axes[0].legend(frameon=False)
dow = students["closing_date"].dt.day_name().value_counts().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
axes[1].bar(dow.index.str[:3], dow.values, color=PRIMARY, width=0.7)
axes[1].set_title("Payments by weekday")
plt.tight_layout(); save(fig, "02_closing"); plt.show()

closed.groupby("is_student")["deal_cycle_days"].describe()

# %% [markdown]
# **Findings:**
# - Lost deals close fast (median within the first days: "couldn't reach" → Lost).
# - Students pay after 2–5 weeks on average: 50% within 17 days, 75% within 43 days.
#   So **payment is not a usable metric for a two-week A/B test**: within 14 days not even half of the future
#   students will have paid. A leading metric is needed (lead qualification).
# - Payments are spread evenly over weekdays and drop at weekends: managers are off and deals wait.
# - 341 paid deals have no closing date in the source data, so the cycle analysis covers 513 students.

# %% [markdown]
# ## 3. Campaign and source effectiveness
# ### 3.1 Sources: volume, lead quality, conversion, cost

# %%
AB = ["A - High", "B - Medium"]
src = deals.groupby("source", **OBS).agg(
    leads=("id", "size"), students=("is_student", "sum"), revenue=("revenue", "sum"),
    share_AB=("quality", lambda q: q.isin(AB).mean()))
src["spend"] = spend.groupby("source", **OBS)["spend"].sum().reindex(src.index).fillna(0)
src["C1"] = src["students"] / src["leads"]
src["CPL"] = src["spend"] / src["leads"]
src["CAC"] = (src["spend"] / src["students"]).replace(np.inf, np.nan)
src["ROMI"] = ((src["revenue"] - src["spend"]) / src["spend"]).replace(np.inf, np.nan)
src = src[src["leads"] > 10].sort_values("leads", ascending=False)
src.style.format({"share_AB": "{:.1%}", "C1": "{:.1%}", "CPL": "€{:.2f}", "CAC": "€{:,.0f}",
                  "ROMI": "{:.0%}", "revenue": "€{:,.0f}", "spend": "€{:,.0f}"}) \
   .background_gradient(subset=["C1", "share_AB"], cmap="Blues")

# %%
paid = src[src["spend"] > 0]
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(paid["CPL"], paid["C1"], s=paid["leads"] / 8, color=PRIMARY, alpha=0.75, edgecolor="white", lw=2)
for name, r in paid.iterrows():
    ax.annotate(name, (r["CPL"], r["C1"]), xytext=(8, 4), textcoords="offset points", fontsize=10, color=viz.TEXT)
pct_axis(ax, decimals=1); ax.set_xlabel("Cost per lead, €"); ax.set_ylabel("Lead → student, C1")
ax.set_title("Paid sources: cheaper leads ≠ better leads (bubble = lead volume)")
save(fig, "02_sources_scatter"); plt.show()

# %% [markdown]
# **Findings by source:**
# - **Organic** and **Webinar** have the best conversion (9.5% and 8.5%) and the highest share of A/B leads.
#   Webinars are the most efficient *paid* channel: CAC ≈ €111.
# - **Facebook Ads** is the main channel by volume: cheap leads (€7) and average conversion, CAC ≈ €172.
# - **Google Ads** leads cost twice as much as Facebook's at the same conversion → CAC ≈ €344, the most expensive large channel.
# - **TikTok** brings cheap but low-quality leads (C1 2.8%, A/B share 6%).
# - **CRM** (repeat touches of the existing base) barely converts (1.6%).

# %% [markdown]
# ### 3.2 Campaigns: lead generation and conversion

# %%
camp = deals.groupby("campaign").agg(leads=("id", "size"), students=("is_student", "sum"),
                                     revenue=("revenue", "sum"))
camp["spend"] = spend.groupby("campaign")["spend"].sum()
camp["C1"] = camp["students"] / camp["leads"]
camp["CAC"] = camp["spend"] / camp["students"].replace(0, np.nan)
top_camp = camp[camp["leads"] >= 150].sort_values("leads", ascending=False)
print(f"Campaigns in total: {len(camp)}, with ≥150 leads: {len(top_camp)}, they bring {top_camp['leads'].sum() / len(deals):.0%} of leads")
print(f"Deals campaigns found in Spend: {camp.index.isin(spend['campaign']).mean():.0%}")
top_camp.style.format({"C1": "{:.1%}", "CAC": "€{:,.0f}", "revenue": "€{:,.0f}", "spend": "€{:,.0f}"}, na_rep="—") \
    .background_gradient(subset="C1", cmap="Blues")

# %%
plot = top_camp.drop(index="(no campaign)").head(15).sort_values("C1")
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.barh(plot.index, plot["C1"], color=PRIMARY, height=0.7)
ax.axvline(deals["is_student"].mean(), color=ACCENT, ls="--", lw=2, label=f"average C1 {deals['is_student'].mean():.1%}")
for i, (c1, n) in enumerate(zip(plot["C1"], plot["leads"])):
    ax.text(c1, i, f"  {c1:.1%} · {n:,} leads", va="center", fontsize=9, color=viz.TEXT_2)
pct_axis(ax, "x"); ax.set_xlim(0, plot["C1"].max() * 1.45); ax.legend(frameon=False, loc="lower right")
ax.set_title("Top-15 campaigns by lead volume: conversion"); ax.grid(axis="y", visible=False)
save(fig, "02_campaigns"); plt.show()

# %% [markdown]
# **Findings by campaign:**
# - `brand_search_eng_DE` (brand search) converts at 9.3%: these people already know the school.
# - Broad Facebook campaigns (`02.07.23wide_DE`, `07.07.23LAL_DE`, `12.09.23interests_Uxui_DE`) convert at 5–5.5%, above average.
# - `performancemax_eng_DE` spent ~€34k (23% of the whole budget!) and brought 1 student: **a candidate to switch off**.
# - Webinar campaigns (`webinar1604`, `webinar1906`, `08.04.24wide_webinar_DE`) bring many sign-ups
#   but almost no payments: registering for a webinar ≠ intent to buy.
# - Only 29% of campaign names in Deals appear in Spend (63% of tagged leads), so spend and leads cannot be
#   matched for many campaigns. Consistent UTM tagging is needed.

# %% [markdown]
# ## 4. Sales team effectiveness

# %%
own = deals.groupby("deal_owner_name", **OBS).agg(
    deals=("id", "size"), students=("is_student", "sum"), revenue=("revenue", "sum"),
    sla_median_h=("sla_sec", lambda s: s.median() / 3600),
    share_AB=("quality", lambda q: q.isin(AB).mean()))
own["C1"] = own["students"] / own["deals"]
own = own[own["deals"] >= 50].sort_values("revenue", ascending=False)
own.style.format({"C1": "{:.1%}", "share_AB": "{:.1%}", "revenue": "€{:,.0f}", "sla_median_h": "{:.1f}"}) \
    .background_gradient(subset=["C1"], cmap="Blues").background_gradient(subset=["sla_median_h"], cmap="Oranges")

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
o = own.sort_values("revenue")
axes[0].barh(o.index.astype(str), o["revenue"], color=PRIMARY, height=0.7)
eur_axis(axes[0], "x"); axes[0].set_title("Revenue by deal owner"); axes[0].grid(axis="y", visible=False)
axes[1].scatter(own["deals"], own["C1"], s=own["revenue"] / 2500 + 20, color=PRIMARY, alpha=0.75, edgecolor="white", lw=2)
for name, r in own.iterrows():
    if r["revenue"] > 150_000 or r["C1"] > 0.1 or r["deals"] > 800 and r["C1"] < 0.02:
        axes[1].annotate(name, (r["deals"], r["C1"]), xytext=(6, 4), textcoords="offset points", fontsize=9)
pct_axis(axes[1]); axes[1].set_xlabel("Deals handled"); axes[1].set_ylabel("C1")
axes[1].set_title("Workload vs conversion (bubble = revenue)")
plt.tight_layout(); save(fig, "02_sales_team"); plt.show()

# %%
cr_calls = calls.groupby("call_owner_name", **OBS).agg(
    calls=("id", "size"), talk_share=("is_conversation", "mean"), avg_talk_min=("duration_sec", lambda s: s.mean() / 60))
cr_calls = cr_calls[cr_calls["calls"] > 500].sort_values("calls", ascending=False)
cr_calls.style.format({"talk_share": "{:.1%}", "avg_talk_min": "{:.1f}", "calls": "{:,}"})

# %% [markdown]
# **Findings on the sales team:**
# - The revenue leaders are **Charlie Davis** and **Ulysses Adams** (≈€1.15M between them, 32% of revenue).
# - **Oliver Taylor** converts 31% of 160 deals: judging by volume and quality, he receives "warm"
#   qualified deals (a closing manager). Comparing his C1 with the others directly would be misleading.
# - **Diana Evans** (927 deals, 1 student), **Rachel White** and **Bob Brown** (0 students) work on
#   first contact and qualification, not closing. They should be judged by the share of qualified leads.
# - Managers' median SLA ranges from 2 to 12+ hours: a 5× spread with the same lead flow.

# %% [markdown]
# ## 5. Payments and products
# ### 5.1 Payment types and their effect on success

# %%
with_pt = deals[deals["payment_type"].ne("Unknown")]
pt = with_pt.groupby("payment_type", **OBS).agg(
    deals=("id", "size"), students=("is_student", "sum"), revenue=("revenue", "sum"),
    avg_offer=("offer_total_amount", "mean"), avg_months=("months_of_study", "mean"))
pt["success_rate"] = pt["students"] / pt["deals"]
pt["revenue_per_student"] = pt["revenue"] / pt["students"].replace(0, np.nan)
pt.style.format({"success_rate": "{:.1%}", "revenue": "€{:,.0f}", "avg_offer": "€{:,.0f}",
                 "revenue_per_student": "€{:,.0f}", "avg_months": "{:.1f}"}, na_rep="—")

# %% [markdown]
# - The payment type is filled in for only 482 deals (managers enter it manually and inconsistently: it is empty for 57% of students).
# - **One Payment** succeeds in 80% of deals with a known type, **Recurring Payments** in 73%. Instalments are the main format
#   (2/3 of students with a known type) and bring more revenue per student thanks to the pricier 11-month courses.
# - **Reservation** barely converts (1 of 5).

# %% [markdown]
# ### 5.2 Popularity and success of products and study formats

# %%
known = deals[deals["product"].isin(["Digital Marketing", "UX/UI Design", "Web Developer"])]
prod = known.groupby(["product", "education_type"], **OBS).agg(
    leads=("id", "size"), students=("is_student", "sum"), revenue=("revenue", "sum"))
prod = prod[prod["leads"] > 20]
prod["C1"] = prod["students"] / prod["leads"]
prod["revenue_per_student"] = prod["revenue"] / prod["students"]
prod.style.format({"C1": "{:.1%}", "revenue": "€{:,.0f}", "revenue_per_student": "€{:,.0f}"}, na_rep="—")

# %%
p2 = prod.reset_index()
p2 = p2[p2["education_type"].isin(["Morning", "Evening"])]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.barplot(p2, x="product", y="students", hue="education_type", ax=axes[0],
            palette=[PRIMARY, ACCENT], hue_order=["Morning", "Evening"], width=0.7, gap=0.1)
axes[0].set_title("Students by product & format"); axes[0].set_xlabel(""); axes[0].legend(title="", frameon=False)
sns.barplot(p2, x="product", y="C1", hue="education_type", ax=axes[1],
            palette=[PRIMARY, ACCENT], hue_order=["Morning", "Evening"], width=0.7, gap=0.1)
pct_axis(axes[1]); axes[1].set_title("Conversion (lead with product → student)"); axes[1].set_xlabel("")
axes[1].legend(title="", frameon=False)
plt.tight_layout(); save(fig, "02_products"); plt.show()

# %% [markdown]
# **Findings on products:**
# - **Digital Marketing** is the most popular (57% of students) and highest-earning product (€2.27M).
# - The morning format dominates (≈80% of students), but **evening groups convert better**
#   (Digital Marketing 46% vs 23%, UX/UI 38% vs 21%): working people choose evenings and decide
#   more deliberately. There are practically no evening Web Developer groups, a possible niche.
# - Web Developer is shorter (6 months) and cheaper → less revenue per student (~€2.7k vs ~€4.8k for DM).

# %% [markdown]
# ## 6. Geographic analysis
# ⚠️ City and German level are filled in for only ~11% of deals: the manager enters them at the consultation stage.
# So C1 among deals with a city (≈33%) is not comparable with the overall C1 (4.2%). That is **selection bias**, not
# a "city effect". We compare cities and language levels **with each other** within this sample.

# %%
geo = deals[deals["city"].ne("Unknown")]
print(geo["country"].value_counts().to_string())
city = geo.groupby("city").agg(deals=("id", "size"), students=("is_student", "sum"), revenue=("revenue", "sum"))
city["C1"] = city["students"] / city["deals"]
city = city.sort_values("deals", ascending=False)
city.head(15).style.format({"C1": "{:.1%}", "revenue": "€{:,.0f}"})

# %%
top = city.head(12).sort_values("deals")
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.barh(top.index, top["deals"], color=MUTED, height=0.7, label="deals")
ax.barh(top.index, top["students"], color=PRIMARY, height=0.7, label="students")
for i, (n, s) in enumerate(zip(top["deals"], top["students"])):
    ax.text(n, i, f"  {s / n:.0%}", va="center", fontsize=9, color=viz.TEXT_2)
ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", visible=False)
ax.set_title("Top-12 cities: deals, students and conversion")
save(fig, "02_cities"); plt.show()

# %%
lvl_order = ["A0", "A1", "A2", "B1", "B2", "C1"]
g = geo[geo["german_level"].isin(lvl_order)].copy()
g["city_group"] = np.where(g["city"].isin(city.head(6).index), g["city"], "Other cities")
pivot_n = pd.crosstab(g["city_group"], g["german_level"]).reindex(columns=lvl_order, fill_value=0)
pivot_c1 = g.pivot_table(index="city_group", columns="german_level", values="is_student", aggfunc="mean",
                         observed=True).reindex(columns=lvl_order)
pivot_c1 = pivot_c1.where(pivot_n >= 5)  # hide cells with < 5 deals

fig, ax = plt.subplots(figsize=(10, 4.5))
sns.heatmap(pivot_c1, annot=pivot_c1.map(lambda v: "" if pd.isna(v) else f"{v:.0%}"), fmt="",
            cmap="Blues", vmin=0, vmax=0.6, linewidths=2, linecolor="#fcfcfb", cbar=False, ax=ax)
ax.set_title("Conversion by city and German level (cells with ≥5 deals)"); ax.set_xlabel(""); ax.set_ylabel("")
save(fig, "02_german_city"); plt.show()

lvl = g.groupby("german_level", **OBS).agg(deals=("id", "size"), students=("is_student", "sum"))
lvl["C1"] = lvl["students"] / lvl["deals"]
lvl.reindex(lvl_order).style.format({"C1": "{:.1%}"})

# %% [markdown]
# **Findings on geography:**
# - 99% of clients with a known city live in **Germany**; a few are in Austria and Poland.
# - **Berlin** is the largest market (182 deals) and has the best conversion among large cities (43%), followed by
#   Munich, Hamburg and Leipzig. Geo-targeting of Berlin and other large cities is worth strengthening.
# - The strongest segment is **Berlin + B1**: 62% conversion.
# - **German level:** the core audience is **B1** (≈68% of known levels). Conversion at B1 (37%) is higher than
#   at B2 (25%) and A2 (30%). People with B2 more often find work without retraining; people with A2 doubt
#   they can handle the course. B1 is the core audience and communication should target it.

# %% [markdown]
# ## 7. Response speed (SLA) and conversion: the bridge to product analytics

# %%
deals["sla_bucket"] = pd.cut(deals["sla_sec"] / 3600, [0, 0.25, 1, 4, 24, 1e6],
                             labels=["< 15 min", "15–60 min", "1–4 h", "4–24 h", "> 24 h"])
sla = deals.groupby("sla_bucket", **OBS).agg(
    leads=("id", "size"), C1=("is_student", "mean"), share_AB=("quality", lambda q: q.isin(AB).mean()),
    no_answer=("lost_reason", lambda l: l.isin(["Doesn't Answer", "Stopped Answering"]).mean()))
display(sla.style.format({"C1": "{:.1%}", "share_AB": "{:.1%}", "no_answer": "{:.1%}"}))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].bar(sla.index.astype(str), sla["share_AB"], color=PRIMARY, width=0.7)
axes[0].set_title("Share of qualified leads (A+B) by first-response time"); pct_axis(axes[0])
axes[1].bar(sla.index.astype(str), sla["C1"], color=PRIMARY, width=0.7)
axes[1].set_title("Lead → student (C1) by first-response time"); pct_axis(axes[1], decimals=1)
for ax, col in zip(axes, ["share_AB", "C1"]):
    for i, v in enumerate(sla[col]):
        ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=10, color=viz.TEXT_2)
plt.tight_layout(); save(fig, "02_sla"); plt.show()

# %%
from scipy.stats import chi2_contingency
fast = deals["sla_bucket"].eq("< 15 min")
slow = deals["sla_bucket"].isin(["1–4 h", "4–24 h", "> 24 h"])
sub = deals[fast | slow]
table = pd.crosstab(fast[fast | slow], sub["quality"].isin(AB))
chi2, p, *_ = chi2_contingency(table)
print(f"A+B share: <15 min = {sub[fast[fast | slow]]['quality'].isin(AB).mean():.1%}, "
      f"> 1 h = {sub[~fast[fast | slow]]['quality'].isin(AB).mean():.1%}; χ² = {chi2:.1f}, p = {p:.1e}")

# %% [markdown]
# **Key finding:** leads answered **within 15 minutes** are 1.8 times more likely to become
# qualified (A+B: 21% vs ~12%) and buy more often (C1 6.6% vs 4.6% when answered after more than a day).
# The difference is statistically significant (χ², p ≪ 0.05). This is observational data (correlation, not causation),
# so the effect has to be **tested with an A/B test** (notebook 03).
#
# ## 8. EDA summary and recommendations
#
# | # | Finding | Recommendation |
# |---|---|---|
# | 1 | C1 = 4.2%; 40% of lost deals are "doesn't answer / stopped answering" | speed up first contact, add messengers/SMS to calls |
# | 2 | Reply < 15 min → almost 2× more A+B leads | "Speed-to-lead" A/B test (notebook 03) |
# | 3 | Leads grew ×3, conversion fell from 5–6% to 3.5% | scale the sales team and SLA control together with the budget |
# | 4 | `performancemax_eng_DE`: €34k and 1 student; Google Ads CAC €344 | move budget to Facebook wide/LAL, webinars, brand search |
# | 5 | Evening groups convert almost twice as well | expand the evening format, launch an evening Web Developer group |
# | 6 | The core is Berlin and level B1 | geo and language targeting of ads |
# | 7 | 71% of campaign names cannot be matched to Spend, key fields are filled in manually | consistent UTM tagging and mandatory CRM fields |
#
# ➡️ Next step: [03_unit_economics_ab_test.ipynb](03_unit_economics_ab_test.ipynb)
