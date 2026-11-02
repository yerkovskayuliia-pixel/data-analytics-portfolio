# %% [markdown]
# # 03 · Product Analytics: Unit Economics, Metric Tree, Hypothesis and A/B Test
#
# Task: calculate unit economics by product → find growth levers → build a metric tree →
# formulate a hypothesis → design an A/B test lasting **no more than 2 weeks**.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

sys.path.insert(0, str(Path.cwd().parent / "src"))
import cleaning as etl
import viz
from viz import PRIMARY, ACCENT, GOOD, MUTED, save, pct_axis, eur_axis

viz.set_style()
pd.set_option("display.width", 200)

data = etl.load_clean()
deals, calls, spend = data["Deals"], data["Calls"], data["Spend"]
PRODUCTS = ["Digital Marketing", "UX/UI Design", "Web Developer"]

# %% [markdown]
# ## 1. Unit economics by product
#
# **Unit**: a lead (UA). Period: the whole dataset (07.2023–06.2024).
#
# | Metric | Formula | Meaning |
# |---|---|---|
# | UA | number of leads | User Acquisition |
# | B | students | Buyers |
# | C1 | B / UA | conversion to buyer |
# | T | number of payments (instalments: 1 payment per month studied; one-off: 1) | Transactions |
# | Revenue | recognised revenue: course price / duration × months studied | |
# | AOV | Revenue / T | average payment |
# | APC | T / B | average payments per student |
# | AC | ad budget | Acquisition Cost |
# | CPA | AC / UA | cost per lead |
# | CAC | AC / B | cost per student |
# | ARPU | Revenue / UA = C1 · AOV · APC | revenue per lead |
# | CLTV | AOV · APC | revenue per student |
# | CM | Revenue − AC | contribution margin (before teaching costs) |
# | ROMI | CM / AC | return on marketing investment |
#
# **Assumptions:**
# 1. 82% of leads have no product (they never reached a consultation). So leads and ad budget are allocated
#    to products **in proportion to each product's share among leads where it is known**. The alternative, counting
#    UA only for leads with a product, would inflate C1 five-fold and understate CPA.
# 2. Teaching costs (teacher salaries, platform) are not in the data → CM is the contribution margin
#    *after marketing*. It is an upper bound of the real margin.
# 3. Revenue is what was recognised **by the export date**, only for months studied (section 1.1): many students
#    are still studying, so Revenue and APC are understated for recent cohorts.

# %%
known = deals[deals["product"].isin(PRODUCTS)]
share = known["product"].value_counts(normalize=True).reindex(PRODUCTS)
UA_total, AC_total = len(deals), spend["spend"].sum()

stud = deals[deals["is_student"] & deals["product"].isin(PRODUCTS)]
ue = pd.DataFrame(index=PRODUCTS)
ue["UA"] = share * UA_total
ue["B"] = stud.groupby("product", observed=True).size().reindex(PRODUCTS)
ue["T"] = stud.groupby("product", observed=True)["payments_count"].sum().reindex(PRODUCTS)
ue["Revenue"] = stud.groupby("product", observed=True)["revenue"].sum().reindex(PRODUCTS)
ue["AC"] = share * AC_total


def add_ratios(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["C1"] = t["B"] / t["UA"]
    t["AOV"] = t["Revenue"] / t["T"]
    t["APC"] = t["T"] / t["B"]
    t["CPA"] = t["AC"] / t["UA"]
    t["CAC"] = t["AC"] / t["B"]
    t["ARPU"] = t["Revenue"] / t["UA"]
    t["CLTV"] = t["AOV"] * t["APC"]
    t["CM"] = t["Revenue"] - t["AC"]
    t["CM_per_UA"] = t["CM"] / t["UA"]
    t["ROMI"] = t["CM"] / t["AC"]
    return t


ue.loc["Total (3 products)"] = ue.sum()
ue = add_ratios(ue)
fmt = {"UA": "{:,.0f}", "B": "{:,.0f}", "T": "{:,.0f}", "Revenue": "€{:,.0f}", "AC": "€{:,.0f}",
       "C1": "{:.2%}", "AOV": "€{:,.0f}", "APC": "{:.2f}", "CPA": "€{:.2f}", "CAC": "€{:,.0f}",
       "ARPU": "€{:.0f}", "CLTV": "€{:,.0f}", "CM": "€{:,.0f}", "CM_per_UA": "€{:.0f}", "ROMI": "{:.0%}"}
ue.style.format(fmt)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
p = ue.loc[PRODUCTS]
for ax, col, title, f in [(axes[0], "C1", "C1: lead → student", "pct"),
                          (axes[1], "CLTV", "CLTV: revenue per student", "eur"),
                          (axes[2], "CM", "Contribution margin after marketing", "eur")]:
    ax.bar(p.index, p[col], color=PRIMARY, width=0.6)
    ax.set_title(title); ax.tick_params(axis="x", labelsize=9)
    pct_axis(ax, decimals=1) if f == "pct" else eur_axis(ax)
    for i, v in enumerate(p[col]):
        ax.text(i, v, f"{v:.1%}" if f == "pct" else f"€{v:,.0f}", ha="center", va="bottom", fontsize=9, color=viz.TEXT_2)
plt.tight_layout(); save(fig, "03_unit_economics"); plt.show()

# %% [markdown]
# **Reading the unit economics:**
# - All three products are profitable after marketing: CAC (≈ €180) is 14–26 times lower than CLTV. Marketing
#   does not limit the business right now; **conversion** does: out of 100 leads, ~4 buy.
# - **Digital Marketing** is the main profit source: the highest CLTV and the most students.
# - **Web Developer** has half the CLTV (the 6-month course is cheaper) at the same cost per lead.
# - APC ≈ 5 payments: students are on average halfway through the course, so revenue will keep growing as they study.

# %% [markdown]
# ### 1.1 Revenue recognition: cash received ≠ revenue earned
#
# Under German law (a tuition contract, § 611 BGB; the realisation principle, § 252 HGB) the school earns revenue
# only for the months a student has already studied. A prepayment for future months is a liability
# (Rechnungsabgrenzung, deferred revenue), not profit: if the student drops out, it has to be refunded.
#
# - `cash_received`: money received by the export date (first payment + monthly payments);
# - `revenue`: recognised revenue = course price / duration × months studied (used everywhere in the project);
# - `prepaid` = `cash_received` − `revenue` > 0: an advance for future months;
# - `refund_due`: the advance of students who dropped out, to be refunded.
#
# The CRM does not record a student's status, so it is **estimated**: finished = studied the whole course; dropped out =
# fewer months studied than have passed since payment, with a 1-month margin; 314 students have no payment date,
# so their status is unknown.

# %%
s = deals[deals["is_student"]]
rec = s.groupby("student_status").agg(students=("id", "size"), cash_received=("cash_received", "sum"),
                                      revenue=("revenue", "sum"), prepaid=("prepaid", "sum"),
                                      refund_due=("refund_due", "sum"))
rec.loc["Total"] = rec.sum()
rec.style.format({"students": "{:,.0f}", **{c: "€{:,.0f}" for c in ["cash_received", "revenue", "prepaid", "refund_due"]}})

# %% [markdown]
# **Finding:** of the €3.58M received, €3.34M is recognised as revenue. €244k is advances for future months, €200k of it
# one-off payments for the whole course. Students who dropped out (54 by this estimate) are owed about €14k back: that
# money must not be shown as profit. The remaining €231k becomes revenue as students study the months they paid for,
# or a refund too if they drop out. Conversely, 182 students have paid €10k less than they have studied (receivables).

# %% [markdown]
# ## 2. Growth levers: sensitivity analysis
# We raise each metric by 10% and see how CM changes. When UA grows, the ad budget grows
# proportionally (CPA is constant); raising C1, AOV or APC needs no extra budget.

# %%
base = ue.loc["Total (3 products)"]


def cm(UA, C1, AOV, APC, CPA):
    return UA * C1 * AOV * APC - UA * CPA


m0 = dict(UA=base["UA"], C1=base["C1"], AOV=base["AOV"], APC=base["APC"], CPA=base["CPA"])
cm0 = cm(**m0)
sens = {}
for k in ["UA", "C1", "AOV", "APC"]:
    m = m0.copy(); m[k] *= 1.10
    sens[k] = cm(**m) - cm0
m = m0.copy(); m["CPA"] *= 0.90
sens["CPA −10%"] = cm(**m) - cm0
sens = pd.Series(sens).rename({"UA": "UA +10%", "C1": "C1 +10%", "AOV": "AOV +10%", "APC": "APC +10%"})
sens_df = pd.DataFrame({"ΔCM, €": sens, "ΔCM, %": sens / cm0})
display(sens_df.style.format({"ΔCM, €": "€{:,.0f}", "ΔCM, %": "{:.1%}"}))

fig, ax = plt.subplots(figsize=(9, 3.8))
s = sens.sort_values()
ax.barh(s.index, s.values, color=[ACCENT if "C1" in i else PRIMARY for i in s.index], height=0.6)
eur_axis(ax, "x"); ax.set_title("Contribution margin uplift from a 10% improvement"); ax.grid(axis="y", visible=False)
for i, v in enumerate(s.values):
    ax.text(v, i, f"  +€{v:,.0f}", va="center", fontsize=10, color=viz.TEXT_2)
ax.set_xlim(0, s.max() * 1.25)
save(fig, "03_sensitivity"); plt.show()

# %% [markdown]
# **Choosing the growth lever.** C1, AOV and APC give the same CM uplift (+€334k, +10.5% for every +10%), traffic growth
# gives less (+€319k, because the budget grows with it), and cutting CPA by 10% adds only +€15k (marketing is already cheap
# relative to revenue: CAC €179 vs CLTV €3,996).
# Of the three equal levers we choose **C1**, because:
# 1. C1 has the most headroom: 96% of leads are lost, and 40% of lost deals are "doesn't answer / stopped answering";
# 2. AOV and APC are the price and length of study; changing them is risky (price affects demand, and APC
#    grows on its own as students keep studying);
# 3. the EDA showed a concrete mechanism for improving C1: **speed of first contact**.

# %% [markdown]
# ## 3. Metric tree

# %%
AB = ["A - High", "B - Medium"]
conv_calls = calls[calls["is_conversation"]].groupby("contact_id")["call_start_time"].min()
deals["first_talk"] = deals["contact_id"].map(conv_calls)
lag = (deals["first_talk"] - deals["created_time"]).dt.total_seconds()
deals["reached_24h"] = lag.between(-3600, 86400)
deals["is_AB"] = deals["quality"].isin(AB)

reached = deals["reached_24h"].mean()
ab_given_reached = deals.loc[deals["reached_24h"], "is_AB"].mean()
c1_ab = deals.loc[deals["is_AB"], "is_student"].mean()
c1_other = deals.loc[~deals["is_AB"], "is_student"].mean()
tree = f"""
CM (contribution margin) = €{base['CM']:,.0f}
├── Revenue = €{base['Revenue']:,.0f}
│   ├── UA (leads) = {base['UA']:,.0f}
│   │   ├── paid channels: budget AC = €{base['AC']:,.0f} / CPA €{base['CPA']:.2f}
│   │   └── free: Organic, CRM, Partnership
│   ├── C1 (lead → student) = {base['C1']:.2%}   ◀── GROWTH LEVER
│   │   ├── reach: conversation ≥30 s within 24 h = {reached:.1%}   ◀── targeted by the hypothesis
│   │   │   └── first-contact SLA (median {deals['sla_sec'].median() / 3600:.1f} h)
│   │   ├── qualification: A+B share among reached leads = {ab_given_reached:.1%}
│   │   └── closing: C1 of A+B leads = {c1_ab:.1%}, of the rest = {c1_other:.2%}
│   ├── AOV (average payment) = €{base['AOV']:,.0f}
│   └── APC (payments per student) = {base['APC']:.2f}
└── AC (marketing) = €{base['AC']:,.0f}
"""
print(tree)

# %% [markdown]
# The tree breaks CM down into operational metrics the sales team can influence every day.
# The hypothesis works through this chain: **SLA ↓ → reach ↑ → qualified leads ↑ → C1 ↑ → Revenue ↑ → CM ↑**.

# %% [markdown]
# ## 4. Hypothesis
#
# From the EDA (notebook 02, section 7):

# %%
deals["sla_bucket"] = pd.cut(deals["sla_sec"] / 3600, [0, 0.25, 1, 4, 24, 1e6],
                             labels=["< 15 min", "15–60 min", "1–4 h", "4–24 h", "> 24 h"])
evidence = deals.groupby("sla_bucket", observed=True).agg(
    leads=("id", "size"), reached_24h=("reached_24h", "mean"), share_AB=("is_AB", "mean"), C1=("is_student", "mean"))
evidence.style.format({"reached_24h": "{:.1%}", "share_AB": "{:.1%}", "C1": "{:.1%}"})

# %%
fig, ax = plt.subplots(figsize=(10, 4.2))
x = np.arange(len(evidence))
ax.plot(x, evidence["reached_24h"], marker="o", color=PRIMARY, label="reached within 24h")
ax.plot(x, evidence["share_AB"], marker="o", color=ACCENT, label="qualified (A+B)")
ax.set_xticks(x, evidence.index.astype(str)); pct_axis(ax)
for xi, (r, q) in enumerate(zip(evidence["reached_24h"], evidence["share_AB"])):
    ax.text(xi, r + 0.03, f"{r:.0%}", ha="center", fontsize=9, color=viz.TEXT_2)
    ax.text(xi, q + 0.03, f"{q:.0%}", ha="center", fontsize=9, color=viz.TEXT_2)
ax.set_title("The faster the first response, the more leads are reached and qualified")
ax.set_xlabel("time to first response (SLA)"); ax.legend(frameon=False); ax.set_ylim(0, 1)
save(fig, "03_hypothesis_evidence"); plt.show()

# %% [markdown]
# **A fair comparison.** Leads in the "> 24 h" bucket cannot, by construction, have a conversation within the first
# 24 hours (1.4% reached), so the fast response is compared only with responses between 15 minutes and 24 hours.
# We also check whether the gap is explained by the time of the request (night leads get slower answers) or by the source.

# %%
fast = deals["sla_bucket"].eq("< 15 min")
slow = deals["sla_bucket"].isin(["15–60 min", "1–4 h", "4–24 h"])
work = deals["created_time"].dt.hour.between(9, 17) & (deals["created_time"].dt.dayofweek < 5)
check = {"all leads": deals.index == deals.index, "working hours (Mon–Fri, 9–18)": work}
for s in ["Facebook Ads", "Google Ads"]:
    check[s] = deals["source"].eq(s)
robust = pd.DataFrame({
    name: {"< 15 min": deals.loc[mask & fast, "reached_24h"].mean(),
           "15 min – 24 h": deals.loc[mask & slow, "reached_24h"].mean()}
    for name, mask in check.items()}).T
robust["gap, pp"] = (robust["< 15 min"] - robust["15 min – 24 h"]) * 100
robust.style.format({"< 15 min": "{:.1%}", "15 min – 24 h": "{:.1%}", "gap, pp": "{:+.1f}"})

# %% [markdown]
# The 16–20 pp gap holds during working hours and within each major source, so it is not just night-time leads or
# the channel mix. It is still observational data, which is why an A/B test is needed.

# %% [markdown]
# > **Hypothesis (H1).** If every new lead is automatically assigned to an available manager within 1 minute
# > with a push notification, the client gets a WhatsApp/SMS message ("A manager will call you within 15 minutes"),
# > and the manager must call **within 15 minutes** during working hours,
# > **then the share of leads with a conversation in the first 24 hours will grow by at least 10 pp**
# > (from ~59% to ≥ 69%), **because** the client is still "warm" and expects the call. In the data, leads answered within 15 min
# > are reached 77% of the time vs 59% with a response between 15 minutes and 24 hours (an 18 pp gap,
# > 16 pp during working hours; we conservatively assume 10 pp).
#
# - **Product metric we influence:** C1 (through reach and qualification).
# - **Primary test metric:** `reached_24h`, the share of leads with a conversation of ≥ 30 seconds within 24 hours.
#   Chosen because (a) it is measurable within 1 day, while payment comes after 17+ days on average and cannot be
#   collected within 2 weeks; (b) it is directly linked to C1 (see the metric tree).
# - **Secondary metrics:** median SLA, A+B lead share, share of `Lost` with the reason "doesn't answer".
# - **Delayed metric:** C1 of the test cohorts, tracked 45 days after the end.
# - **Guardrail metrics (must not get worse):** share of `Non target` leads, calls per manager per day
#   (workload), share of complaints/unsubscribes from the messages.
# - **H0:** the `reached_24h` shares in groups A and B are equal.

# %% [markdown]
# ## 5. A/B test design: the duration is fixed at 14 days
#
# **The test was not run:** the CRM export has no experiment data, so below is the design only. The task limits a test
# to 2 weeks, so we start from the duration: fix 14 days → count the leads that arrive → compute the **MDE** (the
# smallest effect the test can detect at α = 0.05 and 80% power). The test makes sense only if the expected effect is
# at least the MDE.

# %%
recent = deals[deals["created_time"] >= "2024-03-01"]
n_days = recent["created_time"].dt.normalize().nunique()
leads_per_day = len(recent) / n_days
alpha, power, TEST_DAYS = 0.05, 0.80, 14


def mde_for(p_base: float, n_group: int) -> float:
    """MDE as a share for two groups of n_group (z-test for proportions, inverse of Cohen's h)."""
    h = NormalIndPower().solve_power(nobs1=n_group, alpha=alpha, power=power, ratio=1)
    return np.sin(np.arcsin(np.sqrt(p_base)) + h / 2) ** 2 - p_base


# control: leads answered between 15 min and 24 h (the "> 24 h" bucket is excluded, see section 4)
p_control = deals.loc[slow, "reached_24h"].mean()
n_14 = int(leads_per_day * TEST_DAYS / 2)
mde = mde_for(p_control, n_14)
expected = 0.10

design = pd.Series({
    "Test duration": f"{TEST_DAYS} days (2 full weeks, covers weekly seasonality)",
    "Leads per day (since March 2024)": f"{leads_per_day:.0f}",
    "Leads per group in 14 days (50/50)": f"{n_14:,}",
    "Baseline reached_24h (control)": f"{p_control:.1%}",
    "α / power": f"{alpha} / {power:.0%}",
    "MDE in 14 days": f"+{mde * 100:.1f} pp (from {p_control:.1%} to {p_control + mde:.1%})",
    "Expected effect (section 4)": f"+{expected * 100:.0f} pp",
    "Verdict": "expected effect ≥ MDE: a 14-day test makes sense" if expected >= mde
               else "expected effect < MDE: a 14-day test will not show it",
}, name="value")
design.to_frame()

# %% [markdown]
# **Test conditions:**
#
# | Parameter | Decision |
# |---|---|
# | Randomisation unit | lead (new request) |
# | Split | 50/50 by the last digit of the deal ID (even → B, odd → A): deterministic and checkable |
# | Group A (control) | the current lead assignment process |
# | Group B (test) | auto-assignment + push + automatic message to the client + the "call within 15 min" rule |
# | Who takes part | all new leads from all sources; the same managers work with both groups |
# | Duration | exactly 14 days, starting on a Monday; no early stopping (the peeking problem) |
# | Validity check | SRM test (chi-square) on group sizes; comparison of the source mix |
# | Statistical test | two-sided z-test for proportions, α = 0.05; result = difference in shares with a 95% confidence interval |
# | Decision | roll out if `reached_24h` grew significantly and guardrails did not get worse; confirm with C1 after 45 days |
#
# **Risks:** managers know about the test and may also call the control group faster (spillover) → the effect will be
# **underestimated**, i.e. the result is conservative. To reduce this, the "≤ 15 min" rule and notifications are shown
# to managers only on group B leads.

# %% [markdown]
# ## 6. Hypotheses for each product
#
# The product appears in the CRM only at the consultation, so the unit for product hypotheses is a lead with a known
# product. The 14-day metric is the **share of leads that reached payment or an invoice** (`Payment Done` or
# `Waiting For Payment`): payment itself cannot be awaited within 2 weeks (median cycle 17 days), while the client's
# decision is visible right after the consultation.

# %%
commit = deals["stage"].isin(["Payment Done", "Waiting For Payment"])
rows = {}
for prod in PRODUCTS:
    mask = deals["product"].eq(prod)
    per_day = (recent["product"] == prod).sum() / n_days
    n_group = int(per_day * TEST_DAYS / 2)
    base_p = commit[mask].mean()
    rows[prod] = {"leads with product per day": per_day, "leads per group in 14 days": n_group,
                  "baseline: paid or invoiced": base_p, "MDE in 14 days, pp": mde_for(base_p, n_group) * 100,
                  "evening: paid or invoiced": commit[mask & deals["education_type"].eq("Evening")].mean(),
                  "morning: paid or invoiced": commit[mask & deals["education_type"].eq("Morning")].mean(),
                  "share in Waiting For Payment": deals.loc[mask, "stage"].eq("Waiting For Payment").mean(),
                  "share 'changed decision / needs time'": deals.loc[mask, "lost_reason"].isin(
                      ["Changed Decision", "needs time to think"]).mean()}
products_ab = pd.DataFrame(rows).T
pooled_per_day = recent["product"].isin(PRODUCTS).sum() / n_days
pooled_mde = mde_for(commit[deals["product"].isin(PRODUCTS)].mean(), int(pooled_per_day * TEST_DAYS / 2))
print(f"All three products together: {pooled_per_day:.1f} leads a day, MDE in 14 days = +{pooled_mde * 100:.0f} pp")
products_ab.style.format({"leads with product per day": "{:.1f}", "leads per group in 14 days": "{:.0f}",
                          "MDE in 14 days, pp": "+{:.0f}", **{c: "{:.1%}" for c in products_ab.columns[[2, 4, 5, 6, 7]]}})

# %% [markdown]
# | Product | Hypothesis | Evidence in the data | Metric (14 days) | Expected effect | MDE in 14 days |
# |---|---|---|---|---|---|
# | **Digital Marketing** | If every client is offered an evening group on equal terms with a morning one at the consultation, the share reaching payment or an invoice will grow, because working people find evenings easier | evening 47.6% vs 32.9% morning (paid or invoiced); only 14% choose evening groups | paid or invoiced | ≈ +4 pp (if a third more morning clients pick evening) | +35 pp |
# | **UX/UI Design** | If within 24 h after the consultation the client gets an alumni case study and a free portfolio review, fewer leads will change their mind, because the doubt "will I find a job" is removed before the decision | "changed decision / needs time" — 21% of leads with the product | paid or invoiced | +5–10 pp (assumption, no data) | +41 pp |
# | **Web Developer** | If a client who received an invoice gets payment reminders after 24 and 72 h with an instalment option, more invoices will be paid, because deals currently stall at the payment stage | 14.3% of leads with the product are stuck in `Waiting For Payment`, more than any other product | paid or invoiced → share of invoices paid | +5–10 pp (assumption) | +43 pp |
#
# **Verdict on the product hypotheses.** Each product has only 3–4 leads with a known product a day, i.e. 19–30 leads
# per group in 14 days. With this traffic a 2-week test can only detect an effect of +35–43 pp, while a realistic
# effect is 5–10 pp. Even with all three products pooled, the MDE stays around +23 pp. Therefore:
# 1. **A product-level A/B test within 14 days is not possible**: there is not enough data, and the result would be
#    "no difference" whether or not the change works.
# 2. Product hypotheses should be checked as a **pilot** with a before/after comparison or once traffic grows; the A/B test
#    within 14 days runs on the all-lead metric (section 5), where the MDE of +8.4 pp is below the expected effect.
# 3. The website page (`page`) does not help: product pages bring only 1.6–3.2 leads a day. Product tests become possible
#    only when traffic grows or the product is recorded for every lead at sign-up.

# %% [markdown]
# ## 7. Expected business impact
# If the hypothesis is confirmed and the +10 pp in reach carries over to C1 proportionally
# (reached leads convert better than unreached ones), the effect on CM is:

# %%
c1_reached = deals.loc[deals["reached_24h"], "is_student"].mean()
c1_not = deals.loc[~deals["reached_24h"], "is_student"].mean()
c1_new = (reached + 0.10) * c1_reached + (1 - reached - 0.10) * c1_not
m = m0.copy(); m["C1"] = base["C1"] * c1_new / deals["is_student"].mean()
pd.Series({
    "C1 of leads reached within 24 h": f"{c1_reached:.1%}",
    "C1 of the rest": f"{c1_not:.1%}",
    "Expected C1 growth": f"{c1_new / deals['is_student'].mean() - 1:+.1%}",
    "Expected CM growth per year": f"+€{cm(**m) - cm0:,.0f}",
}).to_frame("estimate")

# %% [markdown]
# ## 8. Product analytics summary
# 1. All products are profitable after marketing; CAC €175–189 vs CLTV €2.5–4.5k (ROMI 13–25×). The bottleneck is conversion (C1 ≈ 4%).
# 2. The growth lever is **C1**: the most headroom and a clear improvement mechanism (40% of lost deals are "doesn't answer / stopped answering").
# 3. Metric tree: CM ← Revenue ← C1 ← reach ← first-contact SLA.
# 4. Hypothesis: auto-assigning leads + contact within 15 minutes will raise 24-hour reach by ≥ 10 pp.
# 5. A/B test (design only, no experiment data): the duration is fixed at 14 days, 519 leads per group, MDE = +8.4 pp vs an expected +10 pp, α = 0.05, power 80%, z-test for proportions; final check on C1
#    after 45 days.
# 6. Product hypotheses: Digital Marketing — evening group, UX/UI — alumni case and portfolio review after the
#    consultation, Web Developer — reminders and instalments for issued invoices. With 3–4 leads with a product a day the
#    14-day MDE is +35–43 pp, so product A/B tests within 2 weeks are not possible; they are checked as pilots.
