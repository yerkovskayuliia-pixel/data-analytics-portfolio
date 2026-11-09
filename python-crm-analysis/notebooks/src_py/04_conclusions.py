# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 04 · Answer to the business question
#
# **Question.** The school spends ≈ €150k a year on ads and gets ≈ 20k leads, but only 4.2% pay.
# **Where are sales lost, and what is the cheapest lever to grow profit?**
#
# This notebook does not "discover" anything new: it puts the findings of notebooks 01–03 into one chain
# "question → evidence → decision" and compares the levers in euros. The numbers come from the data and from
# `reports/results_03.json` (saved by notebook 03), never retyped by hand.

# %% jupyter={"source_hidden": true} tags=["hide-input"]
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # pandas dtype backward compatibility, does not affect results
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent / "src"))
import cleaning as etl
import metrics as M
import viz
from viz import PRIMARY, ACCENT, MUTED, save, eur_axis

M.set_lang("en")
viz.set_style()
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)

data = etl.load_clean()
deals, calls, spend = data["Deals"], data["Calls"], data["Spend"]
d = M.add_contact_metrics(deals, calls)
R = json.loads((Path.cwd().parent / "reports" / "results_03.json").read_text(encoding="utf-8"))
fun = M.funnel(d, calls)

# %% [markdown]
# ## 1. Short answer
#
# 1. **Sales are lost at first contact and qualification, not on price.** 31% of leads never talk to a manager and only
#    17% reach a consultation. After the consultation the funnel works: 72% of issued invoices are paid.
# 2. **The cause: the sales team did not keep up with the traffic growth.** Leads tripled, the share of answers within
#    15 minutes fell from 21% to 3%, and conversion fell *within* channels, not because of a change in the channel mix.
# 3. **The cheapest lever is the speed of the first response.** It needs no ad budget; if an A/B test confirms it, it is
#    worth +€150–375k contribution margin a year. In parallel, switching off the Performance Max campaign saves
#    ≈ €33k without any test.

# %% [markdown]
# ## 2. Chain of evidence

# %%
lost = d[d["stage"].eq("Lost")]
no_answer = lost["lost_reason"].isin(["Doesn't Answer", "Stopped Answering"]).mean()
never_talked = 1 - fun.loc["Conversation ≥ 30 s", "% of leads"]
to_consult = fun.loc["Consultation (product set)", "% of leads"]
inv_paid = fun.loc["Student", "step conversion"]
evidence = pd.DataFrame([
    ("Where are sales lost?", f"{never_talked:.0%} of leads never had a conversation; {to_consult:.0%} reach a consultation; "
     f"{inv_paid:.0%} of invoices are paid", "02 · 1.3 funnel"),
    ("Why?", f"{no_answer:.0%} of lost deals are \"doesn't answer / stopped answering\"", "02 · 1.2 lost reasons"),
    ("Is it the traffic?", "no: C1 and the A+B share fell within channels (channel mix < 0.2 pp)", "02 · 2.3 shift-share"),
    ("What changed?", "answers < 15 min: 21% → 3%, median SLA 1 h → 5–7 h", "02 · 2.3"),
    ("Does speed matter?", f"reach 77% vs 59%; controlling for source, time, month and manager +{R['reach_gap_adjusted_pp']:.0f} pp",
     "02 · 7, 03 · 4 regression"),
    ("Which lever is stronger?", f"+10% C1 = +€{R['sensitivity_C1_plus10'] / 1e3:,.0f}k margin with no extra budget; −10% CPA = +€15k",
     "03 · 2 sensitivity"),
    ("Can it be tested in 14 days?", f"yes, if calls run {R['test_window']}: MDE +{R['test_mde_pp']:.1f} pp < effect "
     f"+{R['test_expected_pp']:.1f} pp; on weekdays 9–18 no (MDE +{R['office_hours_mde_pp']:.1f} > +{R['office_hours_expected_pp']:.1f})",
     "03 · 5 test design"),
], columns=["question", "what the data show", "where"]).set_index("question")
evidence

# %% [markdown]
# ## 3. Comparing the levers in euros
#
# | Lever | How ΔCM per year is estimated |
# |---|---|
# | Speed of the first response | scenarios of notebook 03 (section 7): rule compliance 80–100%, 50–100% of the C1 gap carried over |
# | Switch off Performance Max | the campaign's spend minus the revenue of its students (we lose them) |
# | Cut the whole budget by 10% | for comparison: savings minus the lost margin (CPA constant) |
#
# Evening groups and the product hypotheses are not valued in euros: with 3–4 leads with a product a day their effect
# cannot be measured reliably within 2 weeks (notebook 03, section 6), so they go into the plan as pilots.

# %%
pm = "performancemax_eng_DE"
pm_spend = spend.loc[spend["campaign"].eq(pm), "spend"].sum()
pm_deals = d[d["campaign"].eq(pm)]
pm_dcm = pm_spend - pm_deals["revenue"].sum()

g_spend = spend.loc[spend["source"].eq("Google Ads"), "spend"].sum()
g_st = d.loc[d["source"].eq("Google Ads"), "is_student"].sum()
fb_cac = spend.loc[spend["source"].eq("Facebook Ads"), "spend"].sum() / d.loc[d["source"].eq("Facebook Ads"), "is_student"].sum()
print(f"Performance Max: €{pm_spend:,.0f} spend, {len(pm_deals)} leads, {pm_deals['is_student'].sum()} student")
print(f"Google Ads CAC: with PMax €{g_spend / g_st:,.0f}, without PMax €{(g_spend - pm_spend) / (g_st - pm_deals['is_student'].sum()):,.0f} "
      f"(Facebook €{fb_cac:,.0f})")

cut10 = 0.10 * spend["spend"].sum() - 0.10 * (R["CM"] + spend["spend"].sum())   # −10% budget = −10% leads
levers = pd.DataFrame({
    "ΔCM min, €": [R["cm_uplift"]["conservative"], pm_dcm, cut10],
    "ΔCM max, €": [R["cm_uplift"]["optimistic"], pm_dcm, cut10],
    "needs ad budget": ["no", "no (saves money)", "no"],
    "how to check": ["14-day A/B test + C1 after 45 days", "right away, watch Google sign-ups for 2–4 weeks", "—"],
}, index=["Answer ≤ 15 min, 8:00–23:00", "Switch off Performance Max", "Cut the whole budget by 10%"])
levers.style.format({"ΔCM min, €": "€{:+,.0f}", "ΔCM max, €": "€{:+,.0f}"})

# %%
p = levers.iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 3.6))
y = np.arange(len(p))
mid = (p["ΔCM min, €"] + p["ΔCM max, €"]) / 2
ax.barh(y, mid, color=[MUTED, PRIMARY, ACCENT], height=0.55)
rng = p["ΔCM max, €"] > p["ΔCM min, €"]
ax.errorbar(mid[rng], y[rng.values], xerr=(p["ΔCM max, €"] - p["ΔCM min, €"])[rng] / 2, fmt="none", ecolor=viz.TEXT, capsize=6, lw=1.5)
k = lambda v: f"{'+' if v >= 0 else '−'}€{abs(v) / 1e3:,.0f}k"
for yi, (lo, hi) in zip(y, zip(p["ΔCM min, €"], p["ΔCM max, €"])):
    label = k(lo) if lo == hi else f"{k(lo)} … {k(hi)}"
    ax.text(max(hi, 0), yi, "  " + label, va="center", fontsize=10, color=viz.TEXT_2)
ax.axvline(0, color=viz.TEXT_2, lw=1)
ax.set_yticks(y, ["Cut all ads −10%", "Switch off Performance Max", "Answer ≤ 15 min, 8–23"])
eur_axis(ax, "x"); ax.set_xlim(min(0, p["ΔCM min, €"].min()) * 1.3, p["ΔCM max, €"].max() * 1.35)
ax.set_title("Contribution margin per year by lever"); ax.grid(axis="y", visible=False)
save(fig, "04_levers"); plt.show()

# %% [markdown]
# **Reading:** response speed is worth an order of magnitude more than the other levers and needs no ad budget, only
# evening and weekend manager shifts and automatic lead assignment. Hence the decision rule: **the rollout pays off if the
# extra shifts and the automation cost less than €150k a year** (the conservative scenario). Saving on ads gives little:
# marketing is already cheap (CAC €179 vs CLTV ≈ €4k), and an across-the-board budget cut loses more margin than it saves.
# The exception is clearly failing campaigns such as Performance Max.
#
# **Google Ads is not a bad channel.** The high Google Ads CAC (€344) comes almost entirely from Performance Max. Without
# that campaign Google Ads is cheaper than Facebook, so there is no need to move the whole budget from Google to Facebook.

# %% [markdown]
# ## 4. Action plan
#
# | When | Action | Success metric |
# |---|---|---|
# | Week 1 | Switch off `performancemax_eng_DE`; set up automatic lead assignment and the message to the client | spend −€34k/year; Google sign-ups did not drop |
# | Week 1 | 3-day technical pilot of the routing with 8:00–23:00 shifts | ≥ 80% of leads called within 15 min |
# | Weeks 2–3 | A/B test "answer ≤ 15 min" (design: notebook 03, section 5) | `reached_24h` +≥ 9.4 pp, guardrails fine |
# | +45 days | C1 check on the test cohorts; decision on a 100% rollout | C1 of group B above control |
# | In parallel | Product pilots: evening DM group, UX/UI alumni case, Web Dev payment reminders | share "paid or invoiced" before/after |
# | Ongoing | Mandatory CRM fields (product, payment type, closing date) and consistent UTM tagging | fill rate ≥ 95%; 100% of campaigns matched to Spend |
#
# ## 5. Limitations: what could change the answer
#
# - **Observational data.** The "speed → reach" link holds when source, time, month and manager are controlled for, but
#   it does not prove causation. Hence the recommendation: test first, then roll out.
# - **Carrying reach over into C1** is estimated from an observational gap (6.8% vs 2.3%), so the effect is given as a range.
# - **There is no teaching cost in the data**: CM is the margin after marketing, an upper bound of profit.
# - **Recent cohorts have not matured**: C1 for April–June 2024 is understated; the conclusion about the decline also
#   relies on the A+B share, which does not depend on this.
# - **Campaign tagging**: only 30% of campaign names in Deals appear in Spend, so CAC by campaign is an estimate.
# - **The product is known for 17% of leads**: unit economics by product spreads leads and budget proportionally (notebook 03).
