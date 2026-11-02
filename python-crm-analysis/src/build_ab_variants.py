"""
A/B test options for the product hypotheses, all limited to 14 days: reports/tableau/ab_variants.csv.

For each hypothesis the script computes the MDE (α = 0.05, power 80%) of several designs and compares it with the
expected effect, plus the number of days a classic 50/50 test would need. Run: python src/build_ab_variants.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

import cleaning as etl

OUT = Path(__file__).resolve().parents[1] / "reports" / "tableau" / "ab_variants.csv"
ALPHA, POWER, DAYS = 0.05, 0.80, 14
PRODUCTS = ["Digital Marketing", "UX/UI Design", "Web Developer"]


def mde(p, n1, ratio=1.0, alternative="two-sided"):
    """Smallest detectable lift (share) with n1 in the test group and n1 * ratio in control."""
    h = NormalIndPower().solve_power(nobs1=n1, alpha=ALPHA, power=POWER, ratio=ratio, alternative=alternative)
    angle = min(np.arcsin(np.sqrt(p)) + h / 2, np.pi / 2)  # capped: the lift cannot exceed 100%
    return np.sin(angle) ** 2 - p


def days_for(p, effect, per_day):
    n = NormalIndPower().solve_power(proportion_effectsize(p + effect, p), alpha=ALPHA, power=POWER, ratio=1)
    return int(np.ceil(2 * n / per_day))


def build():
    d = etl.load_clean()["Deals"]
    recent = d[d["created_time"] >= "2024-03-01"]
    n_days = recent["created_time"].dt.normalize().nunique()
    commit = d["stage"].isin(["Payment Done", "Waiting For Payment"])
    invoiced = commit
    known = d["product"].isin(PRODUCTS)

    hyps = [
        ("Digital Marketing", "Offer evening group",
         "Evening 47.6% vs morning 32.9%",
         "Paid or invoiced", d["product"].eq("Digital Marketing"), commit, known, commit, 0.04),
        ("UX/UI Design", "Alumni case + portfolio review",
         "21% changed their mind",
         "Paid or invoiced", d["product"].eq("UX/UI Design"), commit, known, commit, 0.075),
        ("Web Developer", "Payment reminders + instalments",
         "14.3% stuck in Waiting For Payment",
         "Invoices paid", d["product"].eq("Web Developer") & invoiced,
         d["stage"].eq("Payment Done"), invoiced, d["stage"].eq("Payment Done"), 0.075),
    ]
    rows = []
    for prod, hyp, evidence, metric, unit, success, pool_unit, pool_success, expected in hyps:
        base = success[unit].mean()
        per_day = recent.index.isin(d.index[unit]).sum() / n_days
        pool_base = pool_success[pool_unit].mean()
        pool_day = recent.index.isin(d.index[pool_unit]).sum() / n_days
        hist = int(unit.sum() - recent.index.isin(d.index[unit]).sum() * 0)  # all history as control
        n14 = per_day * DAYS
        variants = [
            ("1. Classic 50/50", int(n14 / 2), mde(base, int(n14 / 2)), base, per_day),
            ("2. One-sided test", int(n14 / 2), mde(base, int(n14 / 2), alternative="larger"),
             base, per_day),
            ("3. All 3 products, 50/50", int(pool_day * DAYS / 2), mde(pool_base, int(pool_day * DAYS / 2)),
             pool_base, pool_day),
            ("4. Historical control", int(n14), mde(base, int(n14), ratio=hist / n14), base, per_day),
        ]
        for name, n1, m, b, pdays in variants:
            rows.append({"Product": prod, "Hypothesis": hyp, "Evidence": evidence, "Metric": metric, "Variant": name,
                         "Leads per day": round(pdays, 1), "Test group size": n1, "Baseline": round(b, 4),
                         "MDE pp": round(m * 100, 1), "Expected effect pp": round(expected * 100, 1),
                         "Days for classic test": days_for(base, expected, per_day),
                         "Verdict": "detectable in 14 days" if expected >= m else "not detectable in 14 days"})
    # reference: the speed-to-lead test on all leads
    rows.append({"Product": "All leads", "Hypothesis": "Call within 15 minutes",
                 "Evidence": "Reach 77% vs 59%",
                 "Metric": "Reached in 24 h", "Variant": "1. Classic 50/50",
                 "Leads per day": round(len(recent) / n_days, 1), "Test group size": 519, "Baseline": 0.5935,
                 "MDE pp": 8.4, "Expected effect pp": 10.0, "Days for classic test": 10, "Verdict": "detectable in 14 days"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(out[["Product", "Variant", "Leads per day", "Test group size", "Baseline", "MDE pp", "Expected effect pp",
               "Days for classic test", "Verdict"]].to_string())


if __name__ == "__main__":
    build()
