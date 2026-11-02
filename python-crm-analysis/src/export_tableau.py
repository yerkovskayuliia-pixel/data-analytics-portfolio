"""
Exports the cleaned CRM tables for Tableau into reports/tableau/ (one CSV per table, joined in the workbook):

- deals_tableau.csv           one row per lead (deal): funnel, revenue recognition, SLA, reach in 24 h
- contacts_tableau.csv        one row per contact            -> Deals.Contact ID = Contacts.Contact Key
- calls_by_contact.csv        calls aggregated per contact   -> Deals.Contact ID = Calls.Call Contact Key
- spend_by_source_month.csv   ad spend per source and month  -> Deals.Source + Month = Spend Source + Spend Month
Run: python src/export_tableau.py
"""
from pathlib import Path

import pandas as pd

import cleaning as etl

OUT = Path(__file__).resolve().parents[1] / "reports" / "tableau"


def build():
    d = etl.load_clean()
    deals, contacts, calls, spend = d["Deals"], d["Contacts"], d["Calls"], d["Spend"]
    conv = calls[calls["is_conversation"]].groupby("contact_id")["call_start_time"].min()
    lag = (deals["contact_id"].map(conv) - deals["created_time"]).dt.total_seconds()
    sla = pd.cut(deals["sla_sec"] / 3600, [0, .25, 1, 4, 24, 1e6],
                 labels=["1. < 15 min", "2. 15-60 min", "3. 1-4 h", "4. 4-24 h", "5. > 24 h"]).astype(str)
    pd.DataFrame({
        "Lead ID": deals["id"], "Contact ID": deals["contact_id"].fillna("no contact"),
        "Created Date": deals["created_time"].dt.strftime("%Y-%m-%d"), "Month": deals["month"].dt.strftime("%Y-%m"),
        "Source": deals["source"].astype(str), "Campaign": deals["campaign"].astype(str),
        "Product": deals["product"].astype(str), "Education Type": deals["education_type"].astype(str),
        "Deal Owner": deals["deal_owner_name"].astype(str), "Stage": deals["stage"].astype(str),
        "Lost Reason": deals["lost_reason"].astype(str), "Quality": deals["quality"].astype(str),
        "City": deals["city"].astype(str), "German Level": deals["german_level"].astype(str),
        "SLA Bucket": sla.replace("nan", "6. no SLA"),
        "Is Student": deals["is_student"].astype(int),
        "Is Qualified": deals["quality"].isin(["A - High", "B - Medium"]).astype(int),
        "Reached 24h": lag.between(-3600, 86400).astype(int),
        "Student Status": deals["student_status"].replace("", "not a student"),
        "Cash Received": deals["cash_received"].round(2), "Revenue": deals["revenue"].round(2),
        "Prepaid": deals["prepaid"].fillna(0).round(2), "Refund Due": deals["refund_due"].round(2),
    }).to_csv(OUT / "deals_tableau.csv", index=False, encoding="utf-8")

    pd.DataFrame({"Contact Key": contacts["id"], "Contact Owner": contacts["contact_owner_name"].astype(str),
                  "Contact Created": contacts["created_time"].dt.strftime("%Y-%m-%d")}
                 ).to_csv(OUT / "contacts_tableau.csv", index=False, encoding="utf-8")

    c = calls.dropna(subset=["contact_id"]).groupby("contact_id").agg(
        Calls=("id", "size"), Conversations=("is_conversation", "sum"), talk=("duration_sec", "sum"))
    pd.DataFrame({"Call Contact Key": c.index, "Calls": c["Calls"].values, "Conversations": c["Conversations"].values,
                  "Talk Minutes": (c["talk"] / 60).round(1).values}
                 ).to_csv(OUT / "calls_by_contact.csv", index=False, encoding="utf-8")

    s = spend.assign(m=spend["date"].dt.strftime("%Y-%m")).groupby([spend["source"].astype(str), "m"], observed=True) \
        .agg(spend=("spend", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum")).reset_index()
    s.columns = ["Spend Source", "Spend Month", "Month Spend", "Clicks", "Impressions"]
    s["Month Spend"] = s["Month Spend"].round(2)
    s.to_csv(OUT / "spend_by_source_month.csv", index=False, encoding="utf-8")
    print("exported:", {p.name: sum(1 for _ in open(p, encoding="utf-8")) - 1 for p in sorted(OUT.glob("*.csv"))})


if __name__ == "__main__":
    build()
