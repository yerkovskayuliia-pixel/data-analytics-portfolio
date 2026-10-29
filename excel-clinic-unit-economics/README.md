# Clinic Growth & Unit Economics (Excel)

Analysis of a dental clinic's operations over 15 months: client acquisition, doctor performance and profit. The goal was to recommend a strategy that maximises profit in quarter 5.

**Tools:** Excel (formulas, pivot tables, combo charts, dashboard) · Google Sheets

| File | |
|---|---|
| [`clinic_unit_economics.xlsx`](clinic_unit_economics.xlsx) | Workbook: data, calculations and dashboard |
| [`clinic_dashboard.pdf`](clinic_dashboard.pdf) | Dashboard as PDF, viewable without Excel |

## Headline numbers

| | |
|---|---|
| Clients acquired | 3,238 of 4,228 residents (77% know the clinic) |
| First profitable month | Period 12: +€3,382 |
| NPS, Q1 → Q4 | from −18% to 48% |
| Accumulated loss at end of Q4 | −€70,558 |

## Hypotheses and findings

**1. Service quality and doctor workload.** *Do all doctors perform equally well, and does the clinic need a fifth doctor?*
- NPS by doctor ranges from 60% (best) to 28–29% (two doctors in the risk zone), a gap of more than 2×.
- Only one doctor has a positive average service rating (+0.56). The doctor with the most visits (1,305) averages −0.46.
- Doctor utilisation is a steady 87–92%. Lost bookings fell to zero after the 3rd doctor (P7) and the 4th doctor (P10) were hired.
- **Recommendation:** service training for the two low-NPS doctors. Do not hire a fifth doctor.

**2. Profit growth without new clients.** *Can referrals replace paid advertising?*
- Acquisition was split almost evenly: 50.5% advertising and 49.5% referrals, but referrals cost €0.
- After advertising was switched off in Q3, new clients still grew: 611 in Q3, 650 in Q4.
- NPS growth is followed by more referrals in the next period.
- **Recommendation:** invest in service quality instead of advertising. Promoters (PS ≥ 9) bring new clients for free.

**3. Q5 profit forecast.** *What price increase would make the clinic profitable, and when does it break even?*
- Four scenarios for the average check (ACV): current, +5%, +10%, +15%.
- At +10%, monthly profit rises to about €17,800. The accumulated deficit shrinks by €53,363 over P13–P15, with full payback around period 16–17.
- Costs are mostly fixed: doctor salaries €37,900 (37%) and operating costs €28,000 (27%). Service costs are €34,426 (34%), or 34.8% of revenue.
- **Recommendation:** raise ACV by 10%, positioned as a premium service and backed by staff service training.

## Workbook structure

| Sheet | Contents |
|---|---|
| Dashboard Presentation | KPI tiles, 8 charts, conclusions for the three hypotheses |
| REPORT 2.3 | Period metrics: acquisition, retention, monetisation, costs and profit |
| BRANDING / NEEDS / REQUESTS / VISITS 2.3 | Source data: acquisition channel, client profile, booking requests, visits with doctor feedback |
| Pivot Table 6 | Supporting pivot |
| Research examples | Hypothesis backlog |

## Notes

- The workbook was built in Google Sheets and translated to English. Twelve Google-only formulas (`COUNTUNIQUE(FILTER(...))`) were replaced with Excel equivalents (`SUMPRODUCT(COUNTIFS(...))`). All 530 formulas recalculate to the same values in Excel.
- The dataset is a synthetic training case. Client and doctor names are generated and transliterated.
