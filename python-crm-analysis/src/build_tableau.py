"""
Builds reports/tableau/crm_dashboard.twbx: a packaged Tableau workbook over reports/tableau/deals_tableau.csv.

Sheets: KPI strip, leads and C1 by month, C1 and CAC by source, speed to lead, revenue by product,
prepayments by student status, and one dashboard that puts them together.
Run after the CSV export: python src/build_tableau.py
"""
import uuid
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "tableau"
CSV = OUT_DIR / "deals_tableau.csv"
TWBX = OUT_DIR / "crm_dashboard.twbx"
DS = "federated.crm0deals0school01"
CONN = "textscan.crm0deals0csv0001"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"

STRING_DIMS = ["Lead ID", "Created Date", "Month", "Source", "Campaign", "Product", "Education Type", "Deal Owner",
               "Stage", "Lost Reason", "Quality", "City", "German Level", "SLA Bucket", "Student Status"]
INT_MEASURES = ["Is Student", "Is Qualified", "Reached 24h"]
REAL_MEASURES = ["Cash Received", "Revenue", "Prepaid", "Refund Due", "Ad Spend"]

CALCS = {  # name: (caption, formula, format)
    "Calc_Leads": ("Leads", "COUNT([Lead ID])", "n#,##0"),
    "Calc_Students": ("Students", "SUM([Is Student])", "n#,##0"),
    "Calc_C1": ("C1 (lead to student)", "SUM([Is Student]) / COUNT([Lead ID])", "p0.0%"),
    "Calc_CAC": ("CAC", "SUM([Ad Spend]) / SUM([Is Student])", "c\"€\"#,##0"),
    "Calc_Reach": ("Reached in 24h", "SUM([Reached 24h]) / COUNT([Lead ID])", "p0%"),
    "Calc_Qual": ("Qualified A+B", "SUM([Is Qualified]) / COUNT([Lead ID])", "p0%"),
}


def sid(name):
    """Stable <simple-id> per sheet; the same id goes on its window."""
    return f"<simple-id uuid='{{{str(uuid.uuid5(uuid.NAMESPACE_URL, 'crm-dashboard/' + name)).upper()}}}' />"


def q(s):
    return escape(s, {"'": "&apos;", '"': "&quot;"})


def column_decls(names):
    out = []
    for n in names:
        if n in CALCS:
            cap, f, fmt = CALCS[n]
            out.append(f"<column caption='{q(cap)}' datatype='real' default-format='{q(fmt)}' name='[{n}]' role='measure' type='quantitative'>"
                       f"<calculation class='tableau' formula='{q(f)}' /></column>")
        elif n in STRING_DIMS:
            out.append(f"<column datatype='string' name='[{n}]' role='dimension' type='nominal' />")
        elif n in INT_MEASURES:
            out.append(f"<column datatype='integer' name='[{n}]' role='measure' type='quantitative' />")
        else:
            out.append(f"<column datatype='real' default-format='c&quot;€&quot;#,##0' name='[{n}]' role='measure' type='quantitative' />")
    return out


def inst(field, how):
    """Column-instance name and XML for a field used on a sheet."""
    if how == "dim":
        return f"[none:{field}:nk]", f"<column-instance column='[{field}]' derivation='None' name='[none:{field}:nk]' pivot='key' type='nominal' />"
    if how == "sum":
        return f"[sum:{field}:qk]", f"<column-instance column='[{field}]' derivation='Sum' name='[sum:{field}:qk]' pivot='key' type='quantitative' />"
    if how == "cnt":
        return f"[cnt:{field}:qk]", f"<column-instance column='[{field}]' derivation='Count' name='[cnt:{field}:qk]' pivot='key' type='quantitative' />"
    return f"[usr:{field}:qk]", f"<column-instance column='[{field}]' derivation='User' name='[usr:{field}:qk]' pivot='key' type='quantitative' />"


def ref(name):
    return f"[{DS}].{name}"


def worksheet(name, rows, cols, mark, fields, color=None, text=None, label=None, sort=None, base_fields=(),
              members=None, min_value=None):
    """fields: list of (field, how). rows/cols: instance names or ''."""
    deps_cols = sorted({f for f, _ in fields} | set(base_fields))
    insts = [inst(f, h) for f, h in fields]
    # calcs reference base columns: declare them too
    for f, _ in fields:
        if f in CALCS:
            for base in STRING_DIMS + INT_MEASURES + REAL_MEASURES:
                if f"[{base}]" in CALCS[f][1] and base not in deps_cols:
                    deps_cols.append(base)
    deps = "".join(column_decls(deps_cols)) + "".join(x for _, x in insts)
    enc = ""
    if color:
        enc += f"<color column='{ref(color)}' />"
    if text:
        enc += "".join(f"<text column='{ref(t)}' />" for t in text)
    label_xml = ""
    if label:
        runs = "".join(label)
        label_xml = f"<customized-label><formatted-text>{runs}</formatted-text></customized-label>"
    sort_xml = ""
    if sort:
        dim, meas = sort
        sort_xml = (f"<computed-sort column='{ref(dim)}' direction='DESC' using='{ref(meas)}' />")
    filters = slices = ""
    if members:  # keep only these values of a dimension
        field, values = members
        col = ref(inst(field, "dim")[0])
        groups = "".join(f"<groupfilter function='member' level='{inst(field, 'dim')[0]}' member='{q(chr(34) + v + chr(34))}' />"
                         for v in values)
        filters += (f"<filter class='categorical' column='{col}'><groupfilter function='union' "
                    f"user:ui-domain='database' user:ui-enumeration='inclusive' user:ui-marker='enumerate'>{groups}</groupfilter></filter>")
    if min_value:  # hide marks whose aggregate is below a threshold
        measure, low = min_value
        filters += (f"<filter class='quantitative' column='{ref(measure)}' included-values='in-range'><min>{low}</min></filter>"
                    )
        slices = f"<slices><column>{ref(measure)}</column></slices>"
    color_style = ""
    if label:
        color_style = ("<style><style-rule element='cell'><format attr='text-align' value='center' />"
                       "<format attr='vertical-align' value='center' /></style-rule><style-rule element='mark'>"
                       "<format attr='mark-labels-show' value='true' /><format attr='mark-labels-cull' value='true' />"
                       "</style-rule></style>")
    elif not color and mark in ("Bar", "Line"):
        labels_on = "<format attr='mark-labels-show' value='true' />" if text else ""
        color_style = (f"<style><style-rule element='mark'><format attr='mark-color' value='{BLUE}' />{labels_on}"
                       "</style-rule></style>")
    return (f"<worksheet name='{q(name)}'><table><view>"
            f"<datasources><datasource caption='CRM deals' name='{DS}' /></datasources>"
            f"<datasource-dependencies datasource='{DS}'>{deps}</datasource-dependencies>"
            f"{filters}{sort_xml}{slices}<aggregation value='true' /></view><style />"
            f"<panes><pane selection-relaxation-option='selection-relaxation-allow'><view><breakdown value='auto' /></view>"
            f"<mark class='{mark}' />{f'<encodings>{enc}</encodings>' if enc else ''}{label_xml}{color_style}</pane></panes>"
            f"<rows>{ref(rows) if rows else ''}</rows><cols>{ref(cols) if cols else ''}</cols>"
            f"</table>{sid(name)}</worksheet>")


def run(text, bold=False, size=None, color=None):
    a = (" bold='true'" if bold else "") + (f" fontsize='{size}'" if size else "") + (f" fontcolor='{color}'" if color else "")
    return f"<run{a}>{text}</run>"


def field_run(name, size=16, color="#141820"):
    return f"<run bold='true' fontsize='{size}' fontcolor='{color}'><![CDATA[<{ref(name)}>]]></run>"


def build():
    df = pd.read_csv(CSV, nrows=5)
    assert list(df.columns) == STRING_DIMS[:13] + ["SLA Bucket", "Is Student", "Is Qualified", "Reached 24h",
                                                   "Student Status"] + REAL_MEASURES, list(df.columns)
    order = list(df.columns)

    rel_cols = "".join(
        f"<column datatype='{'string' if c in STRING_DIMS else 'integer' if c in INT_MEASURES else 'real'}' name='{q(c)}' ordinal='{i}' />"
        for i, c in enumerate(order))
    meta = []
    for i, c in enumerate(order):
        lt, rt, agg = (("string", 129, "Count") if c in STRING_DIMS else ("integer", 20, "Sum") if c in INT_MEASURES
                       else ("real", 5, "Sum"))
        meta.append(f"<metadata-record class='column'><remote-name>{q(c)}</remote-name><remote-type>{rt}</remote-type>"
                    f"<local-name>[{q(c)}]</local-name><parent-name>[deals_tableau.csv]</parent-name>"
                    f"<remote-alias>{q(c)}</remote-alias><ordinal>{i}</ordinal><local-type>{lt}</local-type>"
                    f"<aggregation>{agg}</aggregation><contains-null>true</contains-null></metadata-record>")
    calc_cols = "".join(column_decls(list(CALCS) + REAL_MEASURES))
    datasource = (
        f"<datasource caption='CRM deals' inline='true' name='{DS}' version='18.1'>"
        f"<connection class='federated'><named-connections><named-connection caption='deals_tableau' name='{CONN}'>"
        f"<connection class='textscan' directory='Data/tableau' filename='deals_tableau.csv' password='' server='' workgroup-auth-mode='as-is' />"
        f"</named-connection></named-connections>"
        f"<relation connection='{CONN}' name='deals_tableau.csv' table='[deals_tableau#csv]' type='table'>"
        f"<columns character-set='UTF-8' header='yes' locale='en_US' separator=','>{rel_cols}</columns></relation>"
        f"<metadata-records>{''.join(meta)}</metadata-records></connection>"
        f"<aliases enabled='yes' />{calc_cols}"
        f"<layout dim-ordering='alphabetic' measure-ordering='alphabetic' show-structure='true' />"
        f"<semantic-values><semantic-value key='[Country].[Name]' value='&quot;Germany&quot;' /></semantic-values>"
        f"</datasource>")

    leads, _ = inst("Calc_Leads", "usr")
    students, _ = inst("Calc_Students", "usr")
    c1, _ = inst("Calc_C1", "usr")
    cac, _ = inst("Calc_CAC", "usr")
    reach, _ = inst("Calc_Reach", "usr")
    qual, _ = inst("Calc_Qual", "usr")
    rev, cash, pre, refund = (inst(f, "sum")[0] for f in ("Revenue", "Cash Received", "Prepaid", "Refund Due"))

    kpis = [("Leads", "Calc_Leads", "usr", "#141820"), ("Students", "Calc_Students", "usr", "#141820"),
            ("C1 lead to student", "Calc_C1", "usr", BLUE), ("CAC", "Calc_CAC", "usr", "#141820"),
            ("Revenue, months studied", "Revenue", "sum", BLUE), ("Cash received", "Cash Received", "sum", "#141820"),
            ("Prepaid (liability)", "Prepaid", "sum", ORANGE), ("Refund due to dropouts", "Refund Due", "sum", ORANGE)]
    kpi_sheets = []
    for caption, field, how, colour in kpis:
        name = f"KPI {caption}"
        value = inst(field, how)[0]
        label = [run(caption, size=10, color="#4b5260"), run("Æ&#10;"), field_run(value, size=18, color=colour)]
        kpi_sheets.append((name, worksheet(name, "", "", "Text", [(field, how)], text=[value], label=label)))

    sheets = kpi_sheets + [
        ("Leads by Month", worksheet("Leads by Month", leads, inst("Month", "dim")[0], "Bar",
                                     [("Month", "dim"), ("Calc_Leads", "usr")])),
        ("C1 by Month", worksheet("C1 by Month", c1, inst("Month", "dim")[0], "Line",
                                  [("Month", "dim"), ("Calc_C1", "usr")])),
        ("C1 by Source", worksheet("C1 by Source", inst("Source", "dim")[0], c1, "Bar",
                                   [("Source", "dim"), ("Calc_C1", "usr")], text=[c1], sort=(inst("Source", "dim")[0], c1))),
        ("CAC by Source", worksheet("CAC by Source", inst("Source", "dim")[0], cac, "Bar",
                                    [("Source", "dim"), ("Calc_CAC", "usr")], text=[cac], sort=(inst("Source", "dim")[0], cac),
                                    min_value=(cac, 1))),
        ("Speed to Lead", worksheet("Speed to Lead", inst("SLA Bucket", "dim")[0], reach, "Bar",
                                    [("SLA Bucket", "dim"), ("Calc_Reach", "usr")], text=[reach],
                                    members=("SLA Bucket", ["1. < 15 min", "2. 15-60 min", "3. 1-4 h", "4. 4-24 h", "5. > 24 h"]))),
        ("Revenue by Product", worksheet("Revenue by Product", inst("Product", "dim")[0], rev, "Bar",
                                         [("Product", "dim"), ("Revenue", "sum")], text=[rev], sort=(inst("Product", "dim")[0], rev),
                                         members=("Product", ["Digital Marketing", "UX/UI Design", "Web Developer"]))),
        ("Prepaid by Status", worksheet("Prepaid by Status", inst("Student Status", "dim")[0], pre, "Bar",
                                        [("Student Status", "dim"), ("Prepaid", "sum")], text=[pre],
                                        sort=(inst("Student Status", "dim")[0], pre),
                                        members=("Student Status", ["active", "unknown", "dropout", "finished"]))),
    ]

    # dashboard: 1200 x 1000, coordinates in 1/100000 of the size
    W, H = 100000, 100000
    zid = iter(range(3, 100))

    def z(name, x, y, w, h):
        title_flag = "false" if name.startswith("KPI ") else "true"
        return f"<zone h='{h}' id='{next(zid)}' name='{q(name)}' show-title='{title_flag}' type-v2='visual' w='{w}' x='{x}' y='{y}' />"
    title = (f"<zone h='5000' id='{next(zid)}' type-v2='text' w='{W}' x='0' y='0'><formatted-text>"
             f"{run('CRM online school · leads, conversion and revenue recognised for months studied', bold=True, size=16)}"
             f"</formatted-text></zone>")
    half, row = W // 2, 26000
    kz = "".join(z(n, (i % 4) * (W // 4), 5000 + (i // 4) * 6000, W // 4, 6000) for i, (n, _) in enumerate(kpi_sheets))
    zones = (title + kz
             + z("Leads by Month", 0, 17000, half, row) + z("C1 by Month", half, 17000, half, row)
             + z("C1 by Source", 0, 17000 + row, half, row) + z("CAC by Source", half, 17000 + row, half, row)
             + z("Speed to Lead", 0, 17000 + 2 * row, W // 3, H - 17000 - 2 * row)
             + z("Revenue by Product", W // 3, 17000 + 2 * row, W // 3, H - 17000 - 2 * row)
             + z("Prepaid by Status", 2 * W // 3, 17000 + 2 * row, W - 2 * W // 3, H - 17000 - 2 * row))
    dashboard = (f"<dashboards><dashboard name='CRM Dashboard'><style />"
                 f"<size maxheight='1000' maxwidth='1200' minheight='1000' minwidth='1200' sizing-mode='fixed' />"
                 f"<datasources><datasource caption='CRM deals' name='{DS}' /></datasources>"
                 f"<zones><zone h='{H}' id='2' type-v2='layout-basic' w='{W}' x='0' y='0'>{zones}</zone></zones>"
                 f"{sid('CRM Dashboard')}</dashboard></dashboards>")
    windows = "".join(f"<window class='worksheet' name='{q(n)}'><cards><edge name='left'><strip size='160'>"
                      f"<card type='pages' /><card type='filters' /><card type='marks' /></strip></edge></cards>{sid(n)}</window>"
                      for n, _ in sheets)
    windows += ("<window class='dashboard' maximized='true' name='CRM Dashboard'><viewpoints>"
                + "".join(f"<viewpoint name='{q(n)}' />" for n, _ in sheets) + f"</viewpoints><active id='-1' />{sid('CRM Dashboard')}</window>")

    twb = ("<?xml version='1.0' encoding='utf-8' ?>\n"
           "<workbook original-version='18.1' source-build='2026.2.0 (20262.26.0603.1643)' source-platform='win' version='18.1' "
           "xmlns:user='http://www.tableausoftware.com/xml/user'>"
           "<document-format-change-manifest><SheetIdentifierTracking /><SortTagCleanup /><WindowsPersistSimpleIdentifiers />"
           "</document-format-change-manifest><preferences />"
           f"<datasources>{datasource}</datasources><worksheets>{''.join(x for _, x in sheets)}</worksheets>"
           f"{dashboard}<windows>{windows}</windows></workbook>")
    (OUT_DIR / "crm_dashboard.twb").write_text(twb, encoding="utf-8")
    with zipfile.ZipFile(TWBX, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("crm_dashboard.twb", twb)
        zf.write(CSV, "Data/tableau/deals_tableau.csv")
    print(f"Saved {TWBX} ({TWBX.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
