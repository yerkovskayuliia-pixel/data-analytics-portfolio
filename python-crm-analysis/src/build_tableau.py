"""
Builds reports/tableau/crm_dashboard.twbx: a packaged Tableau workbook with two dashboards.

Data model (datasource "CRM model"): Deals ⟕ Contacts on Contact ID, ⟕ Calls (aggregated per contact) on Contact ID,
⟕ Spend (per source and month) on Source + Month. Every join is one-to-one per lead, so lead counts stay exact;
ad spend is spread over the leads of its source and month with a FIXED LOD.
Dashboard "CRM Dashboard": KPI cards, monthly leads and C1, C1 and CAC by source (click a source to filter),
speed to lead, revenue by product, prepayments by student status.
Dashboard "A/B Test Options": product hypotheses and the MDE of four 14-day designs (reports/tableau/ab_variants.csv).
Both datasources carry a Hyper extract (Data/Extracts/*.hyper), because Tableau Public only accepts extracts.
Run: python src/export_tableau.py && python src/build_ab_variants.py && python src/build_tableau.py
"""
import uuid
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from tableauhyperapi import (Connection, CreateMode, HyperProcess, Inserter, SqlType, TableDefinition, TableName,
                             Telemetry)

OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "tableau"
TWBX = OUT_DIR / "crm_dashboard.twbx"
DS_CRM, DS_AB = "federated.crm0model0school01", "federated.crm0ab0variants001"
CONN_CRM, CONN_AB = "textscan.crm0model0csv00001", "textscan.crm0ab0csv0000001"

# palette of the Flight Delays project: steel blue, light blue, orange accent, pale blue panel
BLUE, LIGHT, ORANGE, PANEL, INK, GREY = "#4e79a7", "#75a1c7", "#f28e2b", "#e7f1f5", "#1f3b57", "#6b7b8c"

TABLES = ["deals_tableau.csv", "contacts_tableau.csv", "calls_by_contact.csv", "spend_by_source_month.csv"]
INTEGER = {"Is Student", "Is Qualified", "Reached 24h", "Calls", "Conversations", "Clicks", "Impressions",
           "Test group size", "Days for classic test"}
REAL = {"Cash Received": "c&quot;€&quot;#,##0", "Revenue": "c&quot;€&quot;#,##0", "Prepaid": "c&quot;€&quot;#,##0",
        "Refund Due": "c&quot;€&quot;#,##0", "Month Spend": "c&quot;€&quot;#,##0", "Talk Minutes": "n#,##0",
        "Leads per day": "n#,##0.0", "Baseline": "p0.0%", "MDE pp": "n#,##0.0", "Expected effect pp": "n#,##0.0"}

CALCS = {  # name: (caption, formula, format)
    "Calc_AdSpend": ("Ad Spend (allocated)", "[Month Spend] / {FIXED [Source], [Month] : COUNT([Lead ID])}", "c\"€\"#,##0"),
    "Calc_Leads": ("Leads", "COUNT([Lead ID])", "n#,##0"),
    "Calc_Students": ("Students", "SUM([Is Student])", "n#,##0"),
    "Calc_C1": ("C1 (lead to student)", "SUM([Is Student]) / COUNT([Lead ID])", "p0.0%"),
    "Calc_CAC": ("CAC", "SUM([Calc_AdSpend]) / SUM([Is Student])", "c\"€\"#,##0"),
    "Calc_Reach": ("Reached in 24h", "SUM([Reached 24h]) / COUNT([Lead ID])", "p0%"),
    "Calc_Calls": ("Calls per lead", "SUM([Calls]) / COUNT([Lead ID])", "n#,##0.0"),
    "Calc_Talk": ("Talk minutes per lead", "SUM([Talk Minutes]) / COUNT([Lead ID])", "n#,##0.0"),
}

DS = DS_CRM  # datasource the sheet builders write to; switched while building the A/B sheets
COLUMNS = {}  # column name -> datatype, filled from the CSV headers
CAPTIONS = {DS_CRM: "CRM model", DS_AB: "A/B test options"}


def q(s):
    return escape(s, {"'": "&apos;", '"': "&quot;"})


def sid(name):
    return f"<simple-id uuid='{{{str(uuid.uuid5(uuid.NAMESPACE_URL, 'crm-dashboard/' + name)).upper()}}}' />"


def dtype(c):
    return "integer" if c in INTEGER else "real" if c in REAL else "string"


def column_decl(n):
    if n in CALCS:
        cap, f, fmt = CALCS[n]
        return (f"<column caption='{q(cap)}' datatype='real' default-format='{q(fmt)}' name='[{n}]' role='measure' "
                f"type='quantitative'><calculation class='tableau' formula='{q(f)}' /></column>")
    t = dtype(n)
    if t == "string":
        return f"<column datatype='string' name='[{q(n)}]' role='dimension' type='nominal' />"
    fmt = f" default-format='{REAL[n]}'" if n in REAL else ""
    return f"<column datatype='{t}'{fmt} name='[{q(n)}]' role='measure' type='quantitative' />"


def depends(names):
    """Columns a set of fields needs, following calculated fields recursively."""
    out, todo = [], list(names)
    while todo:
        n = todo.pop()
        if n in out:
            continue
        out.append(n)
        if n in CALCS:
            todo += [c for c in list(COLUMNS) + list(CALCS) if f"[{c}]" in CALCS[n][1]]
    return sorted(out)


def inst(field, how):
    deriv, pre, kind, t = {"dim": ("None", "none", "nk", "nominal"), "sum": ("Sum", "sum", "qk", "quantitative"),
                           "avg": ("Avg", "avg", "qk", "quantitative"), "usr": ("User", "usr", "qk", "quantitative")}[how]
    name = f"[{pre}:{field}:{kind}]"
    return name, (f"<column-instance column='[{q(field)}]' derivation='{deriv}' name='{q(name)}' pivot='key' "
                  f"type='{t}' />")


def ref(name):
    return f"[{DS}].{name}"


def shelf(fields):
    if not fields:
        return ""
    if isinstance(fields, str):
        return ref(fields)
    return "(" + " / ".join(ref(f) for f in fields) + ")"


def run(text, bold=False, size=None, color=None):
    a = (" bold='true'" if bold else "") + (f" fontsize='{size}'" if size else "") + (f" fontcolor='{color}'" if color else "")
    # each line break is a run of its own; the style repeats on every line
    return "<run>Æ&#10;</run>".join(f"<run{a}>{part}</run>" for part in text.split(BR)).replace(f"<run{a}></run>", "")


def field_run(name, size=16, color=INK, bold=True):
    b = " bold='true'" if bold else ""
    return f"<run{b} fontsize='{size}' fontcolor='{color}'><![CDATA[<{ref(name)}>]]></run>"


BR = "\n"  # marker for a line break inside dashboard text; becomes its own run

CLEAN_AXES = ("<style-rule element='gridline'><format attr='line-visibility' scope='cols' value='off' />"
              "<format attr='line-visibility' scope='rows' value='off' /></style-rule>"
              "<style-rule element='zeroline'><format attr='line-visibility' value='off' /></style-rule>"
              "<style-rule element='axis'><format attr='tick-color' value='#00000000' /></style-rule>")


def worksheet(name, rows, cols, mark, fields, title=None, color=None, palette=None, text=None, label=None, sort=None,
              members=None, min_value=None, mark_color=BLUE):
    insts = [inst(f, h) for f, h in fields]
    deps = "".join(column_decl(c) for c in depends({f for f, _ in fields})) + "".join(x for _, x in insts)
    enc = (f"<color column='{ref(color)}' />" if color else "") + "".join(f"<text column='{ref(t)}' />" for t in text or [])
    label_xml = f"<customized-label><formatted-text>{''.join(label)}</formatted-text></customized-label>" if label else ""
    sort_xml = f"<computed-sort column='{ref(sort[0])}' direction='DESC' using='{ref(sort[1])}' />" if sort else ""
    filters = slices = ""
    if members:
        field, values = members
        lvl = inst(field, "dim")[0]
        groups = "".join(f"<groupfilter function='member' level='{lvl}' member='{q(chr(34) + v + chr(34))}' />" for v in values)
        filters += (f"<filter class='categorical' column='{ref(lvl)}'><groupfilter function='union' user:ui-domain='database' "
                    f"user:ui-enumeration='inclusive' user:ui-marker='enumerate'>{groups}</groupfilter></filter>")
    if min_value:
        filters += f"<filter class='quantitative' column='{ref(min_value[0])}' included-values='in-range'><min>{min_value[1]}</min></filter>"
        slices = f"<slices><column>{ref(min_value[0])}</column></slices>"
    # table style: colour encoding (sequential palette or categorical map) and quiet axes
    table_rules = ""
    if color and isinstance(palette, str):
        table_rules += (f"<style-rule element='mark'><encoding attr='color' field='{ref(color)}' palette='{palette}' "
                        f"type='interpolated' /></style-rule>")
    elif color and isinstance(palette, dict):
        maps = "".join(f"<map to='{c}'><bucket>&quot;{q(v)}&quot;</bucket></map>" for v, c in palette.items())
        table_rules += f"<style-rule element='mark'><encoding attr='color' field='{ref(color)}' type='palette'>{maps}</encoding></style-rule>"
    if mark in ("Bar", "Line"):
        table_rules += CLEAN_AXES
    table_style = f"<style>{table_rules}</style>" if table_rules else "<style />"
    if mark == "Text":
        pane_rules = ("<style-rule element='cell'><format attr='text-align' value='center' /><format attr='vertical-align' value='center' />"
                      "</style-rule><style-rule element='mark'><format attr='mark-labels-show' value='true' />"
                      "<format attr='mark-labels-cull' value='true' /></style-rule>")
    else:
        colour = "" if color else f"<format attr='mark-color' value='{mark_color}' />"
        shown = "<format attr='mark-labels-show' value='true' />" if text else ""
        pane_rules = f"<style-rule element='mark'>{colour}{shown}</style-rule>" if colour or shown else ""
    pane_style = f"<style>{pane_rules}</style>" if pane_rules else ""
    title_xml = (f"<layout-options><title><formatted-text>{run(q(title), size=13, color=BLUE)}</formatted-text></title></layout-options>"
                 if title else "")
    return (f"<worksheet name='{q(name)}'>{title_xml}<table><view>"
            f"<datasources><datasource caption='{CAPTIONS[DS]}' name='{DS}' /></datasources>"
            f"<datasource-dependencies datasource='{DS}'>{deps}</datasource-dependencies>"
            f"{filters}{sort_xml}{slices}<aggregation value='true' /></view>{table_style}"
            f"<panes><pane selection-relaxation-option='selection-relaxation-allow'><view><breakdown value='auto' /></view>"
            f"<mark class='{mark}' />{f'<encodings>{enc}</encodings>' if enc else ''}{label_xml}{pane_style}</pane></panes>"
            f"<rows>{shelf(rows)}</rows><cols>{shelf(cols)}</cols></table>{sid(name)}</worksheet>")


def relation(table, conn):
    cols = list(pd.read_csv(OUT_DIR / table, nrows=3).columns)
    for c in cols:
        COLUMNS[c] = dtype(c)
    body = "".join(f"<column datatype='{dtype(c)}' name='{q(c)}' ordinal='{i}' />" for i, c in enumerate(cols))
    rel = (f"<relation connection='{conn}' name='{table}' table='[{table.replace('.', '#')}]' type='table'>"
           f"<columns character-set='UTF-8' header='yes' locale='en_US' separator=','>{body}</columns></relation>")
    remote = {"string": 129, "integer": 20, "real": 5}
    meta = "".join(
        f"<metadata-record class='column'><remote-name>{q(c)}</remote-name><remote-type>{remote[dtype(c)]}</remote-type>"
        f"<local-name>[{q(c)}]</local-name><parent-name>[{table}]</parent-name><remote-alias>{q(c)}</remote-alias>"
        f"<ordinal>{i}</ordinal><local-type>{dtype(c)}</local-type>"
        f"<aggregation>{'Count' if dtype(c) == 'string' else 'Sum'}</aggregation>"
        f"<contains-null>true</contains-null></metadata-record>" for i, c in enumerate(cols))
    return rel, meta


def eq(left_table, left, right_table, right):
    return (f"<expression op='='><expression op='[{left_table}].[{left}]' />"
            f"<expression op='[{right_table}].[{right}]' /></expression>")


def join(left, right, condition):
    return f"<relation join='left' type='join'><clause type='join'>{condition}</clause>{left}{right}</relation>"


def write_hyper(df, path):
    """Writes df to a Hyper file as table Extract.Extract (the layout Tableau uses for its own extracts)."""
    sql = {"string": SqlType.text(), "integer": SqlType.big_int(), "real": SqlType.double()}
    table = TableName("Extract", "Extract")
    definition = TableDefinition(table, [TableDefinition.Column(c, sql[dtype(c)]) for c in df.columns])
    rows = [[None if pd.isna(v) else (int(v) if dtype(c) == "integer" else float(v) if dtype(c) == "real" else str(v))
             for c, v in zip(df.columns, r)] for r in df.itertuples(index=False)]
    path.unlink(missing_ok=True)
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(hyper.endpoint, str(path), CreateMode.CREATE_AND_REPLACE) as con:
            con.catalog.create_schema("Extract")
            con.catalog.create_table(definition)
            with Inserter(con, definition) as ins:
                ins.add_rows(rows)
                ins.execute()


def extract_xml(df, dbname):
    """<extract> element pointing at a packaged Hyper file whose columns keep the datasource's local names."""
    remote = {"string": 129, "integer": 20, "real": 5}
    maps = "".join(f"<map key='[{q(c)}]' value='[Extract].[{q(c)}]' />" for c in df.columns)
    meta = "".join(
        f"<metadata-record class='column'><remote-name>{q(c)}</remote-name><remote-type>{remote[dtype(c)]}</remote-type>"
        f"<local-name>[{q(c)}]</local-name><parent-name>[Extract]</parent-name><remote-alias>{q(c)}</remote-alias>"
        f"<ordinal>{i}</ordinal><local-type>{dtype(c)}</local-type>"
        f"<aggregation>{'Count' if dtype(c) == 'string' else 'Sum'}</aggregation>"
        f"<contains-null>true</contains-null></metadata-record>" for i, c in enumerate(df.columns))
    return (f"<extract count='-1' enabled='true' units='records'>"
            f"<connection access_mode='readonly' author-locale='en_US' class='hyper' dbname='{dbname}' "
            f"default-settings='hyper' schema='Extract' sslmode='' tablename='Extract' "
            f"update-time='{pd.Timestamp.now():%m/%d/%Y %I:%M:%S %p}' username='tableau_internal_user'>"
            f"<relation name='Extract' table='[Extract].[Extract]' type='table' /><cols>{maps}</cols>"
            f"<metadata-records>{meta}</metadata-records></connection></extract>")


def read_typed(table):
    df = pd.read_csv(OUT_DIR / table, dtype=str, keep_default_na=False, na_values=[""])
    for c in df.columns:
        if dtype(c) != "string":
            df[c] = pd.to_numeric(df[c])
    return df


def datasource_xml(ds, conn, caption, filename, relation_xml, meta, extra="", extract=""):
    return (f"<datasource caption='{caption}' inline='true' name='{ds}' version='18.1'>"
            f"<connection class='federated'><named-connections><named-connection caption='{caption}' name='{conn}'>"
            f"<connection class='textscan' directory='Data/tableau' filename='{filename}' password='' server='' "
            f"workgroup-auth-mode='as-is' /></named-connection></named-connections>{relation_xml}"
            f"<metadata-records>{meta}</metadata-records></connection><aliases enabled='yes' />{extra}{extract}"
            f"<layout dim-ordering='alphabetic' dim-percentage='0.5' measure-ordering='alphabetic' measure-percentage='0.4' show-structure='true' />"
            f"<semantic-values><semantic-value key='[Country].[Name]' value='&quot;Germany&quot;' /></semantic-values>"
            f"</datasource>")


def build():
    global DS
    # ---- data model: deals ⟕ contacts ⟕ calls per contact ⟕ spend per source and month
    rels, metas = {}, []
    for table in TABLES:
        rels[table], m = relation(table, CONN_CRM)
        metas.append(m)
    d, c, k, s = TABLES
    model = join(join(join(rels[d], rels[c], eq(d, "Contact ID", c, "Contact Key")),
                      rels[k], eq(d, "Contact ID", k, "Call Contact Key")),
                 rels[s], "<expression op='AND'>" + eq(d, "Source", s, "Spend Source") + eq(d, "Month", s, "Spend Month")
                 + "</expression>")
    # extracts: the same left joins in pandas, one row per lead
    flat = (read_typed(d)
            .merge(read_typed(c), how="left", left_on="Contact ID", right_on="Contact Key")
            .merge(read_typed(k), how="left", left_on="Contact ID", right_on="Call Contact Key")
            .merge(read_typed(s), how="left", left_on=["Source", "Month"], right_on=["Spend Source", "Spend Month"]))
    assert len(flat) == len(read_typed(d)), "joins must stay one row per lead"
    ab_flat = read_typed("ab_variants.csv")
    extracts = {"Data/Extracts/crm_model.hyper": flat, "Data/Extracts/ab_options.hyper": ab_flat}
    for name, frame in extracts.items():
        write_hyper(frame, OUT_DIR / Path(name).name)

    crm_ds = datasource_xml(DS_CRM, CONN_CRM, "CRM model", d, model, "".join(metas),
                            "".join(column_decl(x) for x in list(CALCS) + [x for x in COLUMNS if x in REAL]),
                            extract_xml(flat, "Data/Extracts/crm_model.hyper"))
    ab_rel, ab_meta = relation("ab_variants.csv", CONN_AB)
    ab_ds = datasource_xml(DS_AB, CONN_AB, "A/B test options", "ab_variants.csv", ab_rel, ab_meta,
                           "".join(column_decl(x) for x in ("Leads per day", "Baseline", "MDE pp", "Expected effect pp")),
                           extract_xml(ab_flat, "Data/Extracts/ab_options.hyper"))

    leads, c1, cac, reach = (inst(f, "usr")[0] for f in ("Calc_Leads", "Calc_C1", "Calc_CAC", "Calc_Reach"))
    rev, pre = inst("Revenue", "sum")[0], inst("Prepaid", "sum")[0]
    month, source, sla, product, status = (inst(f, "dim")[0] for f in ("Month", "Source", "SLA Bucket", "Product", "Student Status"))

    kpis = [("Leads", "Calc_Leads", "usr", INK), ("Students", "Calc_Students", "usr", INK),
            ("C1 lead → student", "Calc_C1", "usr", BLUE), ("CAC", "Calc_CAC", "usr", INK),
            ("Calls per lead", "Calc_Calls", "usr", INK),
            ("Revenue earned", "Revenue", "sum", BLUE), ("Cash received", "Cash Received", "sum", INK),
            ("Prepaid (liability)", "Prepaid", "sum", ORANGE), ("Refund due", "Refund Due", "sum", ORANGE),
            ("Talk minutes per lead", "Calc_Talk", "usr", INK)]
    kpi_sheets = []
    for caption, field, how, colour in kpis:
        name, value = f"KPI {caption}", inst(field, how)[0]
        label = [run(q(caption), size=11, color=BLUE), run("Æ&#10;"), field_run(value, size=16, color=colour)]
        kpi_sheets.append((name, worksheet(name, "", "", "Text", [(field, how)], text=[value], label=label)))

    sheets = kpi_sheets + [
        ("Leads by Month", worksheet("Leads by Month", leads, month, "Bar", [("Month", "dim"), ("Calc_Leads", "usr"), ("Calc_C1", "usr")],
                                     title="Leads per month · colour = conversion", color=c1, palette="blue_10_0")),
        ("C1 by Month", worksheet("C1 by Month", c1, month, "Line", [("Month", "dim"), ("Calc_C1", "usr")],
                                  title="Lead → student conversion by month", mark_color=ORANGE)),
        ("C1 by Source", worksheet("C1 by Source", source, c1, "Bar", [("Source", "dim"), ("Calc_C1", "usr")],
                                   title="Conversion by source · click a bar to filter", color=c1, palette="blue_10_0",
                                   text=[c1], sort=(source, c1))),
        ("CAC by Source", worksheet("CAC by Source", source, cac, "Bar", [("Source", "dim"), ("Calc_CAC", "usr")],
                                    title="Cost per student (CAC) by paid source", color=cac, palette="orange_10_0",
                                    text=[cac], sort=(source, cac), min_value=(cac, 1))),
        ("Speed to Lead", worksheet("Speed to Lead", sla, reach, "Bar", [("SLA Bucket", "dim"), ("Calc_Reach", "usr")],
                                    title="Reached in 24 h by first-response time", color=reach, palette="blue_10_0",
                                    text=[reach], members=("SLA Bucket", ["1. < 15 min", "2. 15-60 min", "3. 1-4 h", "4. 4-24 h", "5. > 24 h"]))),
        ("Revenue by Product", worksheet("Revenue by Product", product, rev, "Bar", [("Product", "dim"), ("Revenue", "sum")],
                                         title="Revenue for months studied", color=rev, palette="blue_10_0", text=[rev],
                                         sort=(product, rev), members=("Product", ["Digital Marketing", "UX/UI Design", "Web Developer"]))),
        ("Prepaid by Status", worksheet("Prepaid by Status", status, pre, "Bar", [("Student Status", "dim"), ("Prepaid", "sum")],
                                        title="Prepayments (liability) by student status", color=pre, palette="orange_10_0",
                                        text=[pre], sort=(status, pre), members=("Student Status", ["active", "unknown", "dropout", "finished"]))),
    ]

    DS = DS_AB
    prod_d, hyp_d, ev_d, met_d, var_d, verdict_d = (inst(f, "dim")[0] for f in
                                                    ("Product", "Hypothesis", "Evidence", "Metric", "Variant", "Verdict"))
    mde_v, exp_v, days_v = (inst(f, "avg")[0] for f in ("MDE pp", "Expected effect pp", "Days for classic test"))
    ab_label = [run("MDE "), field_run(mde_v, size=9), run(" pp · expected "), field_run(exp_v, size=9, bold=False), run(" pp")]
    hyp_label = [run("expected "), field_run(exp_v, size=10), run(" pp · a classic 50/50 test needs "),
                 field_run(days_v, size=10), run(" days")]
    ab_sheets = [
        ("Product Hypotheses", worksheet("Product Hypotheses", [prod_d, hyp_d, ev_d, met_d], "", "Text",
                                         [(f, "dim") for f in ("Product", "Hypothesis", "Evidence", "Metric")]
                                         + [("Expected effect pp", "avg"), ("Days for classic test", "avg")],
                                         title="One hypothesis per product", text=[exp_v, days_v], label=hyp_label)),
        ("A/B Options: MDE in 14 days", worksheet("A/B Options: MDE in 14 days", [prod_d, var_d], mde_v, "Bar",
                                                  [("Product", "dim"), ("Variant", "dim"), ("Verdict", "dim"),
                                                   ("MDE pp", "avg"), ("Expected effect pp", "avg")],
                                                  title="Smallest effect a 14-day test can detect (MDE), by design",
                                                  color=verdict_d, palette={"detectable in 14 days": BLUE,
                                                                            "not detectable in 14 days": ORANGE},
                                                  text=[mde_v, exp_v], label=ab_label)),
    ]
    DS = DS_CRM

    # ---- dashboards: 1200 x 1000 px, zone coordinates in 1/100000 of the size
    W, H = 100000, 100000
    zid = iter(range(3, 200))

    def style(border=None, fill=None, margin=4):
        f = ((f"<format attr='border-color' value='{border}' /><format attr='border-style' value='solid' />"
              f"<format attr='border-width' value='1' />" if border else "")
             + (f"<format attr='background-color' value='{fill}' />" if fill else "")
             + f"<format attr='margin' value='{margin}' />")
        return f"<zone-style>{f}</zone-style>"

    def vis(name, x, y, w, h, border=None):
        show = "false" if name.startswith("KPI ") else "true"
        return (f"<zone h='{h}' id='{next(zid)}' name='{q(name)}' show-title='{show}' type-v2='visual' w='{w}' x='{x}' y='{y}'>"
                f"{style(border)}</zone>")

    text_ids = []

    def txt(x, y, w, h, runs, border=None, fill=None):
        i = next(zid)
        text_ids.append(i)
        return (f"<zone h='{h}' id='{i}' type-v2='text' w='{w}' x='{x}' y='{y}'><formatted-text>{runs}</formatted-text>"
                f"{style(border, fill, 8)}</zone>")

    side, gap = 18000, 1000
    x0, cw = side + gap, W - side - gap
    br = BR
    insights = (run("Key insights", bold=True, size=13, color=BLUE) + run(br)
                + run("• 4.2% of leads become students; 31% never had a conversation, only 17% reach a consultation." + br, size=10, color=INK)
                + run("• Answer within 15 min: 77% reached in 24 h vs 59% later." + br, size=10, color=INK)
                + run("• Leads tripled, answers within 15 min fell from 21% to 3%, conversion fell inside every channel." + br, size=10, color=INK)
                + run("• Revenue earned counts only months studied: €244k is prepayment, ≈€14k is owed to dropouts." + br + br,
                      size=10, color=INK)
                + run("Data model", bold=True, size=13, color=BLUE) + run(br)
                + run("Deals ⟕ Contacts on Contact ID" + br + "Deals ⟕ Calls per contact on Contact ID" + br
                      + "Deals ⟕ Spend on Source + Month" + br
                      + "Ad spend is spread over the leads of each source-month (FIXED LOD)." + br + br, size=10, color=GREY)
                + run("Tip: click a source in “Conversion by source” to filter the dashboard.", size=10, color=ORANGE))
    kw, kh = cw // 5, 6500
    kz = "".join(vis(n, x0 + (i % 5) * kw, 7000 + (i // 5) * kh, kw, kh, border=LIGHT) for i, (n, _) in enumerate(kpi_sheets))
    top = 7000 + 2 * kh + gap
    row = (H - top) // 3
    half, third = cw // 2, cw // 3
    zones = (txt(0, 0, W, 6500, run("CRM Online School · Leads, Conversion and Revenue", size=24, color=BLUE))
             + txt(0, 7000, side, H - 7000, insights, border=LIGHT, fill=PANEL) + kz
             + vis("Leads by Month", x0, top, half, row) + vis("C1 by Month", x0 + half, top, cw - half, row)
             + vis("C1 by Source", x0, top + row, half, row) + vis("CAC by Source", x0 + half, top + row, cw - half, row)
             + vis("Speed to Lead", x0, top + 2 * row, third, H - top - 2 * row)
             + vis("Revenue by Product", x0 + third, top + 2 * row, third, H - top - 2 * row)
             + vis("Prepaid by Status", x0 + 2 * third, top + 2 * row, cw - 2 * third, H - top - 2 * row))
    note = (run("Why product A/B tests do not fit into 14 days", bold=True, size=13, color=BLUE) + run(br)
            + run("Leads with a known product are only 3–4 a day, so even the best 14-day design detects +18–29 pp, "
                  "far above the expected +4–7.5 pp. Only the speed-to-lead test fits into 14 days, and only if the 15-minute rule "
                  "runs daily 8:00–23:00 (MDE +9.4 pp vs expected +12.2 pp); in office hours only it would not (MDE +13.6 pp "
                  "vs +11.0 pp). Product hypotheses are checked as pilots.", size=11, color=INK))
    ab_zones = (txt(0, 0, W, 6500, run("Product Hypotheses and A/B Test Options · 14 days, α = 0.05, power 80%", size=22, color=BLUE))
                + txt(0, 7000, W, 9000, note, border=LIGHT, fill=PANEL)
                + vis("Product Hypotheses", 0, 17000, W, 26000) + vis("A/B Options: MDE in 14 days", 0, 44000, W, H - 44000))

    def dashboard(name, ds, ds_caption, inner):
        top_align = "".join(f"<format attr='vertical-align' id='dash-text_{i}' value='top' />" for i in text_ids)
        return (f"<dashboard name='{q(name)}'><style><style-rule element='dash-text'>{top_align}</style-rule></style>"
                f"<size maxheight='1000' maxwidth='1200' minheight='1000' minwidth='1200' sizing-mode='fixed' />"
                f"<datasources><datasource caption='{ds_caption}' name='{ds}' /></datasources>"
                f"<zones><zone h='{H}' id='{next(zid)}' type-v2='layout-basic' w='{W}' x='0' y='0'>{inner}</zone></zones>"
                f"{sid(name)}</dashboard>")

    dashboards = ("<dashboards>" + dashboard("CRM Dashboard", DS_CRM, "CRM model", zones)
                  + dashboard("A/B Test Options", DS_AB, "A/B test options", ab_zones) + "</dashboards>")
    actions = ("<actions><action caption='Filter by source' name='[Action1]'>"
               "<activation auto-clear='true' type='on-select' />"
               "<source dashboard='CRM Dashboard' type='sheet' worksheet='C1 by Source' />"
               "<command command='tsc:tsl-filter'><param name='exclude' value='C1 by Source' />"
               "<param name='field-captions' value='Source' /><param name='target' value='CRM Dashboard' /></command>"
               "</action></actions>")
    all_sheets = sheets + ab_sheets
    # worksheets are hidden: they live inside the two dashboards, so viewers see only the dashboards as tabs
    windows = "".join(f"<window class='worksheet' hidden='true' name='{q(n)}'><cards><edge name='left'><strip size='160'>"
                      f"<card type='pages' /><card type='filters' /><card type='marks' /></strip></edge></cards>{sid(n)}</window>"
                      for n, _ in all_sheets)
    for dname, group, maxim in (("CRM Dashboard", sheets, " maximized='true'"), ("A/B Test Options", ab_sheets, "")):
        windows += (f"<window class='dashboard'{maxim} name='{q(dname)}'><viewpoints>"
                    + "".join(f"<viewpoint name='{q(n)}' />" for n, _ in group)
                    + f"</viewpoints><active id='-1' />{sid(dname)}</window>")

    twb = ("<?xml version='1.0' encoding='utf-8' ?>\n"
           "<workbook original-version='18.1' source-build='2026.2.0 (20262.26.0603.1643)' source-platform='win' version='18.1' "
           "xmlns:user='http://www.tableausoftware.com/xml/user'>"
           "<document-format-change-manifest><SheetIdentifierTracking /><SortTagCleanup /><WindowsPersistSimpleIdentifiers />"
           "</document-format-change-manifest><preferences />"
           f"<datasources>{crm_ds}{ab_ds}</datasources>{actions}<worksheets>{''.join(x for _, x in all_sheets)}</worksheets>"
           f"{dashboards}<windows>{windows}</windows></workbook>")
    (OUT_DIR / "crm_dashboard.twb").write_text(twb, encoding="utf-8")
    with zipfile.ZipFile(TWBX, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("crm_dashboard.twb", twb)
        for table in TABLES + ["ab_variants.csv"]:
            zf.write(OUT_DIR / table, f"Data/tableau/{table}")
        for name in extracts:
            zf.write(OUT_DIR / Path(name).name, name)
    print(f"Saved {TWBX} ({TWBX.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
