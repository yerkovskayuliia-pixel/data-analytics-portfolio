"""
Собирает интерактивный дашборд reports/dashboard.html (один файл, Plotly.js).

Данные — компактная выгрузка очищенных сделок и расходов, встроенная в HTML как JSON;
фильтры (источник, продукт, период) пересчитывают все метрики в браузере.
Запуск: python src/build_dashboard.py
"""
import json
from pathlib import Path

import pandas as pd

import cleaning as etl

OUT = Path(__file__).resolve().parents[1] / "reports" / "dashboard.html"


def build():
    data = etl.load_clean()
    d, s = data["Deals"], data["Spend"]
    cols = {
        "m": d["month"].dt.strftime("%Y-%m"),
        "src": d["source"].astype(str),
        "prod": d["product"].astype(str),
        "own": d["deal_owner_name"].astype(str),
        "q": d["quality"].astype(str).str[0].where(d["quality"].ne("Not rated"), "-"),
        "st": d["is_student"].astype(int),
        "rev": d["revenue"].round(0).astype(int),
        "sla": pd.cut(d["sla_sec"] / 3600, [0, 0.25, 1, 4, 24, 1e6],
                      labels=["<15m", "15-60m", "1-4h", "4-24h", ">24h"]).astype(str),
        "city": d["city"].astype(str),
        "camp": d["campaign"].astype(str),
    }
    deals = pd.DataFrame(cols)
    # словарное кодирование строк, чтобы HTML был компактным
    dicts, enc = {}, {}
    for c in ["m", "src", "prod", "own", "q", "sla", "city", "camp"]:
        cat = pd.Categorical(deals[c])
        dicts[c] = list(cat.categories)
        enc[c] = cat.codes.tolist()
    for c in ["st", "rev"]:
        enc[c] = deals[c].tolist()

    sp = (s.assign(m=s["date"].dt.strftime("%Y-%m"), src=s["source"].astype(str))
            .groupby(["m", "src", "campaign"], as_index=False)["spend"].sum().round(2))
    payload = {"dict": dicts, "deals": enc,
               "spend": {"m": sp["m"].tolist(), "src": sp["src"].tolist(),
                         "camp": sp["campaign"].tolist(), "v": sp["spend"].tolist()}}
    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CRM Sales Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root{--bg:#f4f3ef;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e3e1da;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--muted:#a8a69c}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2e2e2b;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--muted:#6f6e68}}
:root[data-theme="dark"]{--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2e2e2b;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--muted:#6f6e68}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:22px;margin:0 0 2px}.sub{color:var(--ink2);margin:0 0 16px}
.filters{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin-bottom:16px}
.filters label{display:flex;flex-direction:column;font-size:12px;color:var(--ink2);gap:4px}
select{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);min-width:150px}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);cursor:pointer}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.kpi .l{font-size:12px;color:var(--ink2)}.kpi .v{font-size:24px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;min-width:0}
.card h2{font-size:14px;margin:0 0 4px}.card p{margin:0 0 6px;color:var(--ink2);font-size:12px}
.chart{height:300px}.wide{grid-column:1/-1}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}.tbl{overflow-x:auto}
footer{margin-top:20px;color:var(--ink2);font-size:12px}
@media (max-width:760px){.grid{grid-template-columns:1fr}.kpi .v{font-size:20px}}
</style></head><body><div class="wrap">
<h1>Online school X: CRM sales &amp; marketing dashboard</h1>
<p class="sub">Leads, students, revenue and acquisition cost, July 2023 to June 2024. Use the filters to recalculate every number on the page.</p>
<div class="filters">
 <label>Source<select id="fSrc"></select></label>
 <label>Product<select id="fProd"></select></label>
 <label>From<select id="fFrom"></select></label>
 <label>To<select id="fTo"></select></label>
 <button id="reset">Reset</button>
 <button id="theme" title="Toggle light/dark">◐</button>
</div>
<div class="kpis" id="kpis"></div>
<div class="grid">
 <div class="card"><h2>Leads by month</h2><p>New leads created in the CRM.</p><div class="chart" id="cMonth"></div></div>
 <div class="card"><h2>Conversion by lead month (C1)</h2><p>Share of the month's leads that became students. The last months are still maturing.</p><div class="chart" id="cC1"></div></div>
 <div class="card"><h2>Funnel by lead quality</h2><p>Manager rating predicts payment: A converts ~34%, D/E almost never.</p><div class="chart" id="cQual"></div></div>
 <div class="card"><h2>Speed to first response (SLA)</h2><p>Leads answered within 15 minutes buy more often.</p><div class="chart" id="cSla"></div></div>
 <div class="card wide"><h2>Marketing sources</h2><p>Spend is attributed by source; CAC = spend ÷ students. Organic, CRM and Partnership have no ad spend.</p><div class="tbl" id="tSrc"></div></div>
 <div class="card"><h2>Products</h2><p>Students and revenue by course.</p><div class="chart" id="cProd"></div></div>
 <div class="card"><h2>Top cities</h2><p>City is filled only for ~11% of deals (at consultation stage).</p><div class="chart" id="cCity"></div></div>
 <div class="card wide"><h2>Sales team</h2><p>Deal owners with at least 30 deals in the current selection.</p><div class="tbl" id="tOwn"></div></div>
</div>
<footer>Source: cleaned CRM export (Deals, Spend). Student = stage "Payment Done" with months of study filled, demo access excluded. Revenue = payments actually received by the export date.</footer>
</div>
<script>
const P=/*__DATA__*/null;
const D=P.dict, X=P.deals, N=X.st.length;
const $=id=>document.getElementById(id);
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmtE=v=>v>=1e6?`€${(v/1e6).toFixed(2)}M`:v>=1e3?`€${(v/1e3).toFixed(1)}k`:`€${Math.round(v)}`;
const pct=(v,d=1)=>isFinite(v)?(v*100).toFixed(d)+'%':'—';
function opts(sel,vals,all){sel.innerHTML=(all?`<option value="">${all}</option>`:'')+vals.map(v=>`<option>${v}</option>`).join('')}
const months=D.m;
opts($('fSrc'),[...D.src].sort(),'All sources');
opts($('fProd'),D.prod.filter(p=>!['Unknown','Data Analytics','Find yourself in IT'].includes(p)),'All products');
opts($('fFrom'),months);opts($('fTo'),months);$('fTo').value=months[months.length-1];
function rows(){
 const s=$('fSrc').value,p=$('fProd').value,a=months.indexOf($('fFrom').value),b=months.indexOf($('fTo').value);
 const si=s?D.src.indexOf(s):-1,pi=p?D.prod.indexOf(p):-1,idx=[];
 for(let i=0;i<N;i++){const m=X.m[i];if(m<a||m>b)continue;if(si>=0&&X.src[i]!==si)continue;if(pi>=0&&X.prod[i]!==pi)continue;idx.push(i)}
 return {idx,s,p,a,b};
}
function spendBy(f){const S=P.spend,o={};for(let i=0;i<S.v.length;i++){const mi=months.indexOf(S.m[i]);if(mi<f.a||mi>f.b)continue;if(f.s&&S.src[i]!==f.s)continue;o[S.src[i]]=(o[S.src[i]]||0)+S.v[i]}return o}
function group(idx,key){const g={};for(const i of idx){const k=X[key][i];const r=g[k]||(g[k]={n:0,st:0,rev:0});r.n++;r.st+=X.st[i];r.rev+=X.rev[i]}return g}
const layout=(extra)=>Object.assign({margin:{l:48,r:12,t:8,b:40},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
 font:{family:'system-ui,Segoe UI,sans-serif',size:12,color:css('--ink2')},xaxis:{gridcolor:css('--line'),zeroline:false},
 yaxis:{gridcolor:css('--line'),zeroline:false},showlegend:false,hoverlabel:{bgcolor:css('--card'),bordercolor:css('--line'),font:{color:css('--ink')}}},extra);
const cfg={displayModeBar:false,responsive:true};
function render(){
 const f=rows(),idx=f.idx,sp=spendBy(f),spend=Object.values(sp).reduce((a,b)=>a+b,0);
 let st=0,rev=0;for(const i of idx){st+=X.st[i];rev+=X.rev[i]}
 const spendOk=!f.p;
 const K=[['Leads',idx.length.toLocaleString('en')],['Students',st.toLocaleString('en')],['Conversion C1',pct(st/idx.length,2)],
  ['Revenue',fmtE(rev)],['Ad spend',spendOk?fmtE(spend):'—'],['CAC',spendOk&&st?fmtE(spend/st):'—'],['Revenue / student',st?fmtE(rev/st):'—']];
 $('kpis').innerHTML=K.map(([l,v])=>`<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div></div>`).join('');
 // month
 const gm=group(idx,'m'),ms=months.filter((_,i)=>i>=f.a&&i<=f.b),mi=ms.map(m=>months.indexOf(m));
 Plotly.react('cMonth',[{type:'bar',x:ms,y:mi.map(i=>(gm[i]||{}).n||0),marker:{color:css('--s1')},name:'Leads',hovertemplate:'%{x}<br>Leads: %{y:,}<extra></extra>'}],layout({}),cfg);
 Plotly.react('cC1',[{type:'scatter',mode:'lines+markers',x:ms,y:mi.map(i=>gm[i]?gm[i].st/gm[i].n:null),line:{color:css('--s1'),width:2},marker:{size:8},
  customdata:mi.map(i=>gm[i]?gm[i].st:0),hovertemplate:'%{x}<br>C1: %{y:.1%}<br>Students: %{customdata}<extra></extra>'}],layout({yaxis:{tickformat:'.0%',gridcolor:css('--line'),rangemode:'tozero'}}),cfg);
 // quality
 const gq=group(idx,'q'),ql=['A','B','C','D','E','-'].filter(q=>D.q.includes(q)),qn={A:'A High',B:'B Medium',C:'C Low',D:'D Non target',E:'E Non qualified','-':'Not rated'};
 Plotly.react('cQual',[{type:'bar',x:ql.map(q=>qn[q]),y:ql.map(q=>{const r=gq[D.q.indexOf(q)];return r?r.st/r.n:0}),marker:{color:css('--s1')},
  customdata:ql.map(q=>(gq[D.q.indexOf(q)]||{n:0}).n),hovertemplate:'%{x}<br>C1: %{y:.1%}<br>Leads: %{customdata:,}<extra></extra>'}],layout({yaxis:{tickformat:'.0%',gridcolor:css('--line')}}),cfg);
 // sla
 const gs=group(idx,'sla'),sl=['<15m','15-60m','1-4h','4-24h','>24h'];
 Plotly.react('cSla',[{type:'bar',x:sl,y:sl.map(k=>{const r=gs[D.sla.indexOf(k)];return r?r.st/r.n:0}),marker:{color:css('--s1')},
  customdata:sl.map(k=>(gs[D.sla.indexOf(k)]||{n:0}).n),hovertemplate:'First response %{x}<br>C1: %{y:.2%}<br>Leads: %{customdata:,}<extra></extra>'}],layout({yaxis:{tickformat:'.1%',gridcolor:css('--line')}}),cfg);
 // sources table
 const gsrc=group(idx,'src');
 const srows=Object.entries(gsrc).map(([k,r])=>{const name=D.src[k],sv=spendOk?(sp[name]||0):NaN;return {name,...r,sv}}).sort((a,b)=>b.n-a.n);
 $('tSrc').innerHTML=`<table><tr><th>Source</th><th>Leads</th><th>Students</th><th>C1</th><th>Revenue</th><th>Spend</th><th>CPL</th><th>CAC</th></tr>`+
  srows.map(r=>`<tr><td>${r.name}</td><td>${r.n.toLocaleString('en')}</td><td>${r.st}</td><td>${pct(r.st/r.n)}</td><td>${fmtE(r.rev)}</td><td>${isNaN(r.sv)?'—':fmtE(r.sv)}</td><td>${r.sv>0?'€'+(r.sv/r.n).toFixed(2):'—'}</td><td>${r.sv>0&&r.st?fmtE(r.sv/r.st):'—'}</td></tr>`).join('')+'</table>';
 // products
 const gp=group(idx,'prod'),pl=['Digital Marketing','UX/UI Design','Web Developer'];
 Plotly.react('cProd',[{type:'bar',orientation:'h',y:pl,x:pl.map(p=>(gp[D.prod.indexOf(p)]||{st:0}).st),marker:{color:css('--s1')},
  customdata:pl.map(p=>(gp[D.prod.indexOf(p)]||{rev:0}).rev),hovertemplate:'%{y}<br>Students: %{x}<br>Revenue: €%{customdata:,.0f}<extra></extra>'}],layout({margin:{l:120,r:12,t:8,b:40}}),cfg);
 // cities
 const gc=group(idx,'city');const cl=Object.entries(gc).filter(([k])=>D.city[k]!=='Unknown').sort((a,b)=>b[1].n-a[1].n).slice(0,10).reverse();
 Plotly.react('cCity',[{type:'bar',orientation:'h',y:cl.map(([k])=>D.city[k]),x:cl.map(([,r])=>r.n),marker:{color:css('--muted')},name:'Deals',hovertemplate:'%{y}<br>Deals: %{x}<extra></extra>'},
  {type:'bar',orientation:'h',y:cl.map(([k])=>D.city[k]),x:cl.map(([,r])=>r.st),marker:{color:css('--s1')},name:'Students',hovertemplate:'%{y}<br>Students: %{x}<extra></extra>'}],
  layout({barmode:'overlay',margin:{l:100,r:12,t:8,b:40},showlegend:true,legend:{orientation:'h',y:1.12,x:0}}),cfg);
 // owners
 const go=group(idx,'own');const orows=Object.entries(go).filter(([,r])=>r.n>=30).sort((a,b)=>b[1].rev-a[1].rev);
 $('tOwn').innerHTML=`<table><tr><th>Deal owner</th><th>Deals</th><th>Students</th><th>C1</th><th>Revenue</th><th>Revenue / deal</th></tr>`+
  orows.map(([k,r])=>`<tr><td>${D.own[k]}</td><td>${r.n.toLocaleString('en')}</td><td>${r.st}</td><td>${pct(r.st/r.n)}</td><td>${fmtE(r.rev)}</td><td>€${Math.round(r.rev/r.n)}</td></tr>`).join('')+'</table>';
}
['fSrc','fProd','fFrom','fTo'].forEach(id=>$(id).addEventListener('change',render));
$('reset').onclick=()=>{$('fSrc').value='';$('fProd').value='';$('fFrom').value=months[0];$('fTo').value=months[months.length-1];render()};
$('theme').onclick=()=>{const r=document.documentElement,dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=dark?'light':'dark';render()};
render();
</script></body></html>"""

if __name__ == "__main__":
    build()
