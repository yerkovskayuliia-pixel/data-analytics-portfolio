"""
Shared metric definitions: one source of truth for notebooks 02–04 and the build scripts.

- add_contact_metrics: first conversation, reach in 24 h, quality A+B, SLA buckets
- funnel:              lead → call → conversation → consultation → invoice → student
- speed_test_design:   coverage, expected effect and MDE of the speed-to-lead A/B test

Table labels are Russian by default (the original notebooks); call set_lang("en") for the English notebooks.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower

AB = ["A - High", "B - Medium"]
SLA_BINS = [0, 0.25, 1, 4, 24, 1e6]
SLA_LABELS = ["< 15 min", "15–60 min", "1–4 h", "4–24 h", "> 24 h"]
SLOW = ["15–60 min", "1–4 h", "4–24 h"]          # buckets the intervention can speed up
ALPHA, POWER, TEST_DAYS = 0.05, 0.80, 14
RECENT_FROM = "2024-03-01"                        # traffic level used to size the test

# rule coverage options: (first hour, last hour inclusive, weekdays covered: 5 = Mon–Fri, 7 = every day)
_WINDOWS = {"office": (9, 17, 5), "weekday_8_21": (8, 20, 5), "daily_8_21": (8, 20, 7), "daily_8_23": (8, 22, 7)}
_CHOSEN = "daily_8_23"
_TEXT = {
    "ru": {
        "windows": {"office": "пн–пт 9:00–18:00 (текущий график)", "weekday_8_21": "пн–пт 8:00–21:00",
                    "daily_8_21": "ежедневно 8:00–21:00", "daily_8_23": "ежедневно 8:00–23:00"},
        "steps": ["Лид", "Был звонок", "Разговор ≥ 30 с", "Консультация (указан продукт)", "Счёт или оплата", "Студент"],
        "step": "шаг", "leads": "лидов", "share": "% от лидов", "step_conv": "конверсия шага", "lost": "потеряно на шаге",
        "w_share": "доля лидов в окне", "w_per_day": "лидов в окне в день", "w_n": "лидов на группу за 14 дней",
        "w_base": "baseline дозвона в окне", "w_effect": "ожидаемый эффект в окне, п.п.", "w_mde": "MDE в окне, п.п.",
        "all_effect": "эффект на всех лидах, п.п.", "all_mde": "MDE на всех лидах, п.п.",
        "w_ok": "тест в окне имеет смысл",
    },
    "en": {
        "windows": {"office": "Mon–Fri 9:00–18:00 (current schedule)", "weekday_8_21": "Mon–Fri 8:00–21:00",
                    "daily_8_21": "daily 8:00–21:00", "daily_8_23": "daily 8:00–23:00"},
        "steps": ["Lead", "Called", "Conversation ≥ 30 s", "Consultation (product set)", "Invoice or payment", "Student"],
        "step": "step", "leads": "leads", "share": "% of leads", "step_conv": "step conversion", "lost": "lost at step",
        "w_share": "share of leads in window", "w_per_day": "window leads per day", "w_n": "leads per group in 14 days",
        "w_base": "baseline reach in window", "w_effect": "expected effect in window, pp", "w_mde": "MDE in window, pp",
        "all_effect": "effect on all leads, pp", "all_mde": "MDE on all leads, pp",
        "w_ok": "test in window makes sense",
    },
}
L = _TEXT["ru"]
WINDOWS, CHOSEN_WINDOW = {}, ""


def set_lang(lang: str = "ru"):
    """Switches table labels and window names: 'ru' (default) or 'en'."""
    global L, WINDOWS, CHOSEN_WINDOW
    L = _TEXT[lang]
    WINDOWS = {L["windows"][k]: v for k, v in _WINDOWS.items()}
    CHOSEN_WINDOW = L["windows"][_CHOSEN]


set_lang("ru")


def add_contact_metrics(deals: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    """Adds first_talk, talked (ever), reached_24h, is_AB and sla_bucket to a copy of deals."""
    d = deals.copy()
    first_talk = calls[calls["is_conversation"]].groupby("contact_id")["call_start_time"].min()
    d["first_talk"] = d["contact_id"].map(first_talk)
    lag = (d["first_talk"] - d["created_time"]).dt.total_seconds()
    d["talked"] = d["first_talk"].notna()
    d["reached_24h"] = lag.between(-3600, 86400)   # 1 h tolerance: calls are logged in local time
    d["is_AB"] = d["quality"].isin(AB)
    d["sla_bucket"] = pd.cut(d["sla_sec"] / 3600, SLA_BINS, labels=SLA_LABELS)
    return d


def mde(p_base: float, n_group: int, alpha: float = ALPHA, power: float = POWER,
        alternative: str = "two-sided") -> float:
    """Minimum detectable lift (share) for a two-proportion test with n_group leads per group (Cohen's h)."""
    h = NormalIndPower().solve_power(nobs1=n_group, alpha=alpha, power=power, ratio=1, alternative=alternative)
    return np.sin(min(np.arcsin(np.sqrt(p_base)) + h / 2, np.pi / 2)) ** 2 - p_base


def in_window(ts: pd.Series, first_hour: int, last_hour: int, weekdays: int) -> pd.Series:
    return ts.dt.hour.between(first_hour, last_hour) & (ts.dt.dayofweek < weekdays)


def funnel(d: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    """Cumulative funnel: each step requires all previous ones. d must come from add_contact_metrics."""
    conds = [pd.Series(True, index=d.index), d["contact_id"].isin(calls["contact_id"]), d["talked"],
             d["product"].ne("Unknown"), d["stage"].isin(["Payment Done", "Waiting For Payment"]), d["is_student"]]
    mask, rows = pd.Series(True, index=d.index), []
    for name, cond in zip(L["steps"], conds):
        mask = mask & cond
        rows.append((name, int(mask.sum())))
    n = L["leads"]
    f = pd.DataFrame(rows, columns=[L["step"], n]).set_index(L["step"])
    f[L["share"]] = f[n] / f[n].iloc[0]
    f[L["step_conv"]] = f[n] / f[n].shift().fillna(f[n].iloc[0])
    f[L["lost"]] = (f[n].shift() - f[n]).fillna(0).astype(int)
    return f


def speed_test_design(d: pd.DataFrame, compliance: float = 1.0) -> pd.DataFrame:
    """
    For each coverage window of the «call within 15 min» rule:
    leads per day, sample per group in 14 days, baseline reach, expected effect and MDE.

    Expected effect (in the window): slow leads (15 min – 24 h) get the reach rate of fast leads (< 15 min);
    leads with no SLA or > 24 h are assumed unaffected (conservative). compliance = share of test leads
    really called within 15 minutes.
    """
    recent = d[d["created_time"] >= RECENT_FROM]
    n_days = recent["created_time"].dt.normalize().nunique()
    p_fast = d.loc[d["sla_bucket"].eq("< 15 min"), "reached_24h"].mean()
    p_all = recent["reached_24h"].mean()
    rows = {}
    for name, (h0, h1, wd) in WINDOWS.items():
        w = in_window(recent["created_time"], h0, h1, wd)
        sub = recent[w]
        slow = sub["sla_bucket"].isin(SLOW)
        effect_w = (slow * (p_fast - sub["reached_24h"])).sum() / len(sub) * compliance
        n_w = int(w.sum() / n_days * TEST_DAYS / 2)
        rows[name] = {
            L["w_share"]: w.mean(),
            L["w_per_day"]: w.sum() / n_days,
            L["w_n"]: n_w,
            L["w_base"]: sub["reached_24h"].mean(),
            L["w_effect"]: effect_w * 100,
            L["w_mde"]: mde(sub["reached_24h"].mean(), n_w) * 100,
            L["all_effect"]: effect_w * w.mean() * 100,
            L["all_mde"]: mde(p_all, int(len(recent) / n_days * TEST_DAYS / 2)) * 100,
        }
    t = pd.DataFrame(rows).T
    t[L["w_ok"]] = t[L["w_effect"]] >= t[L["w_mde"]]
    return t


def break_even_compliance(d: pd.DataFrame, window: str = None) -> float:
    """Smallest compliance at which the expected effect in the window still reaches the MDE."""
    full = speed_test_design(d).loc[window or CHOSEN_WINDOW]
    return full[L["w_mde"]] / full[L["w_effect"]]
