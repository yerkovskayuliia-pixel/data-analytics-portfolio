"""
ETL для CRM онлайн-школы: загрузка сырых выгрузок, очистка и сохранение.

Каждая функция clean_* возвращает очищенный DataFrame и журнал (log) —
список строк «что сделано и сколько строк затронуто». Журнал выводится
в ноутбуке 01_data_cleaning и служит обоснованием для защиты.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Длинные ID (19 цифр) читаем строкой: во float они теряют точность,
# и связь Deals -> Contacts -> Calls ломается.
ID_COLS = {"Id": str, "CONTACTID": str, "Contact Name": str}

NON_GERMAN_CITIES = {
    "Wien": "Austria", "Graz": "Austria", "Salzburg": "Austria", "Linz": "Austria",
    "Innsbruck": "Austria", "Warszawa": "Poland", "Bialystok": "Poland",
    "Dabrowa Gornicza": "Poland", "Gdansk": "Poland", "Krakow": "Poland",
    "Wroclaw": "Poland", "Poznan": "Poland", "Podskalie": "Slovakia",
    "Tatabanya": "Hungary", "Moscow": "Russia", "Tashkent": "Uzbekistan",
    "Kyiv": "Ukraine", "Zürich": "Switzerland", "Basel": "Switzerland",
}


def load_raw() -> dict[str, pd.DataFrame]:
    """Читает 4 выгрузки CRM из data/raw (xlsx сохраняет кириллицу и точные ID)."""
    return {
        name: pd.read_excel(RAW / f"{name}.xlsx", dtype=ID_COLS)
        for name in ["Contacts", "Calls", "Spend", "Deals"]
    }


def to_snake(df: pd.DataFrame) -> pd.DataFrame:
    """'Call Duration (in seconds)' -> 'call_duration_in_seconds'."""
    df = df.copy()
    df.columns = [re.sub(r"[^0-9a-z]+", "_", c.lower()).strip("_") for c in df.columns]
    return df


# ---------------------------------------------------------------- Contacts
def clean_contacts(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log = []
    df = to_snake(raw)
    n = len(df)
    df = df.drop_duplicates("id")
    log.append(f"Дубликаты по id удалены: {n - len(df)}")

    for col in ["created_time", "modified_time"]:
        df[col] = pd.to_datetime(df[col], format="%d.%m.%Y %H:%M")
    log.append("created_time, modified_time -> datetime64 (формат дд.мм.гггг чч:мм)")

    df["contact_owner_name"] = df["contact_owner_name"].fillna("Unassigned").astype("category")
    log.append("Пропусков нет; contact_owner_name -> category")
    return df.reset_index(drop=True), log


# ---------------------------------------------------------------- Calls
def clean_calls(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log = []
    df = to_snake(raw).rename(columns={"call_duration_in_seconds": "duration_sec",
                                       "contactid": "contact_id"})
    n = len(df)
    df = df.drop_duplicates("id")
    log.append(f"Дубликаты по id удалены: {n - len(df)}")

    empty = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty)
    log.append(f"Удалены полностью пустые столбцы: {empty}")

    df["call_start_time"] = pd.to_datetime(df["call_start_time"], format="%d.%m.%Y %H:%M")

    n = len(df)
    df = df[df["duration_sec"].notna() & (df["duration_sec"] >= 0)]
    log.append(f"Удалены звонки без длительности/с отрицательной длительностью: {n - len(df)}")
    df["duration_sec"] = df["duration_sec"].astype(int)

    # Входящие и пропущенные звонки не имеют статуса исходящего вызова — это не пропуск, а «не применимо».
    df["outgoing_call_status"] = df["outgoing_call_status"].fillna("Not applicable")
    df["scheduled_in_crm"] = df["scheduled_in_crm"].fillna(0).astype(bool)
    log.append("outgoing_call_status: NaN -> 'Not applicable' (входящие/пропущенные); "
               "scheduled_in_crm -> bool")

    no_contact = df["contact_id"].isna().sum()
    log.append(f"Звонков без contact_id: {no_contact} — оставлены для анализа нагрузки "
               "менеджеров, но не участвуют в связке со сделками")

    df["is_conversation"] = df["duration_sec"] >= 30
    log.append("Добавлен флаг is_conversation (>= 30 сек): медиана звонка 8 сек, "
               "короткие звонки — недозвоны/автоответчики")
    for col in ["call_owner_name", "call_type", "call_status", "outgoing_call_status"]:
        df[col] = df[col].astype("category")
    return df.reset_index(drop=True), log


# ---------------------------------------------------------------- Spend
def clean_spend(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log = []
    df = to_snake(raw)
    n = len(df)
    df = df.drop_duplicates()
    log.append(f"Полные дубликаты удалены: {n - len(df)} (все — нулевые строки без расходов)")

    df["date"] = pd.to_datetime(df["date"])
    df["campaign"] = df["campaign"].fillna("(no campaign)")
    df[["adgroup", "ad"]] = df[["adgroup", "ad"]].fillna("(not set)")
    log.append("campaign/adgroup/ad: NaN -> '(no campaign)'/'(not set)' — расход "
               "реальный, просто не размечен, удалять его нельзя")

    neg = (df[["impressions", "spend", "clicks"]] < 0).any(axis=1).sum()
    log.append(f"Отрицательных значений impressions/spend/clicks: {neg}")
    df["source"] = df["source"].astype("category")
    return df.reset_index(drop=True), log


# ---------------------------------------------------------------- Deals
def parse_money(x) -> float:
    """'€ 3.500,00' / '$11.000,00' / 1000 -> float (европейский формат: точка — тысячи)."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = re.sub(r"[€$\s]", "", str(x)).replace(".", "").replace(",", ".")
    return float(s) if s else np.nan


def parse_sla(x) -> float:
    """SLA приходит как time (чч:мм:сс) или timedelta (> 24 ч) -> секунды."""
    if pd.isna(x):
        return np.nan
    if hasattr(x, "total_seconds"):
        return x.total_seconds()
    if hasattr(x, "hour"):
        return x.hour * 3600 + x.minute * 60 + x.second
    m = re.match(r"(?:(\d+) )?(\d+):(\d+):(\d+)", str(x))
    if not m:
        return np.nan
    d, h, mi, s = (int(g or 0) for g in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s


CYR_TO_LAT = str.maketrans({"а": "a", "б": "b", "в": "b", "с": "c", "А": "a",
                            "Б": "b", "В": "b", "С": "c"})


def normalize_german(x) -> str:
    """Свободный текст менеджера -> уровень CEFR (A1..C2).

    Менеджеры писали кириллицей (б1, в1 = B1), в разном регистре и с комментариями
    («б1 (проходит Б2)»). Берём первый упомянутый уровень — текущий, а не цель.
    """
    if pd.isna(x):
        return "Unknown"
    s = str(x).translate(CYR_TO_LAT).lower()
    m = re.search(r"([abc])\s*([12])", s)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}"
    if re.search(r"нулев|никак|^0$|a0", s):
        return "A0"
    return "Unknown"


def normalize_city(x) -> str:
    if pd.isna(x) or str(x).strip() in {"-", ".", ""}:
        return "Unknown"
    s = str(x).strip()
    # «Karl-Liebknecht str. 24, Hildburghausen, Thüringen» -> Hildburghausen
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        s = next((p for p in parts if not re.search(r"\d|str\.|straße|poland", p, re.I)), parts[0])
    s = re.sub(r"\s+", " ", s).title()
    return {"Munich": "München", "Muenchen": "München", "Cologne": "Köln",
            "Frankfurt Am Main": "Frankfurt", "Nuremberg": "Nürnberg",
            "Duesseldorf": "Düsseldorf", "Vienna": "Wien", "Warsaw": "Warszawa",
            "Helmstidde": "Helmstedt"}.get(s, s)


def realized_revenue(row) -> float:
    """Выручка, фактически полученная к моменту выгрузки.

    Первый платёж = initial_amount_paid, остаток курса делится на оставшиеся
    месяцы и оплачивается помесячно. Студент, отучившийся N месяцев, заплатил
    initial + (N-1) * ежемесячный платёж. При единовременной оплате (initial == offer)
    выручка = offer. Не может превышать offer.
    """
    offer, init, months, dur = (row["offer_total_amount"], row["initial_amount_paid"],
                                row["months_of_study"], row["course_duration"])
    if pd.isna(offer):
        return np.nan
    init = 0.0 if pd.isna(init) else init
    if init >= offer or dur <= 1:
        return offer
    monthly = (offer - init) / (dur - 1)
    return min(offer, init + max(months - 1, 0) * monthly)


def clean_deals(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log = []
    df = to_snake(raw).rename(columns={"contact_name": "contact_id"})

    n = len(df)
    df = df[df["id"].notna()].drop_duplicates("id")
    log.append(f"Удалены пустые строки/дубли id: {n - len(df)}")

    n = len(df)
    df = df[df["lost_reason"].ne("Duplicate")]
    log.append(f"Удалены технические дубли лидов (Lost Reason = 'Duplicate', см. FAQ): {n - len(df)}")

    df = df.drop(columns=["content", "term"])
    log.append("Удалены неактуальные столбцы content, term (детализация объявлений, "
               "не используется в анализе; кампания и источник остаются)")

    # --- даты
    df["created_time"] = pd.to_datetime(df["created_time"], format="%d.%m.%Y %H:%M")
    df["closing_date"] = pd.to_datetime(df["closing_date"], format="%d.%m.%Y")
    bad = df["closing_date"] < df["created_time"].dt.normalize()
    df.loc[bad, "closing_date"] = pd.NaT
    log.append(f"closing_date раньше created_time — ошибка ввода, -> NaT: {bad.sum()}")
    df["deal_cycle_days"] = (df["closing_date"] - df["created_time"].dt.normalize()).dt.days

    df["sla_sec"] = df["sla"].map(parse_sla)
    df = df.drop(columns="sla")
    log.append("SLA (чч:мм:сс и 'дни чч:мм:сс') -> sla_sec (секунды)")

    # --- деньги
    for col in ["initial_amount_paid", "offer_total_amount"]:
        df[col] = df[col].map(parse_money)
    log.append("initial_amount_paid / offer_total_amount: '€ 3.500,00', '$11.000,00' -> float")

    # Offer 1 200 при первом платеже 11 000 по 11-месячному курсу — пропущен ноль в цене курса.
    typo = (df["initial_amount_paid"] > df["offer_total_amount"]) & (df["offer_total_amount"] < 2000) \
        & (df["course_duration"] == 11)
    df.loc[typo, "offer_total_amount"] = df.loc[typo, "initial_amount_paid"]
    over = df["initial_amount_paid"] > df["offer_total_amount"]
    df["amount_anomaly_flag"] = over | typo
    df.loc[over, "initial_amount_paid"] = df.loc[over, "offer_total_amount"]
    log.append(f"initial > offer (очевидная ошибка, FAQ): offer исправлен как опечатка в {typo.sum()} "
               f"строках, initial ограничен offer в {over.sum()} строках; флаг amount_anomaly_flag")

    df["is_demo"] = df["offer_total_amount"].isin([0, 1, 9])
    log.append(f"Демо-доступы за символическую плату (offer 0/1/9 €): {df['is_demo'].sum()}")

    # --- статус студента
    df["is_paid"] = df["stage"].eq("Payment Done")
    df["is_student"] = df["is_paid"] & df["months_of_study"].notna() & ~df["is_demo"]
    log.append(f"Payment Done: {df['is_paid'].sum()}; из них студентов (months_of_study заполнен, "
               f"не демо): {df['is_student'].sum()}. Без months_of_study деньги не учитываем (FAQ)")

    df["revenue"] = np.where(df["is_student"], df.apply(realized_revenue, axis=1), 0.0)
    df["payments_count"] = np.where(
        df["is_student"],
        np.where(df["initial_amount_paid"] >= df["offer_total_amount"], 1,
                 df["months_of_study"].clip(lower=1)),
        0)
    log.append("revenue: выручка, полученная к дате выгрузки (первый платёж + помесячные "
               "платежи за отученные месяцы); payments_count: число транзакций")

    # --- категориальные пропуски
    df["quality"] = df["quality"].replace("F", np.nan).fillna("Not rated")
    df["lost_reason"] = df["lost_reason"].fillna("Not lost")
    df["deal_owner_name"] = df["deal_owner_name"].fillna("Unassigned")
    df["campaign"] = df["campaign"].fillna("(no campaign)")
    df["page"] = df["page"].fillna("(not set)")
    df["education_type"] = df["education_type"].replace("#REF!", np.nan).fillna("Unknown")
    df["payment_type"] = df["payment_type"].fillna("Unknown")
    df["product"] = df["product"].fillna("Unknown")
    log.append("Категориальные пропуски заполнены явными метками: quality 'Not rated' (+3 строки 'F'), "
               "lost_reason 'Not lost', product/payment_type/education_type 'Unknown', "
               "campaign '(no campaign)'; '#REF!' — ошибка Excel -> Unknown")

    df["city"] = df["city"].map(normalize_city)
    df["country"] = df["city"].map(lambda c: "Unknown" if c == "Unknown"
                                   else NON_GERMAN_CITIES.get(c, "Germany"))
    df["german_level"] = df["level_of_deutsch"].map(normalize_german)
    df = df.drop(columns="level_of_deutsch")
    log.append("city: '-' -> Unknown, адреса -> город, единое написание; добавлен country. "
               "level_of_deutsch (б1, в1, 'B1 (до февраля)') -> german_level A0..C2/Unknown")

    for col in ["deal_owner_name", "quality", "stage", "lost_reason", "source", "product",
                "education_type", "payment_type", "german_level", "country"]:
        df[col] = df[col].astype("category")
    df["month"] = df["created_time"].dt.to_period("M").dt.to_timestamp()
    return df.reset_index(drop=True), log


def run_etl(save: bool = True) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    raw = load_raw()
    cleaners = {"Contacts": clean_contacts, "Calls": clean_calls,
                "Spend": clean_spend, "Deals": clean_deals}
    clean, logs = {}, {}
    for name, fn in cleaners.items():
        clean[name], logs[name] = fn(raw[name])
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        for name, df in clean.items():
            df.to_csv(PROCESSED / f"{name.lower()}_clean.csv", index=False)
            df.to_pickle(PROCESSED / f"{name.lower()}_clean.pkl")
    return clean, logs


def load_clean() -> dict[str, pd.DataFrame]:
    """Быстрая загрузка очищенных таблиц из pickle (типы сохраняются)."""
    return {name: pd.read_pickle(PROCESSED / f"{name.lower()}_clean.pkl")
            for name in ["Contacts", "Calls", "Spend", "Deals"]}


if __name__ == "__main__":
    _, logs = run_etl()
    for name, lines in logs.items():
        print(f"\n== {name}")
        print("\n".join(f"- {l}" for l in lines))
