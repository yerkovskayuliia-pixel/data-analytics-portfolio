# %% [markdown]
# # 01 · Очистка и подготовка данных CRM
#
# **Проект:** анализ CRM онлайн-школы программирования X (июль 2023 – июнь 2024).
# **Цель ноутбука:** превратить 4 «сырые» выгрузки CRM в чистые, связанные между собой таблицы,
# на которых можно считать воронку, эффективность маркетинга и юнит-экономику.
#
# | Таблица | Что содержит | Ключ |
# |---|---|---|
# | Contacts | контакты (лиды) | `Id` |
# | Calls | звонки менеджеров | `CONTACTID` → Contacts |
# | Spend | рекламные расходы по дням | `Source`, `Campaign` → Deals |
# | Deals | потенциальные сделки | `Contact Name` → Contacts |
#
# Вся логика очистки вынесена в модуль [`src/cleaning.py`](../src/cleaning.py), чтобы её можно было
# переиспользовать и тестировать. Здесь — проверка исходных данных, запуск ETL и контроль результата.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent / "src"))
import cleaning as etl

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)

# %% [markdown]
# ## 1. Исходные данные
#
# Берём `.xlsx`, а не `.csv`: в CSV-выгрузке кириллица в поле *Level of Deutsch* превратилась в `?`,
# а 19-значные ID округлены как float (связь таблиц ломается). Все ID читаем как строки.

# %%
raw = etl.load_raw()
overview = pd.DataFrame({
    name: {"строк": len(df), "столбцов": df.shape[1],
           "полных дубликатов": df.duplicated().sum(),
           "ячеек с пропусками, %": round(df.isna().mean().mean() * 100, 1)}
    for name, df in raw.items()
}).T
overview

# %%
def missing_report(df):
    m = df.isna().sum()
    return (pd.DataFrame({"пропусков": m, "%": (m / len(df) * 100).round(1),
                          "тип": df.dtypes.astype(str)})
            .query("пропусков > 0").sort_values("%", ascending=False))

missing_report(raw["Deals"])

# %% [markdown]
# **Что видно в Deals до очистки:**
# - `Payment Type`, `Months of study`, `Course duration`, `City`, `Level of Deutsch` заполнены у малой доли
#   сделок — менеджер заполняет их только на поздних стадиях воронки. Это не ошибка, а особенность процесса:
#   пропуск означает «до этого этапа сделка не дошла», поэтому такие поля **не удаляем и не заполняем средним**.
# - Суммы хранятся текстом в разных форматах: `1000`, `€ 3.500,00`, `$11.000,00`.
# - `SLA` — смесь `time` (чч:мм:сс) и `timedelta` (больше суток).

# %%
raw["Deals"][["Initial Amount Paid", "Offer Total Amount", "SLA", "Level of Deutsch", "City"]].dropna().sample(8, random_state=7)

# %%
missing_report(raw["Calls"])

# %% [markdown]
# ## 2. Правила очистки и их обоснование
#
# | # | Таблица | Проблема | Решение | Обоснование |
# |---|---|---|---|---|
# | 1 | Deals | `Lost Reason = Duplicate` | удалить | по FAQ это технический дубль лида; иначе завышаем число лидов и занижаем конверсию |
# | 2 | Deals | `Closing Date` < `Created Time` | → `NaT` | закрыть сделку раньше создания невозможно (FAQ: ошибка, решает аналитик) |
# | 3 | Deals | `Initial Amount Paid` > `Offer Total Amount` | 2 строки с offer 1 200 по 11-мес. курсу — опечатка в offer; остальные — initial ограничен offer | первый платёж не может превышать стоимость курса; консервативная оценка выручки |
# | 4 | Deals | offer = 0 / 1 / 9 € | флаг `is_demo` | по FAQ это демо-доступы за символическую плату — не студенты |
# | 5 | Deals | `Payment Done`, но `Months of study` пуст | не считаем студентом | FAQ: «если месяц не заполнен — студент не учится, деньги учитывать нельзя» |
# | 6 | Deals | суммы текстом, SLA смешанного типа | → float / секунды | для расчётов нужны числа |
# | 7 | Deals | `Level of Deutsch` свободным текстом (б1, в1, «B1 (до февраля)») | → CEFR `A0…C2` | кириллические б/в = B; берём текущий уровень (первый упомянутый) |
# | 8 | Deals | города: `-`, адреса, разное написание | → `city`, `country` | для географического анализа |
# | 9 | Deals | `Content`, `Term` | удалить | детализация объявления; кампания и источник сохраняются |
# | 10 | Deals | категориальные пропуски | явные метки `Unknown`, `Not rated`, `Not lost` | пропуск сам по себе информативен; удаление строк исказило бы воронку |
# | 11 | Calls | `Dialled Number`, `Tag` пустые на 100% | удалить | нет информации |
# | 12 | Calls | нет длительности | удалить строки (83) | звонок без длительности нельзя анализировать |
# | 13 | Calls | `Outgoing Call Status` пуст у входящих | → `Not applicable` | это не пропуск, а «не применимо» |
# | 14 | Spend | 917 полных дублей | удалить | все — нулевые строки, на суммы не влияют |
# | 15 | Spend | `Campaign` пуст | → `(no campaign)` | расход реальный, удалять нельзя |
#
# **Выручка.** По FAQ факт оплаты — статус `Payment Done`, а сумма первого платежа — `Initial Amount Paid`.
# Остаток курса оплачивается помесячно, поэтому выручка, полученная к дате выгрузки:
#
# $$revenue = initial + (months\_of\_study - 1) \cdot \frac{offer - initial}{course\_duration - 1},\quad revenue \le offer$$
#
# При единовременной оплате (`initial = offer`) выручка = `offer`.

# %%
clean, logs = etl.run_etl(save=True)
for name, lines in logs.items():
    print(f"\n=== {name}")
    for line in lines:
        print(" •", line)

# %% [markdown]
# ## 3. Контроль результата

# %%
pd.DataFrame({
    name: {"строк до": len(raw[name]), "строк после": len(df),
           "столбцов": df.shape[1], "полных дубликатов": df.duplicated().sum()}
    for name, df in clean.items()
}).T

# %%
deals = clean["Deals"]
deals.dtypes.to_frame("dtype").T

# %%
checks = {
    "id сделок уникальны": deals["id"].is_unique,
    "нет closing_date < created_time": not (deals["closing_date"] < deals["created_time"].dt.normalize()).any(),
    "initial <= offer везде": not (deals["initial_amount_paid"] > deals["offer_total_amount"]).any(),
    "у студентов есть months_of_study": deals.loc[deals["is_student"], "months_of_study"].notna().all(),
    "у студентов есть продукт": deals.loc[deals["is_student"], "product"].ne("Unknown").all(),
    "выручка только у студентов": (deals.loc[~deals["is_student"], "revenue"] == 0).all(),
    "contact_id сделок есть в Contacts, %": round(deals["contact_id"].isin(clean["Contacts"]["id"]).mean() * 100, 1),
    "contact_id звонков есть в Contacts, %": round(clean["Calls"]["contact_id"].dropna().isin(clean["Contacts"]["id"]).mean() * 100, 1),
}
pd.Series(checks, name="проверка").to_frame()

# %%
deals["german_level"].value_counts().to_frame("сделок")

# %% [markdown]
# ## 4. Итог
#
# - Сделок после очистки: **19 822** (удалено 1 771 технический дубль), звонков: **95 791**, строк расходов: **19 862**.
# - Студентов (оплата + заполнен срок обучения, не демо): **837** из 856 сделок `Payment Done`.
# - Связь таблиц работает: 99.7% сделок и 100% звонков с контактом находят свой контакт.
# - Очищенные таблицы сохранены в `data/processed/` в двух форматах: **CSV** (для Tableau/Sheets) и
#   **pickle** (для Python: сохраняет типы `datetime`, `category`, `bool` и загружается мгновенно).
#
# ➡️ Следующий шаг — [02_eda_analysis.ipynb](02_eda_analysis.ipynb)
