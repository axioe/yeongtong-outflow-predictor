# %% [markdown]
# # 01. Build dong × month training dataset
# - 입력: `데이터_csv/` 의 실거래가, 단지-행정동 매핑, 행정동별 월별 인구현황
# - 출력: `데이터_csv/영통구_행정동별_월별_학습데이터셋.csv`
# - 한 행 = 예측 대상 월(t) × 행정동, 목표 `outflow` = t월 인구 증감 < 0 이면 1
# - 독립변수는 모두 t-1월까지의 정보로 계산 (예측 시점 이후 정보가 섞이지 않도록)

# %%
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "데이터_csv"
OUT_PATH = DATA_DIR / "영통구_행정동별_월별_학습데이터셋.csv"

NEW_BUILD_YEARS = 5                 # 신축 기준: 계약연도 - 건축년도 <= 5
FIRST_TRADE_MONTH = "2023-11"       # 2023.10은 거래가 이틀치(10/30~31)뿐인 부분월이라 제외
LAST_TARGET_MONTH = "2026-09"       # 인구 현황의 마지막 월

# %% Load source data
# 원본 CSV의 컬럼명은 한국어이므로 그대로 읽은 뒤 아래에서 영어 변수로 바꿔 사용
trades = pd.read_csv(DATA_DIR / "영통구_아파트_실거래가_2023.10-2026.10.csv", dtype={"번지": str})
mapping = pd.read_csv(DATA_DIR / "영통구_단지_행정동_매핑.csv", dtype={"번지": str})
pop = pd.read_csv(DATA_DIR / "영통구_행정동별_월별_인구현황.csv")

# %% 1. Attach dong to each trade
# 단지(법정동·번지·단지명·도로명)를 키로 행정동을 붙이고, 해제 거래와 부분월은 제외
key = ["법정동", "번지", "단지명", "도로명"]
trades = trades.merge(mapping[key + ["행정동"]], on=key, how="left", validate="many_to_one")
assert trades["행정동"].notna().all(), "some complexes are not mapped to a dong"

trades = trades.rename(columns={"행정동": "dong", "단지키": "complex", "건축년도": "build_year",
                                "㎡당가격(만원)": "price_m2"})
trades["ym"] = pd.PeriodIndex(trades["계약년월"].astype(str), freq="M")
trades = trades[(trades["해제여부"] == "N") & (trades["ym"] >= pd.Period(FIRST_TRADE_MONTH, "M"))].copy()
trades["is_new"] = (pd.to_datetime(trades["계약일자"]).dt.year - trades["build_year"] <= NEW_BUILD_YEARS).astype(int)

dongs = sorted(pop.loc[pop["구분"] == "행정동", "행정동"].unique())
months = pd.period_range(FIRST_TRADE_MONTH, LAST_TARGET_MONTH, freq="M")
grid = pd.MultiIndex.from_product([dongs, months], names=["dong", "ym"])

# %% 2. Monthly trade aggregates per dong
# 행정동 × 월 단위로 거래량, 중위 ㎡당 가격, 신축 거래 비중, 평균 건축년도를 집계
g = trades.groupby(["dong", "ym"])
monthly = pd.DataFrame({
    "trade_cnt": g.size(),
    "price_m2": g["price_m2"].median(),
    "new_ratio": g["is_new"].mean(),
    "build_year_avg": g["build_year"].mean(),
}).reindex(grid)
monthly["trade_cnt"] = monthly["trade_cnt"].fillna(0).astype(int)   # 거래가 없는 달은 0건, 가격·특성은 NaN 유지

# %% 3. Complex-based price change rates
# s월과 s-k월에 모두 거래된 단지끼리 중위 ㎡당 가격 비율을 구하고, 행정동별 중앙값을 사용
# (같은 단지끼리 비교해서 신축·구축 거래 비중 변화가 가격 변동으로 잘못 잡히는 것을 줄임)
complex_price = (trades.groupby(["dong", "complex", "ym"])["price_m2"].median()
                   .unstack("ym").reindex(columns=months))
for k in (1, 3, 12):
    ratio = complex_price / complex_price.shift(k, axis=1) - 1
    chg = ratio.groupby(level="dong").median().stack()
    chg.index.names = ["dong", "ym"]
    monthly[f"chg_{k}m"] = chg.reindex(grid)


# %% 4. Shift to t-1 and build lag features
def lag(col, n):
    # 행정동별로 n개월 이전 값을 가져옴
    return monthly.groupby(level="dong")[col].shift(n)


X = pd.DataFrame(index=grid)
X["price_m2"] = lag("price_m2", 1)
X["price_chg_1m"] = lag("chg_1m", 1)
X["price_chg_3m"] = lag("chg_3m", 1)
X["price_chg_12m"] = lag("chg_12m", 1)
X["price_chg_1m_lag2"] = lag("chg_1m", 2)      # 2개월 전 월간 가격 변동률
X["price_chg_1m_lag3"] = lag("chg_1m", 3)      # 3개월 전 월간 가격 변동률
X["trade_cnt"] = lag("trade_cnt", 1)
X["trade_cnt_3m"] = (monthly.groupby(level="dong")["trade_cnt"]
                     .transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean()))   # t-1~t-3 평균
X["new_ratio"] = lag("new_ratio", 1)
X["build_year_avg"] = lag("build_year_avg", 1)

# %% 5. Population features (t-1) and target (t)
p = pop[pop["구분"] == "행정동"].rename(columns={
    "행정동": "dong", "인구증감률(%)": "pop_chg_rate", "인구_합계": "population",
    "세대당인구": "hh_size", "인구증감_전월대비": "pop_chg"})
p["ym"] = pd.PeriodIndex(p["기준년월"].astype(str), freq="M")
p = p.set_index(["dong", "ym"]).sort_index()
prev = p.groupby(level="dong")[["pop_chg_rate", "population", "hh_size"]].shift(1)

X["pop_chg_rate_lag1"] = prev["pop_chg_rate"].reindex(grid)
X["population_lag1"] = prev["population"].reindex(grid)
X["hh_size_lag1"] = prev["hh_size"].reindex(grid)
X["pop_chg"] = p["pop_chg"].reindex(grid)      # 목표 계산·예상 인구 감소용 (모델 입력으로 쓰면 안 됨)
X["outflow"] = (X["pop_chg"] < 0).astype(int)

# %% 6. Tidy up and save
# t-1월 거래 집계가 존재하는 2023.12부터 저장 (앞쪽의 구조적 결측은 채우지 않고 NaN으로 둠)
X = X.reset_index()
X["month"] = X["ym"].dt.month
X["ym"] = X["ym"].astype(str)
cols = ["ym", "dong", "month", "price_m2", "price_chg_1m", "price_chg_3m", "price_chg_12m",
        "price_chg_1m_lag2", "price_chg_1m_lag3", "trade_cnt", "trade_cnt_3m", "new_ratio",
        "build_year_avg", "pop_chg_rate_lag1", "population_lag1", "hh_size_lag1", "pop_chg", "outflow"]
X = X[X["ym"] >= "2023-12"][cols].reset_index(drop=True)
X.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print(f"Saved: {OUT_PATH.name} {X.shape}, {X['ym'].min()} ~ {X['ym'].max()}")
print("Missing ratio:\n", X.isna().mean().round(3)[lambda s: s > 0])
print("Outflow ratio:", round(X["outflow"].mean(), 3))
