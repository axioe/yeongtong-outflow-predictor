# %% [markdown]
# # 06. Streamlit app: outflow prediction and risk grade by dong (report 4. prototyping)
# - 수업 노트북(ml2_EX "streamlit을 활용한 당뇨병 예측")의 흐름을 따름:
#   joblib으로 저장한 모델 불러오기 → st.slider로 입력 받기 → st.button으로 예측 → st.write로 결과 출력
# - 모델은 04번에서 저장한 model/final_rf_pipeline.joblib, 위험등급 계산은 05번과 같은 방식
# - 화면 1: 행정동·월을 고르고 입력값(가격 변동률 등)을 바꿔 가며 유출 확률·위험등급 확인
# - 화면 2: 선택한 월의 12개 행정동 위험 순위
# - 앱 화면 글자는 보고서 캡처용으로 한국어 사용 (코드 이름은 영어 유지)
#
# 사전설치 : pip install streamlit
# 실행 : 터미널에서 ML 폴더로 이동 후  streamlit run code/06_streamlit_app.py
#        (# %% 셀 실행이나 python 실행으로는 화면이 뜨지 않음)

# %%
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "데이터_csv"
MODEL_DIR = BASE_DIR / "model"

# 폰트지정 (그래프에 한글이 들어가므로 필요)
plt.rcParams["font.family"] = "Malgun Gothic"
# 마이너스 부호 깨짐 지정
plt.rcParams["axes.unicode_minus"] = False

# %% [CONFIG]
TARGET = "outflow"
TRAIN_END = "2025-09"
GRADE_QUANTILES = (0.5, 0.8)      # 05번과 같은 등급 기준 (학습 구간 예상 감소의 50%, 80% 분위수)
GRADES = ["LOW", "MEDIUM", "HIGH"]
GRADE_COLORS = {"LOW": "#0ca30c", "MEDIUM": "#fab219", "HIGH": "#d03b3b"}

# 화면 1에서 슬라이더로 바꿔 볼 수 있는 변수 (나머지 변수는 선택한 행의 실제 값을 그대로 사용)
SLIDER_FEATURES = {
    "price_chg_1m": "최근 1개월 가격 변동률 (0.01 = 1%)",
    "price_chg_3m": "최근 3개월 가격 변동률 (0.01 = 1%)",
    "price_chg_1m_lag2": "2개월 전 월간 가격 변동률 (0.01 = 1%)",
    "price_m2": "㎡당 중위 가격 (만원)",
    "trade_cnt": "전월 거래량 (건)",
    "pop_chg_rate_lag1": "전월 인구 증감률 (%)",
}

# %% Load model and data
pipeline = joblib.load(MODEL_DIR / "final_rf_pipeline.joblib")
meta = joblib.load(MODEL_DIR / "features_meta.joblib")
FEATURES = meta["cat_features"] + meta["num_features"]
THRESHOLD = meta["threshold"]

df = pd.read_csv(DATA_DIR / "영통구_행정동별_월별_학습데이터셋.csv")
df = df.dropna(subset=FEATURES).reset_index(drop=True)
df["outflow_prob"] = pipeline.predict_proba(df[FEATURES])[:, 1]

# %% Risk grade (05번과 같은 계산: 감소율·등급 기준은 학습 구간에서만 계산)
train = df[df["ym"] <= TRAIN_END]
outflow_rate = (train[train[TARGET] == 1]
                .assign(chg_rate=lambda d: d["pop_chg"] / d["population_lag1"])
                .groupby("dong")["chg_rate"].mean().abs())
df["avg_outflow_rate"] = df["dong"].map(outflow_rate)
df["expected_decrease"] = df["outflow_prob"] * df["population_lag1"] * df["avg_outflow_rate"]
q_low, q_high = df.loc[df["ym"] <= TRAIN_END, "expected_decrease"].quantile(GRADE_QUANTILES)


def to_grade(x):
    return pd.cut(x, bins=[-np.inf, q_low, q_high, np.inf], labels=GRADES, right=False).astype(str)


df["risk_grade"] = to_grade(df["expected_decrease"])

dongs = sorted(df["dong"].unique())
months = sorted(df["ym"].unique(), reverse=True)

# %% Streamlit app
st.title("영통구 행정동별 인구 유출 예측")
st.write("아파트 매매가 변동과 전월 인구 정보를 이용해, 행정동의 다음 달 인구가 줄어들지(유출) 예측합니다.")
st.write(f"모델: 규제한 Random Forest (트리 200개, 최대 깊이 2) · "
         f"위험등급 기준: 예상 감소 {q_low:.1f}명 이상 MEDIUM, {q_high:.1f}명 이상 HIGH")

# ---------- 화면 1: 행정동·월별 예측 ----------
st.write("### 1. 행정동별 유출 예측")
# selectbox: 행정동 이름과 월은 숫자 범위가 아니라 정해진 목록에서 골라야 하므로 slider 대신 사용
dong = st.selectbox("행정동", dongs)
ym = st.selectbox("예측 대상 월", months)
row = df[(df["dong"] == dong) & (df["ym"] == ym)].iloc[0]

st.write("아래 값을 바꿔 볼 수 있습니다. (기본값 = 예측 대상 월의 직전 월까지 확인된 실제 값)")
inputs = {}
for col, label in SLIDER_FEATURES.items():
    lo, hi = float(df[col].min()), float(df[col].max())
    step = 1.0 if col in ("price_m2", "trade_cnt") else 0.001
    inputs[col] = st.slider(label, min_value=lo, max_value=hi, value=float(row[col]), step=step)

# 예측하기 버튼
if st.button("예측하기"):
    # 슬라이더로 바꾼 값 + 나머지 변수는 선택한 행의 실제 값
    input_data = pd.DataFrame([{**{c: row[c] for c in FEATURES}, **inputs}])[FEATURES]
    prob = pipeline.predict_proba(input_data)[0, 1]
    expected = prob * row["population_lag1"] * row["avg_outflow_rate"]
    grade = to_grade(pd.Series([expected]))[0]

    # 결과 출력
    if prob >= THRESHOLD:
        st.write(f"예측 결과: **유출** — {ym} {dong}의 인구가 줄어들 가능성이 높습니다.")
    else:
        st.write(f"예측 결과: **유입·유지** — {ym} {dong}의 인구가 유지되거나 늘어날 가능성이 높습니다.")
    st.write(f"- 유출 확률: **{prob:.1%}**")
    st.write(f"- 예상 인구 감소: **{expected:.1f}명** "
             f"(= 유출 확률 × 전월 인구 {row['population_lag1']:,.0f}명 × 동 평균 감소율 {row['avg_outflow_rate']:.3%})")
    st.write(f"- 위험등급: **{grade}**")
    st.write(f"- 실제 결과: 인구 증감 {row['pop_chg']:+,.0f}명 "
             f"({'유출' if row[TARGET] == 1 else '유입·유지'})")

# ---------- 화면 2: 선택한 월의 행정동 위험 순위 ----------
st.write("### 2. 행정동 위험 순위")
rank_month = st.selectbox("조회 월", months, key="rank_month")
rank = (df[df["ym"] == rank_month]
        .sort_values("expected_decrease", ascending=False)
        [["dong", "outflow_prob", "population_lag1", "expected_decrease", "risk_grade", "pop_chg"]])
# dataframe: 12개 행정동의 결과를 표로 한 번에 보여주기 위해 사용 (화면 표시용 컬럼명만 한국어)
st.dataframe(rank.round({"outflow_prob": 3, "expected_decrease": 1})
             .rename(columns={"dong": "행정동", "outflow_prob": "유출 확률", "population_lag1": "전월 인구",
                              "expected_decrease": "예상 감소(명)", "risk_grade": "위험등급",
                              "pop_chg": "실제 증감(명)"}),
             hide_index=True, height=35 * (len(rank) + 1) + 3)     # 12개 동이 스크롤 없이 모두 보이도록

plot_df = rank.sort_values("expected_decrease")
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(plot_df["dong"], plot_df["expected_decrease"], color=[GRADE_COLORS[g] for g in plot_df["risk_grade"]])
ax.axvline(q_low, color="gray", linestyle="--", linewidth=1)
ax.axvline(q_high, color="gray", linestyle="--", linewidth=1)
ax.set_title(f"행정동별 예상 인구 감소 ({rank_month})")
ax.set_xlabel("예상 감소 (명)  ·  점선: MEDIUM / HIGH 기준")
plt.tight_layout()
# pyplot: matplotlib 그래프를 Streamlit 화면에 표시하기 위해 사용
st.pyplot(fig)
