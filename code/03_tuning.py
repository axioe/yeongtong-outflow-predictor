# %% [markdown]
# # 03. Hyperparameter tuning with GridSearchCV (report 3.2.3 / 3.2.4)
# - 데이터·분할·전처리는 02번과 동일: 학습 2024.03~2025.09 / 테스트 2025.10~2026.09
# - 수업 노트북(mlEx "랜덤포레스트", ml2_EX "자전거 수요량 예측")처럼 GridSearchCV로 조합을 모두 시도
# - 튜닝은 학습 구간 안에서만 진행. 테스트 구간은 마지막 성능 확인에만 사용
# - 교차검증은 노트북의 cv=3 대신 TimeSeriesSplit 사용 (이유는 아래 CV 셀 주석 참고)
# - 탐색 대상·범위는 [CONFIG] 셀의 SEARCHES만 고치면 됨
# 사전설치 : pip install pandas scikit-learn xgboost

# %%
import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, roc_curve
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "데이터_csv"
RESULT_PATH = DATA_DIR / "results_tuning.csv"

# %% [CONFIG] features, split, CV, search grids
TARGET = "outflow"
CAT_FEATURES = ["dong", "month"]
NUM_FEATURES = [
    "price_m2",
    "price_chg_1m", "price_chg_3m",
    "price_chg_1m_lag2", "price_chg_1m_lag3",
    "trade_cnt", "trade_cnt_3m",
    "new_ratio", "build_year_avg",
    "pop_chg_rate_lag1", "population_lag1", "hh_size_lag1",
]
TRAIN_END = "2025-09"
THRESHOLD = 0.5
RANDOM_STATE = 42

N_SPLITS = 4            # 교차검증 폴드 수
VALID_MONTHS = 3        # 폴드마다 검증에 쓰는 개월 수
SCORING = "roc_auc"     # 최적 조합 선택 기준 (정확도로 바꾸려면 "accuracy", F1은 "f1")
N_JOBS = -1             # n_jobs=-1: 모든 CPU 코어 사용. PicklingError가 나면 1로 바꾸기

# 탐색할 모델과 그리드. 키 = 결과 표에 나올 이름, 값 = (분류기, param_grid)
# 파라미터 이름 앞의 "model__"은 Pipeline 안의 분류기 단계 이름
SEARCHES = {
    # 1차: 노트북 랜덤포레스트 예제와 같은 항목을 넓게 탐색
    "Random Forest (broad)": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100, 200, 500],      # 트리의 개수
            "model__max_depth": [None, 5, 10, 20],       # 트리의 최대 깊이
            "model__min_samples_split": [2, 5, 10],      # 노드를 분할하기 위한 최소 샘플 수
            "model__min_samples_leaf": [1, 2, 4],        # 리프 노드에 있어야 하는 최소 샘플 수
        },
    ),
    # 2차: 학습 데이터가 228행으로 적으므로 트리를 얕게·잎을 크게 제한해 과적합 억제
    "Random Forest (regularized)": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [200, 500],
            "model__max_depth": [2, 3, 4],
            "model__min_samples_split": [10, 20, 40],
            "model__min_samples_leaf": [5, 10, 20],
        },
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {
            "model__C": [0.001, 0.01, 0.1, 1, 10],      # C 작음: 규제 강함(단순한 경계), C 큼: 규제 약함(복잡한 경계)
        },
    ),
    # 노트북 "자전거 수요량 예측"의 XGBoost 그리드 항목을 분류 문제에 맞게 사용
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1],                # 각 트리 학습에 쓰는 데이터 비율, 과적합 방지
            "model__colsample_bytree": [0.8, 1],         # 각 트리에서 쓰는 변수 비율, 과적합 방지
            "model__reg_lambda": [1, 10],                # L2 정규화 (노트북의 'lambda'와 같은 항목)
        },
    ),
}

# %% Load data and split by time (02번과 동일)
df = pd.read_csv(DATA_DIR / "영통구_행정동별_월별_학습데이터셋.csv")
df = df.dropna(subset=CAT_FEATURES + NUM_FEATURES)
df = df.sort_values(["ym", "dong"]).reset_index(drop=True)     # 시간 순서대로 정렬 (TimeSeriesSplit에 필요)

train = df[df["ym"] <= TRAIN_END].reset_index(drop=True)
test = df[df["ym"] > TRAIN_END].reset_index(drop=True)
X_train, y_train = train[CAT_FEATURES + NUM_FEATURES], train[TARGET]
X_test, y_test = test[CAT_FEATURES + NUM_FEATURES], test[TARGET]
print(f"Train : {len(train)} rows ({train['ym'].min()} ~ {train['ym'].max()})")
print(f"Test  : {len(test)} rows ({test['ym'].min()} ~ {test['ym'].max()})")

# %% Time-series CV folds
# 노트북의 cv=3(또는 5)은 데이터를 순서와 상관없이 나누므로, 미래 월로 학습하고 과거 월을 검증하는 폴드가 생김
# → 시계열에서는 TimeSeriesSplit으로 "과거 월로 학습 → 바로 다음 월들로 검증"하도록 나눔
# 매달 행정동 12개 행이 있으므로 test_size = 12 × VALID_MONTHS 로 두면 폴드가 월 경계에서 정확히 잘림
n_dongs = train["dong"].nunique()
assert (train.groupby("ym").size() == n_dongs).all(), "every month must have all dongs"
CV = TimeSeriesSplit(n_splits=N_SPLITS, test_size=n_dongs * VALID_MONTHS)

for i, (tr, va) in enumerate(CV.split(X_train), 1):
    print(f"Fold {i}: train {train['ym'].iloc[tr].min()} ~ {train['ym'].iloc[tr].max()} "
          f"| valid {train['ym'].iloc[va].min()} ~ {train['ym'].iloc[va].max()}")


# %% Helpers (02번과 동일)
def evaluate(y_true, y_prob):
    y_pred = (y_prob >= THRESHOLD).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1": report["1"]["f1-score"],
        "ROC-AUC": auc(fpr, tpr),
    }


preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
])

# %% Run grid searches
rows, best_models = [], {}
for name, (model, param_grid) in SEARCHES.items():
    start = time.time()
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=CV, scoring=SCORING,
                               n_jobs=N_JOBS)
    grid_search.fit(X_train, y_train)          # 최적 조합을 찾은 뒤 학습 구간 전체로 다시 학습함
    best_models[name] = grid_search.best_estimator_

    tr = evaluate(y_train, grid_search.best_estimator_.predict_proba(X_train)[:, 1])
    te = evaluate(y_test, grid_search.best_estimator_.predict_proba(X_test)[:, 1])
    params = {k.replace("model__", ""): v for k, v in grid_search.best_params_.items()}
    rows.append({
        "model": name,
        f"best_cv_{SCORING}": grid_search.best_score_,
        **{f"test_{k}": v for k, v in te.items()},
        "train_Accuracy": tr["Accuracy"], "train_ROC-AUC": tr["ROC-AUC"],
        "gap_Accuracy": tr["Accuracy"] - te["Accuracy"], "gap_ROC-AUC": tr["ROC-AUC"] - te["ROC-AUC"],
        "n_combinations": len(grid_search.cv_results_["params"]),
        "seconds": round(time.time() - start, 1),
        "best_params": params,
    })
    print(f"[{name}] best CV {SCORING} {grid_search.best_score_:.4f} | {params}")

res = pd.DataFrame(rows)

# %% Result 1: best hyperparameters (report 3.2.3)
for r in rows:
    print(f"\n{r['model']}")
    for k, v in r["best_params"].items():
        print(f"  {k:<20} {v}")

# %% Result 2: CV score, test performance, overfitting gap (report 3.2.3 / 3.2.4)
cols = ["model", f"best_cv_{SCORING}", "test_Accuracy", "test_Precision", "test_Recall", "test_F1",
        "test_ROC-AUC", "train_Accuracy", "train_ROC-AUC", "gap_Accuracy", "gap_ROC-AUC",
        "n_combinations", "seconds"]
pd.set_option("display.width", 250)
print(res[cols].round(4).to_string(index=False))

# %% Save results
out = res.copy()
out["best_params"] = out["best_params"].astype(str)
out.round(4).to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
print("Saved:", RESULT_PATH)
