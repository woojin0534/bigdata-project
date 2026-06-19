import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

ML_BASE   = ["platform", "primary_genre", "type", "release_year", "duration_minutes"]
ML_CAT    = ["platform", "primary_genre", "type"]
ML_TARGET = "imdb_rating"


def _build_dataset(df):
    """OHE 변환 + 파생 특성. train_model과 예측 시 동일 로직."""
    df_ml = df[ML_BASE + [ML_TARGET]].copy()

    df_ml["duration_missing"] = df_ml["duration_minutes"].isna().astype(int)

    genre_mean   = df_ml.groupby("primary_genre")["duration_minutes"].transform("mean")
    overall_mean = df_ml["duration_minutes"].mean()
    df_ml["duration_minutes"] = df_ml["duration_minutes"].fillna(genre_mean).fillna(overall_mean)

    df_ml = df_ml.dropna(subset=[ML_TARGET])

    known_cats = {col: sorted(df_ml[col].dropna().unique().tolist()) for col in ML_CAT}

    # LabelEncoding 대신 OneHotEncoding — 범주 간 가상 순서 부여 없음
    df_enc = pd.get_dummies(df_ml.drop(columns=[ML_TARGET]), columns=ML_CAT, dtype=int)
    y = df_ml[ML_TARGET].values
    feature_cols = list(df_enc.columns)

    return df_enc, y, feature_cols, known_cats


@st.cache_resource
def train_model(df):
    X, y, feature_cols, known_cats = _build_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)

    # 5-Fold 교차 검증
    cv_scores  = cross_val_score(rf, X_train, y_train, cv=5, scoring="r2")
    cv_r2_mean = float(cv_scores.mean())
    cv_r2_std  = float(cv_scores.std())

    compare_list = [("선형 회귀", LinearRegression()), ("랜덤 포레스트", rf)]
    xgb_available = False
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbosity=0)
        xgb.fit(X_train, y_train)
        compare_list.append(("XGBoost", xgb))
        xgb_available = True
    except ImportError:
        pass

    cmp_rows = []
    for name, m in compare_list:
        if name == "선형 회귀":
            m.fit(X_train, y_train)
        p = m.predict(X_test)
        cmp_rows.append({
            "모델":  name,
            "MAE":  round(mean_absolute_error(y_test, p), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, p))), 3),
            "R²":   round(r2_score(y_test, p), 3),
        })
    df_compare = pd.DataFrame(cmp_rows)

    tune_rows = []
    for n_est in [50, 100, 200, 300]:
        m_t = RandomForestRegressor(n_estimators=n_est, random_state=42, n_jobs=-1)
        m_t.fit(X_train, y_train)
        p_t = m_t.predict(X_test)
        tune_rows.append({
            "n_estimators": n_est,
            "MAE":  round(mean_absolute_error(y_test, p_t), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, p_t))), 3),
            "R²":   round(r2_score(y_test, p_t), 3),
        })
    df_tune = pd.DataFrame(tune_rows)

    return (
        rf, feature_cols, known_cats,
        mae, rmse, r2,
        cv_r2_mean, cv_r2_std,
        len(X_train),
        df_compare, df_tune,
        xgb_available,
    )
