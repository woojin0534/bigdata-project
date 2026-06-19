import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import PLATFORM_COLORS, load_data
from src.features import train_model


df, df_year = load_data()

YEAR_MIN = int(df_year["release_year"].min()) if not df_year.empty else 1990
YEAR_MAX = int(df_year["release_year"].max()) if not df_year.empty else 2025

st.title("🤖 IMDB 평점 예측기 (머신러닝)")
st.caption(
    "플랫폼 · 장르 · 유형 · 출시 연도 · 런타임을 입력하면 "
    "RandomForest 모델이 예상 IMDB 평점을 예측합니다. "
    "선형 회귀 · 랜덤 포레스트 · XGBoost 3모델의 성능도 비교합니다."
)
st.divider()

with st.spinner("모델 학습 중... (최초 1회만 실행됩니다)"):
    (
        model, feature_cols, known_cats,
        mae, rmse, r2,
        cv_r2_mean, cv_r2_std,
        n_train,
        df_compare, df_tune,
        xgb_available,
    ) = train_model(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("학습 데이터", f"{n_train:,}건")
m2.metric("평균 절대 오차 (MAE)", f"{mae:.3f}점")
m3.metric("RMSE", f"{rmse:.3f}점")
m4.metric("결정계수 (R²)", f"{r2:.3f}")

st.markdown("---")

# 5-Fold 교차 검증
st.subheader("📐 5-Fold 교차 검증 (Cross-Validation)")
st.caption(
    "학습 데이터를 5개 구간으로 나눠 순환 검증합니다. "
    "홀드아웃 R²와 CV R² 차이가 작을수록 과적합 없이 일반화됩니다."
)
cv1, cv2 = st.columns(2)
cv1.metric("CV R² 평균 (5-Fold)", f"{cv_r2_mean:.3f}")
cv2.metric("CV R² 표준편차", f"{cv_r2_std:.4f}")
if cv_r2_std < 0.05:
    st.success("✅ 표준편차 < 0.05 — 모델이 안정적으로 일반화됩니다.")
else:
    st.warning("⚠️ 표준편차가 크면 학습 데이터 구성에 따라 성능 변동이 있을 수 있습니다.")

st.markdown("---")

# 모델 비교
st.subheader("📊 모델 성능 비교 (선형회귀 vs 랜덤포레스트 vs XGBoost)")
st.caption(
    "같은 학습/테스트 데이터로 세 모델을 비교합니다. "
    "MAE·RMSE는 낮을수록, R²는 높을수록 좋습니다."
)

if not xgb_available:
    st.info(
        "ℹ️ XGBoost가 설치되어 있지 않아 비교에서 제외되었습니다. "
        "`pip install xgboost` 로 추가하면 3모델 비교가 활성화됩니다."
    )

bar_colors = [
    PLATFORM_COLORS["Netflix"],
    PLATFORM_COLORS["Amazon Prime Video"],
    PLATFORM_COLORS["Disney+"],
]
fig_cmp = px.bar(
    df_compare.melt(id_vars="모델", var_name="지표", value_name="값"),
    x="모델",
    y="값",
    color="모델",
    facet_col="지표",
    text_auto=".3f",
    color_discrete_sequence=bar_colors[: len(df_compare)],
)
fig_cmp.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig_cmp.update_layout(template="plotly_white", height=350, showlegend=False)
fig_cmp.update_traces(textposition="outside")
st.plotly_chart(fig_cmp, use_container_width=True)
st.dataframe(df_compare.set_index("모델"), use_container_width=True)

st.success(
    "💡 **인사이트**: 랜덤 포레스트와 XGBoost가 선형 회귀보다 낮은 MAE·RMSE, 높은 R²를 보입니다. "
    "IMDB 평점은 비선형 관계를 가지므로 앙상블 기법이 더 효과적입니다. "
    "단, R² 절댓값이 크지 않은 이유는 평점이 주관적 요소(감독·배우·스토리 등)에 크게 의존하기 때문입니다."
)

with st.expander("📐 하이퍼파라미터 선택 과정 (n_estimators 탐색)"):
    st.caption("RandomForest의 트리 수(n_estimators)를 50·100·200·300으로 바꿔가며 최적값을 탐색했습니다.")
    fig_tune = px.line(
        df_tune.melt(id_vars="n_estimators", var_name="지표", value_name="값"),
        x="n_estimators", y="값", color="지표", markers=True,
        facet_col="지표", title="n_estimators별 RandomForest 성능 변화",
    )
    fig_tune.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_tune.update_layout(template="plotly_white", height=280, showlegend=False)
    st.plotly_chart(fig_tune, use_container_width=True)
    st.dataframe(df_tune.set_index("n_estimators"), use_container_width=True)
    st.markdown(
        "> **선택 근거**: 200 → 300으로 늘려도 MAE 개선이 0.001 미만으로 미미합니다. "
        "학습 시간 대비 효율을 고려해 **n_estimators=200**을 최종 선택했습니다."
    )

st.markdown("---")

# Feature Importance — OHE 컬럼을 카테고리별로 합산
_label_map = {
    "release_year":     "출시 연도",
    "duration_minutes": "런타임",
    "duration_missing": "런타임 결측 여부",
}
_prefix_map = {
    "platform_":      "플랫폼",
    "primary_genre_": "장르",
    "type_":          "유형 (영화/TV)",
}

grouped_imp = {}
for col, imp in zip(feature_cols, model.feature_importances_):
    label = _label_map.get(col)
    if label is None:
        for prefix, pname in _prefix_map.items():
            if col.startswith(prefix):
                label = pname
                break
    grouped_imp[label or col] = grouped_imp.get(label or col, 0.0) + imp

feat_imp = (
    pd.DataFrame(list(grouped_imp.items()), columns=["특성", "중요도"])
    .sort_values("중요도", ascending=True)
)

fig_imp = px.bar(
    feat_imp,
    x="중요도",
    y="특성",
    orientation="h",
    text=feat_imp["중요도"].map(lambda v: f"{v:.1%}"),
    title="📊 예측에 영향을 미치는 특성 중요도 (Feature Importance)",
)
fig_imp.update_traces(marker_color="#4E79A7", textposition="outside")
fig_imp.update_layout(
    template="plotly_white",
    height=50 * len(feat_imp) + 120,
    showlegend=False,
    xaxis=dict(
        title="중요도",
        tickformat=".0%",
        range=[0, feat_imp["중요도"].max() * 1.3],
    ),
)
st.plotly_chart(fig_imp, use_container_width=True)
st.success(
    "💡 **인사이트**: '런타임 결측 여부(duration_missing)'가 단순 원본 컬럼을 넘어 "
    "실질적인 예측 기여도를 가집니다. TV 시리즈는 런타임이 결측되는 경향이 있어 "
    "유형(영화/TV)을 간접적으로 나타내는 파생 특성임을 확인할 수 있습니다."
)

st.markdown("---")

# 예측기
st.subheader("🎯 나의 콘텐츠 평점 예측해보기")

col_a, col_b = st.columns(2)
with col_a:
    platform_in = st.selectbox("플랫폼", sorted(known_cats["platform"]))
    genre_in    = st.selectbox("장르",   sorted(known_cats["primary_genre"]))
    type_in     = st.selectbox("유형",   sorted(known_cats["type"]))
with col_b:
    year_in     = st.slider("출시 연도", YEAR_MIN, YEAR_MAX, min(2020, YEAR_MAX))
    duration_in = st.slider("런타임 (분)", 10, 300, 100)

if st.button("⭐ 평점 예측하기", type="primary"):
    errors = []
    if platform_in not in known_cats["platform"]:
        errors.append(f"알 수 없는 플랫폼: '{platform_in}'")
    if genre_in not in known_cats["primary_genre"]:
        errors.append(f"알 수 없는 장르: '{genre_in}'")
    if type_in not in known_cats["type"]:
        errors.append(f"알 수 없는 유형: '{type_in}'")

    if errors:
        st.error(" | ".join(errors))
    else:
        try:
            row = {col: 0 for col in feature_cols}
            row["release_year"]     = year_in
            row["duration_minutes"] = duration_in
            row["duration_missing"] = 0
            for prefix, val in [
                ("platform", platform_in),
                ("primary_genre", genre_in),
                ("type", type_in),
            ]:
                ohe_col = f"{prefix}_{val}"
                if ohe_col in row:
                    row[ohe_col] = 1

            X_in = pd.DataFrame([row])[feature_cols]
            pred = round(float(model.predict(X_in)[0]), 2)

            if pred >= 7.5:
                grade, color = "우수 (High)", "#2ecc71"
            elif pred >= 6.5:
                grade, color = "보통 (Average)", "#f39c12"
            else:
                grade, color = "저조 (Low)", "#e74c3c"

            st.markdown(
                f"""
                <div style="text-align:center; padding:28px; border-radius:14px;
                            background:{color}22; border:2px solid {color};">
                    <h2 style="color:{color}; margin:0;">⭐ 예상 IMDB 평점: {pred} / 10</h2>
                    <p style="color:{color}; font-size:18px; margin:10px 0 0 0;">{grade}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"예측 오류: {type(e).__name__} — {e}")
