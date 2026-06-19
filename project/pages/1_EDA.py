import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import PLATFORM_COLORS, load_data, render_sidebar


df, df_year = load_data()
year_range, selected_platforms = render_sidebar(df, df_year)
df_f = df[
    (df.release_year >= year_range[0])
    & (df.release_year <= year_range[1])
    & df.platform.isin(selected_platforms)
]

st.title("📊 EDA & 데이터 이해")
st.caption("데이터 구조·결측·이상치·분포를 탐색하고 파생 특성을 발견합니다.")
st.divider()

# ── 1. 데이터 구조 ──────────────────────────────────────────────────────────────
st.subheader("1. 데이터 구조 확인")

c1, c2, c3, c4 = st.columns(4)
c1.metric("총 행 수", f"{len(df):,}편")
c2.metric("컬럼 수", f"{len(df.columns)}개")
c3.metric("플랫폼 수", f"{df.platform.nunique()}개")
c4.metric("장르 수", f"{df.primary_genre.nunique()}개")

with st.expander("컬럼 목록 및 데이터 타입 보기"):
    dtype_df = pd.DataFrame(
        {
            "컬럼": df.columns,
            "타입": [str(t) for t in df.dtypes.values],
            "비결측 수": [df[c].notna().sum() for c in df.columns],
            "예시 값": [
                str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] > 0 else "N/A"
                for c in df.columns
            ],
        }
    )
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

with st.expander("원본 데이터 미리보기 (상위 10행)"):
    st.dataframe(df.head(10), use_container_width=True)

st.info(
    "💡 **발견**: 총 3,046편, 8개 컬럼으로 구성된 데이터셋입니다. "
    "정수형(`release_year`, `duration_minutes`)과 실수형(`imdb_rating`), "
    "범주형(`platform`, `type`, `primary_genre`, `country`) 컬럼이 혼재합니다."
)

st.markdown("---")

# ── 2. 결측값 분석 ──────────────────────────────────────────────────────────────
st.subheader("2. 결측값 분석")

_miss_s = df.isnull().sum()
missing = pd.DataFrame({"컬럼": _miss_s.index, "결측 수": _miss_s.values})
missing["결측률 (%)"] = (missing["결측 수"] / len(df) * 100).round(1)
missing = missing[missing["결측 수"] > 0].sort_values("결측 수", ascending=False)

col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(missing, use_container_width=True, hide_index=True)
with col2:
    fig_miss = px.bar(
        missing,
        x="컬럼",
        y="결측률 (%)",
        text="결측률 (%)",
        color="결측률 (%)",
        color_continuous_scale="Reds",
    )
    fig_miss.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_miss.update_layout(
        template="plotly_white",
        height=320,
        coloraxis_showscale=False,
        yaxis=dict(
            range=[0, missing["결측률 (%)"].max() * 1.35] if not missing.empty else [0, 50]
        ),
    )
    st.plotly_chart(fig_miss, use_container_width=True)

dur_pct = missing.loc[missing["컬럼"] == "duration_minutes", "결측률 (%)"]
dur_val = float(dur_pct.values[0]) if not dur_pct.empty else 0.0
st.warning(
    f"⚠️ **발견**: `duration_minutes`(런타임) 결측률이 **{dur_val:.1f}%**로 가장 높습니다.  \n"
    "TMDB는 TV 시리즈에 에피소드별 런타임을 제공하지 않는 경우가 많아 발생하는 **구조적 결측**입니다.  \n"
    "→ 이 결측 여부를 `duration_missing` 플래그(0/1)로 도출해 ML 모델의 파생 특성으로 활용했습니다."
)

st.markdown("---")

# ── 3. 이상치 확인 ──────────────────────────────────────────────────────────────
st.subheader("3. 이상치 확인")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**IMDB 평점 분포 (히스토그램)**")
    df_r = df_f.dropna(subset=["imdb_rating"])
    avg_r = df_r["imdb_rating"].mean()
    fig_hist = px.histogram(
        df_r,
        x="imdb_rating",
        nbins=40,
        color_discrete_sequence=["#4E79A7"],
        labels={"imdb_rating": "IMDB 평점"},
    )
    fig_hist.add_vline(
        x=avg_r,
        line_dash="dash",
        line_color="red",
        annotation_text=f"평균 {avg_r:.2f}",
        annotation_position="top right",
    )
    fig_hist.update_layout(
        template="plotly_white", height=320, yaxis_title="콘텐츠 수"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown("**영화 런타임 분포 (플랫폼별 박스 플롯)**")
    df_dur = df_f[(df_f.type == "Movie") & df_f.duration_minutes.notna()]
    fig_box = px.box(
        df_dur,
        x="platform",
        y="duration_minutes",
        color="platform",
        color_discrete_map=PLATFORM_COLORS,
        labels={"duration_minutes": "런타임 (분)", "platform": "플랫폼"},
    )
    fig_box.update_layout(
        template="plotly_white", height=320, showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)

rating_low = int((df["imdb_rating"] < 2).sum())
dur_high = int((df[(df.type == "Movie")]["duration_minutes"] > 250).sum())
st.info(
    f"💡 **발견**: IMDB 평점 2점 미만 이상치 **{rating_low}건** (투표 수가 적은 비주류 콘텐츠), "
    f"영화 런타임 250분 초과 이상치 **{dur_high}건** 존재 — ML 모델 학습 시 정상 범위 데이터만 사용합니다."
)

st.markdown("---")

# ── 4. 분포 탐색 ───────────────────────────────────────────────────────────────
st.subheader("4. 주요 컬럼 분포 탐색")

tab1, tab2, tab3, tab4 = st.tabs(["연도별", "장르별", "플랫폼 × 유형", "국가별"])

with tab1:
    yr_cnt = df_f.groupby(["release_year", "type"]).size().reset_index(name="count")
    fig_yr = px.bar(
        yr_cnt,
        x="release_year",
        y="count",
        color="type",
        barmode="stack",
        labels={"release_year": "출시 연도", "count": "편수", "type": "유형"},
        color_discrete_map={"Movie": "#4E79A7", "TV Show": "#F28E2B"},
    )
    fig_yr.update_layout(
        template="plotly_white", height=380, hovermode="x unified"
    )
    st.plotly_chart(fig_yr, use_container_width=True)
    st.success(
        "💡 **인사이트**: 2015년 이후 TV 시리즈 비중이 급격히 증가했습니다. "
        "넷플릭스 오리지널 시리즈 투자 확대와 맞물려 OTT 플랫폼들이 '드라마 시리즈 전쟁'에 돌입했음을 보여줍니다."
    )

with tab2:
    genre_cnt = df_f["primary_genre"].value_counts().head(15).reset_index()
    genre_cnt.columns = ["장르", "편수"]
    fig_genre = px.bar(
        genre_cnt,
        y="장르",
        x="편수",
        orientation="h",
        text="편수",
        color="편수",
        color_continuous_scale="Blues",
    )
    fig_genre.update_layout(
        template="plotly_white",
        height=480,
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    fig_genre.update_traces(textposition="outside")
    st.plotly_chart(fig_genre, use_container_width=True)
    st.success(
        "💡 **인사이트**: Drama·Action·Comedy 3개 장르가 전체의 절반 이상을 차지합니다. "
        "플랫폼과 무관하게 공통적인 핵심 장르이지만, 비율에는 플랫폼별 차이가 있습니다."
    )

with tab3:
    pt_cnt = df_f.groupby(["platform", "type"]).size().reset_index(name="count")
    fig_pt = px.bar(
        pt_cnt,
        x="platform",
        y="count",
        color="type",
        barmode="group",
        labels={"platform": "플랫폼", "count": "편수", "type": "유형"},
        color_discrete_map={"Movie": "#4E79A7", "TV Show": "#F28E2B"},
    )
    fig_pt.update_layout(template="plotly_white", height=380)
    st.plotly_chart(fig_pt, use_container_width=True)
    st.success(
        "💡 **인사이트**: Netflix는 영화와 TV 시리즈 비율이 균형적인 반면, "
        "Disney+는 영화 비중이 높습니다 (Pixar·Marvel 등 영화 IP 자산 반영)."
    )

with tab4:
    cntry_cnt = (
        df_f.dropna(subset=["country"])
        .groupby("country")
        .size()
        .reset_index(name="편수")
        .sort_values("편수", ascending=False)
        .head(15)
    )
    fig_c = px.bar(
        cntry_cnt,
        y="country",
        x="편수",
        orientation="h",
        text="편수",
        color="편수",
        color_continuous_scale="Greens",
        labels={"country": "국가"},
    )
    fig_c.update_layout(
        template="plotly_white",
        height=480,
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
    )
    fig_c.update_traces(textposition="outside")
    st.plotly_chart(fig_c, use_container_width=True)
    st.success(
        "💡 **인사이트**: 미국 콘텐츠가 압도적 1위입니다. "
        "그러나 한국(South Korea)이 일본·인도와 함께 상위권을 차지하며 "
        "비영어권 콘텐츠의 글로벌 확장을 보여줍니다."
    )

st.markdown("---")

# ── 5. 특성 발견 ───────────────────────────────────────────────────────────────
st.subheader("5. 파생 특성 발견 (Feature Engineering)")

st.markdown(
    """
ML 예측기 및 시각화에 활용한 **파생 특성** 목록입니다.
원본 컬럼을 그대로 쓰지 않고, 데이터를 **가공·결합·도출**해 만든 새로운 특성입니다.

| 파생 특성 | 원본 → 도출 방법 | 활용 목적 |
|---|---|---|
| `duration_missing` | `duration_minutes` 결측 여부 → 0/1 이진 플래그 | ML 모델 입력 특성 — 결측 패턴이 TV/영화 구분과 상관 |
| `share_pct` | K-콘텐츠 수 ÷ 전체 콘텐츠 수 × 100 | 연도별 K-콘텐츠 점유율 추이 시각화 |
| `period` | 출시 연도 → 5년 구간 그룹화 (`2000~2004` 등) | 플랫폼별 장르 전략 변화 시각화 |
"""
)

col_fe1, col_fe2 = st.columns(2)
with col_fe1:
    dur_miss_cnt = df["duration_minutes"].isna().sum()
    feat_df = pd.DataFrame(
        {
            "duration_missing": [0, 1],
            "레이블": ["런타임 있음", "런타임 결측"],
            "편수": [len(df) - dur_miss_cnt, dur_miss_cnt],
        }
    )
    fig_fe = px.pie(
        feat_df,
        names="레이블",
        values="편수",
        color_discrete_sequence=["#2ecc71", "#e74c3c"],
        hole=0.4,
        title="duration_missing 분포",
    )
    fig_fe.update_layout(height=280, template="plotly_white")
    st.plotly_chart(fig_fe, use_container_width=True)

with col_fe2:
    tv_miss    = df[df.type == "TV Show"]["duration_minutes"].isna().sum()
    tv_total   = (df.type == "TV Show").sum()
    movie_miss  = df[df.type == "Movie"]["duration_minutes"].isna().sum()
    movie_total = (df.type == "Movie").sum()
    st.markdown("**결측 패턴과 콘텐츠 유형의 관계**")
    st.dataframe(
        pd.DataFrame(
            {
                "유형": ["TV 시리즈", "영화"],
                "전체": [tv_total, movie_total],
                "duration 결측": [tv_miss, movie_miss],
                "결측률": [
                    f"{tv_miss/tv_total*100:.1f}%",
                    f"{movie_miss/movie_total*100:.1f}%",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        "> TV 시리즈의 결측률이 영화보다 훨씬 높아, "
        "`duration_missing = 1`이 'TV 시리즈일 가능성'을 간접적으로 나타내는 정보를 담고 있습니다."
    )

st.success(
    "✅ **결론**: `duration_missing`은 원본 데이터에 없는 파생 특성으로, "
    "ML 모델의 Feature Importance에서도 상위권에 위치해 IMDB 평점 예측에 실질적 기여를 확인했습니다."
)
