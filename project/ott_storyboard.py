import os
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="글로벌 OTT 트렌드 분석",
    page_icon="📺",
    layout="wide",
)

# ── 상수 ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PLATFORM_COLORS = {
    "Netflix": "#E50914",
    "Amazon Prime Video": "#00A8E0",
    "Disney+": "#113CCF",
}
PLATFORMS = ["Netflix", "Amazon Prime Video", "Disney+"]

YEAR_MIN, YEAR_MAX = 1995, 2024


def hex_to_rgba(hex_color: str, alpha: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "streaming_catalog.csv"))
    df_year = pd.read_csv(os.path.join(BASE_DIR, "yearly_release_trends.csv"))
    return df, df_year


df, df_year = load_data()

df_p = df[(df.release_year >= YEAR_MIN) & (df.release_year <= YEAR_MAX)]
dy_p = df_year[(df_year.release_year >= YEAR_MIN) & (df_year.release_year <= YEAR_MAX)]

# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.title("📊 글로벌 OTT 트렌드 시각화 스토리보드")
st.markdown(
    "**Netflix · Disney+ · Amazon Prime Video** 데이터로 읽는 스트리밍 시장의 흐름  \n"
    f"분석 기간: {YEAR_MIN}–{YEAR_MAX} | 전체 콘텐츠 {len(df):,}편"
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PART 1  시장 전체의 흐름 읽기
# ─────────────────────────────────────────────────────────────────────────────
st.header("PART 1. 시장 전체의 흐름 읽기 (The Big Picture)")

# ── 차트 1-1: 누적 영역 차트 ──────────────────────────────────────────────────
st.subheader("① 연도별 콘텐츠 업로드 추이 — 영화 vs TV 시리즈")
st.caption(
    "전통적인 영화 중심 시장에서 OTT 오리지널 시리즈가 "
    "언제부터 급부상했는지 누적 면적으로 확인합니다."
)

fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=dy_p.release_year,
        y=dy_p.movies,
        name="영화 (Movie)",
        stackgroup="one",
        fillcolor="rgba(78,121,167,0.75)",
        line=dict(color="#4E79A7", width=1.5),
        hovertemplate="%{x}년  영화 %{y}편<extra></extra>",
    )
)
fig1.add_trace(
    go.Scatter(
        x=dy_p.release_year,
        y=dy_p.tv_shows,
        name="TV 시리즈 (TV Show)",
        stackgroup="one",
        fillcolor="rgba(242,142,43,0.75)",
        line=dict(color="#F28E2B", width=1.5),
        hovertemplate="%{x}년  TV 시리즈 %{y}편<extra></extra>",
    )
)
fig1.update_layout(
    xaxis_title="연도",
    yaxis_title="콘텐츠 수 (편)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    height=450,
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ── 차트 1-2: Bump Chart ──────────────────────────────────────────────────────
st.subheader("② Top 5 장르 연도별 순위 변천사 (Bump Chart)")
st.caption(
    "10년 전 유행했던 장르와 최근 급부상한 장르의 "
    "세대교체를 순위 흐름 곡선으로 표현합니다."
)

top5_genres = df_p["primary_genre"].value_counts().head(5).index.tolist()

genre_year = (
    df_p.groupby(["release_year", "primary_genre"]).size().reset_index(name="count")
)
genre_year["rank"] = (
    genre_year.groupby("release_year")["count"]
    .rank(ascending=False, method="first")
    .astype(int)
)
g5 = genre_year[genre_year.primary_genre.isin(top5_genres)].sort_values("release_year")

bump_colors = px.colors.qualitative.Bold

fig2 = go.Figure()
for i, genre in enumerate(top5_genres):
    gd = g5[g5.primary_genre == genre]
    fig2.add_trace(
        go.Scatter(
            x=gd.release_year,
            y=gd["rank"],
            name=genre,
            mode="lines+markers",
            line=dict(color=bump_colors[i % len(bump_colors)], width=3),
            marker=dict(size=9, color=bump_colors[i % len(bump_colors)]),
            customdata=gd["count"],
            hovertemplate=(
                f"<b>{genre}</b>  %{{x}}년: %{{y}}위  (%{{customdata}}편)<extra></extra>"
            ),
        )
    )
fig2.update_layout(
    xaxis_title="연도",
    yaxis=dict(
        title="순위",
        autorange="reversed",
        tickmode="linear",
        tick0=1,
        dtick=1,
        range=[0.5, 5.5],
    ),
    legend_title="장르",
    hovermode="x",
    template="plotly_white",
    height=450,
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PART 2  플랫폼별 3파전 비교
# ─────────────────────────────────────────────────────────────────────────────
st.header("PART 2. 플랫폼별 3파전 비교 (Netflix vs Disney+ vs Prime Video)")

# ── 차트 2-1: 100% 누적 막대 ──────────────────────────────────────────────────
st.subheader("③ 플랫폼별 장르 포트폴리오 (100% 누적 막대)")
st.caption(
    "Netflix는 스릴러·드라마, Disney+는 애니메이션·가족 등 "
    "각 플랫폼의 콘텐츠 정체성을 한눈에 비교합니다."
)

df3 = df_p[df_p.platform.isin(PLATFORMS)]
pivot = df3.groupby(["platform", "primary_genre"]).size().unstack(fill_value=0)

top8 = df3["primary_genre"].value_counts().head(8).index.tolist()
avail = [g for g in top8 if g in pivot.columns]
pivot_t = pivot[avail].copy()
other_cols = [c for c in pivot.columns if c not in avail]
pivot_t["기타"] = pivot[other_cols].sum(axis=1) if other_cols else 0
pct = pivot_t.div(pivot_t.sum(axis=1), axis=0) * 100

genre_palette = (
    px.colors.qualitative.Pastel
    + px.colors.qualitative.Safe
    + px.colors.qualitative.Antique
)

fig3 = go.Figure()
for i, genre in enumerate(pivot_t.columns):
    fig3.add_trace(
        go.Bar(
            name=genre,
            x=PLATFORMS,
            y=[pct.loc[p, genre] if p in pct.index else 0 for p in PLATFORMS],
            marker_color=genre_palette[i % len(genre_palette)],
            hovertemplate=f"<b>{genre}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        )
    )
fig3.update_layout(
    barmode="stack",
    xaxis_title="플랫폼",
    yaxis=dict(title="비율 (%)", range=[0, 100]),
    legend_title="장르",
    template="plotly_white",
    height=500,
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── 차트 2-2: 바이올린 플롯 ───────────────────────────────────────────────────
st.subheader("④ 플랫폼별 영화 런타임 분포 (바이올린 플롯)")
st.caption(
    "Disney+의 짧은 애니메이션(20~50분)부터 Netflix·Prime의 장편 영화(90~120분)까지 "
    "런타임 밀도 분포를 비교합니다."
)

df_movie = df[
    (df.type == "Movie")
    & (df.platform.isin(PLATFORMS))
    & (df.duration_minutes.between(30, 240))
]

fig4 = go.Figure()
for platform in PLATFORMS:
    pdata = df_movie[df_movie.platform == platform]
    fig4.add_trace(
        go.Violin(
            y=pdata.duration_minutes,
            name=platform,
            box_visible=True,
            meanline_visible=True,
            fillcolor=hex_to_rgba(PLATFORM_COLORS[platform]),
            line_color=PLATFORM_COLORS[platform],
            hoverinfo="y+name",
        )
    )
fig4.update_layout(
    yaxis_title="런타임 (분)",
    template="plotly_white",
    height=500,
    showlegend=True,
)
st.plotly_chart(fig4, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PART 3  K-콘텐츠 위상 분석
# ─────────────────────────────────────────────────────────────────────────────
st.header("PART 3. 메인 하이라이트: K-콘텐츠의 위상 분석")

kr = df[df.country == "South Korea"]
c1, c2, c3 = st.columns(3)
c1.metric("한국 콘텐츠 총 편수", f"{len(kr):,}편")
c2.metric("글로벌 점유율", f"{len(kr) / len(df) * 100:.1f}%")
c3.metric("진출 플랫폼 수", f"{kr.platform.nunique()}개")

st.markdown("---")

# ── 차트 3-1: Bar + Line 혼합 ─────────────────────────────────────────────────
st.subheader("⑤ 한국 콘텐츠 제작 개수 & 점유율 성장도")
st.caption(
    "막대로는 절대적인 편수 증가를, 선으로는 전체 대비 "
    "점유율(%) 우상향 곡선을 함께 읽습니다."
)

kr_yr = kr.groupby("release_year").size().reset_index(name="kr_count")
tot_yr = df.groupby("release_year").size().reset_index(name="total")
merged = kr_yr.merge(tot_yr, on="release_year").query(
    f"{YEAR_MIN} <= release_year <= {YEAR_MAX}"
)
merged["share_pct"] = merged.kr_count / merged.total * 100

fig5 = make_subplots(specs=[[{"secondary_y": True}]])
fig5.add_trace(
    go.Bar(
        x=merged.release_year,
        y=merged.kr_count,
        name="한국 콘텐츠 수",
        marker_color="#CD2E3A",
        opacity=0.8,
        hovertemplate="%{x}년  한국 콘텐츠 %{y}편<extra></extra>",
    ),
    secondary_y=False,
)
fig5.add_trace(
    go.Scatter(
        x=merged.release_year,
        y=merged.share_pct,
        name="점유율 (%)",
        line=dict(color="#003478", width=3),
        marker=dict(size=8),
        mode="lines+markers",
        hovertemplate="%{x}년  점유율 %{y:.2f}%<extra></extra>",
    ),
    secondary_y=True,
)
fig5.update_layout(
    template="plotly_white",
    height=500,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig5.update_yaxes(title_text="콘텐츠 수 (편)", secondary_y=False)
fig5.update_yaxes(title_text="점유율 (%)", secondary_y=True)
fig5.update_xaxes(title_text="연도")
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ── 차트 3-2: 트리맵 ──────────────────────────────────────────────────────────
st.subheader("⑥ 한국 콘텐츠 플랫폼 × 장르 분포 (트리맵)")
st.caption(
    "한국 콘텐츠가 어느 플랫폼에 집중되어 있고, "
    "어떤 장르로 해외 시장을 공략하는지 면적 크기로 표현합니다."
)

kr_tree = (
    kr.groupby(["platform", "primary_genre"]).size().reset_index(name="count")
)
fig6 = px.treemap(
    kr_tree,
    path=["platform", "primary_genre"],
    values="count",
    color="count",
    color_continuous_scale="RdYlGn",
)
fig6.update_traces(
    hovertemplate="<b>%{label}</b><br>콘텐츠 수: %{value}편<extra></extra>"
)
fig6.update_layout(height=600)
st.plotly_chart(fig6, use_container_width=True)


st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PART 4  IMDB 평점 예측기 (ML)
# ─────────────────────────────────────────────────────────────────────────────
st.header("PART 4. IMDB 평점 예측기 (머신러닝)")
st.caption(
    "플랫폼 · 장르 · 유형 · 출시 연도 · 런타임을 입력하면 "
    "RandomForest 모델이 예상 IMDB 평점을 예측합니다."
)

ML_FEATURES = ["platform", "primary_genre", "type", "release_year", "duration_minutes"]
ML_TARGET = "imdb_rating"
ML_CAT = ["platform", "primary_genre", "type"]


@st.cache_resource
def train_model(df):
    df_ml = df[ML_FEATURES + [ML_TARGET]].dropna()
    df_enc = df_ml.copy()
    le_dict = {}
    for col in ML_CAT:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_dict[col] = le
    X = df_enc[ML_FEATURES]
    y = df_enc[ML_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))
    return model, le_dict, mae, r2


with st.spinner("모델 학습 중... (최초 1회만 실행됩니다)"):
    model, le_dict, mae, r2 = train_model(df)

m1, m2, m3 = st.columns(3)
m1.metric("학습 데이터", f"{int(len(df[ML_FEATURES + [ML_TARGET]].dropna()) * 0.8):,}건")
m2.metric("평균 절대 오차 (MAE)", f"{mae:.3f}점")
m3.metric("결정계수 (R²)", f"{r2:.3f}")

st.markdown("---")

feat_imp = pd.DataFrame({
    "특성": ["플랫폼", "장르", "유형 (영화/TV)", "출시 연도", "런타임"],
    "중요도": model.feature_importances_,
}).sort_values("중요도", ascending=True)

fig_imp = px.bar(
    feat_imp,
    x="중요도",
    y="특성",
    orientation="h",
    color="중요도",
    color_continuous_scale="Blues",
    title="📊 예측에 영향을 미치는 특성 중요도 (Feature Importance)",
)
fig_imp.update_layout(template="plotly_white", height=300, showlegend=False)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")
st.subheader("🎯 나의 콘텐츠 평점 예측해보기")

col_a, col_b = st.columns(2)
with col_a:
    platform_in = st.selectbox("플랫폼", sorted(df["platform"].dropna().unique()))
    genre_in = st.selectbox("장르", sorted(df["primary_genre"].dropna().unique()))
    type_in = st.selectbox("유형", ["Movie", "TV Show"])
with col_b:
    year_in = st.slider("출시 연도", 1995, 2024, 2020)
    duration_in = st.slider("런타임 (분)", 10, 300, 100)

if st.button("⭐ 평점 예측하기", type="primary"):
    try:
        input_row = {
            "platform": le_dict["platform"].transform([platform_in])[0],
            "primary_genre": le_dict["primary_genre"].transform([genre_in])[0],
            "type": le_dict["type"].transform([type_in])[0],
            "release_year": year_in,
            "duration_minutes": duration_in,
        }
        X_in = pd.DataFrame([input_row])[ML_FEATURES]
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
        st.error(f"예측 오류: {e}")
