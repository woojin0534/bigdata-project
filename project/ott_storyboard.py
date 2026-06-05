import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

YEAR_MIN = int(df_year["release_year"].min()) if not df_year.empty else 1990
YEAR_MAX = int(df_year["release_year"].max()) if not df_year.empty else 2025

# ── 사이드바 필터 ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 필터")
    year_range = st.slider("분석 기간", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX))
    selected_platforms = st.multiselect(
        "플랫폼 선택",
        options=PLATFORMS,
        default=PLATFORMS,
    )
    if not selected_platforms:
        selected_platforms = PLATFORMS
    st.caption("선택한 필터는 모든 차트에 실시간 반영됩니다.")

year_min_sel, year_max_sel = year_range

df_p = df[(df.release_year >= year_min_sel) & (df.release_year <= year_max_sel)]
dy_p = df_year[(df_year.release_year >= year_min_sel) & (df_year.release_year <= year_max_sel)]

_csv = df_p[df_p.platform.isin(selected_platforms)].to_csv(index=False).encode("utf-8-sig")
st.sidebar.markdown("---")
st.sidebar.download_button(
    "📥 필터 데이터 다운로드",
    data=_csv,
    file_name="ott_filtered.csv",
    mime="text/csv",
)

# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.title("📊 글로벌 OTT 트렌드 시각화 스토리보드")
st.markdown(
    "**Netflix · Disney+ · Amazon Prime Video** 데이터로 읽는 스트리밍 시장의 흐름  \n"
    f"분석 기간: {year_min_sel}–{year_max_sel} | 전체 콘텐츠 {len(df_p):,}편"
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
max_rank = int(g5["rank"].max()) if not g5.empty else 5
fig2.update_layout(
    xaxis_title="연도",
    yaxis=dict(
        title="순위",
        autorange="reversed",
        tickmode="linear",
        tick0=1,
        dtick=1,
        range=[0.5, max_rank + 0.5],
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

df3 = df_p[df_p.platform.isin(selected_platforms)]
if df3.empty:
    st.info("선택한 기간·플랫폼에 해당하는 데이터가 없습니다.")
else:
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
                x=selected_platforms,
                y=[pct.loc[p, genre] if p in pct.index else 0 for p in selected_platforms],
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
    "Disney+의 짧은 애니메이션(20–50분)부터 Netflix·Prime의 장편 영화(90–120분)까지 "
    "런타임 밀도 분포를 비교합니다."
)

df_movie = df_p[
    (df_p.type == "Movie")
    & (df_p.platform.isin(selected_platforms))
    & (df_p.duration_minutes.between(30, 240))
]
if df_movie.empty:
    st.info("선택한 기간·플랫폼에 해당하는 영화 런타임 데이터가 없습니다.")
else:
    fig4 = go.Figure()
    for platform in selected_platforms:
        pdata = df_movie[df_movie.platform == platform]
        c = PLATFORM_COLORS.get(platform, "#888888")
        fig4.add_trace(
            go.Violin(
                y=pdata.duration_minutes,
                name=platform,
                box_visible=True,
                meanline_visible=True,
                fillcolor=hex_to_rgba(c),
                line_color=c,
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

st.markdown("---")

# ── 차트 2-3: 플랫폼별 평균 IMDB 평점 비교 ───────────────────────────────────
st.subheader("⑤ 플랫폼별 평균 IMDB 평점 비교")
st.caption(
    "편수·장르에 이어 '콘텐츠 품질' 관점에서 플랫폼을 비교합니다. "
    "오차 막대(±1σ)로 평점 분포의 넓이도 확인하세요."
)

plat_stat = (
    df_p[df_p.platform.isin(selected_platforms)]
    .dropna(subset=["imdb_rating"])
    .groupby("platform")["imdb_rating"]
    .agg(avg="mean", cnt="count", std="std")
    .reset_index()
    .sort_values("avg")
)

fig_pr = go.Figure()
for _, row in plat_stat.iterrows():
    c = PLATFORM_COLORS.get(row["platform"], "#888888")
    fig_pr.add_trace(
        go.Bar(
            x=[row["avg"]],
            y=[row["platform"]],
            orientation="h",
            name=row["platform"],
            marker_color=c,
            error_x=dict(type="data", array=[row["std"]], visible=True, color=c),
            hovertemplate=(
                f"<b>{row['platform']}</b><br>"
                f"평균: %{{x:.2f}}점  (n={int(row['cnt']):,}편)<extra></extra>"
            ),
        )
    )
fig_pr.update_layout(
    xaxis=dict(title="평균 IMDB 평점", range=[0, 10]),
    yaxis_title="",
    showlegend=False,
    template="plotly_white",
    height=max(200, len(selected_platforms) * 80),
)
st.plotly_chart(fig_pr, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PART 3  K-콘텐츠 위상 분석
# ─────────────────────────────────────────────────────────────────────────────
st.header("PART 3. 메인 하이라이트: K-콘텐츠의 위상 분석")

kr = df_p[df_p.country == "South Korea"]
kr_avg = kr["imdb_rating"].dropna().mean()
global_avg = df_p["imdb_rating"].dropna().mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("한국 콘텐츠 총 편수", f"{len(kr):,}편")
c2.metric("글로벌 점유율", f"{len(kr) / len(df_p) * 100:.1f}%" if len(df_p) > 0 else "0.0%")
c3.metric("진출 플랫폼 수", f"{kr.platform.nunique()}개")
c4.metric(
    "평균 IMDB 평점",
    f"{kr_avg:.2f}점" if pd.notna(kr_avg) else "데이터 없음",
    f"{kr_avg - global_avg:+.2f} vs 글로벌" if pd.notna(kr_avg) and pd.notna(global_avg) else None,
)

st.markdown("---")

# ── 차트 3-1: Bar + Line 혼합 ─────────────────────────────────────────────────
st.subheader("⑥ 한국 콘텐츠 제작 개수 & 점유율 성장도")
st.caption(
    "막대로는 절대적인 편수 증가를, 선으로는 전체 대비 "
    "점유율(%) 우상향 곡선을 함께 읽습니다."
)

kr_yr = kr.groupby("release_year").size().reset_index(name="kr_count")
tot_yr = df_p.groupby("release_year").size().reset_index(name="total")
merged = tot_yr.merge(kr_yr, on="release_year", how="left")
merged["kr_count"] = merged["kr_count"].fillna(0).astype(int)
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

# ── 차트 3-2: K-콘텐츠 vs 글로벌 평균 IMDB 평점 추이 ─────────────────────────
st.subheader("⑦ K-콘텐츠 vs 글로벌 평균 IMDB 평점 추이")
st.caption(
    "양적 성장에 그치지 않고 질적으로도 경쟁력을 갖추었는지, "
    "K-콘텐츠 평균 평점과 글로벌 평균을 연도별로 비교합니다."
)

kr_rating_yr = (
    kr.dropna(subset=["imdb_rating"])
    .groupby("release_year")["imdb_rating"]
    .mean()
    .reset_index(name="kr_avg")
)
global_rating_yr = (
    df_p.dropna(subset=["imdb_rating"])
    .groupby("release_year")["imdb_rating"]
    .mean()
    .reset_index(name="global_avg")
)
rating_merged = global_rating_yr.merge(kr_rating_yr, on="release_year", how="left")

fig_kr_r = go.Figure()
fig_kr_r.add_trace(
    go.Scatter(
        x=rating_merged.release_year,
        y=rating_merged.global_avg,
        name="글로벌 평균",
        mode="lines+markers",
        line=dict(color="#888888", width=2, dash="dash"),
        marker=dict(size=7),
        hovertemplate="%{x}년  글로벌 평균 %{y:.2f}점<extra></extra>",
    )
)
fig_kr_r.add_trace(
    go.Scatter(
        x=rating_merged.release_year,
        y=rating_merged.kr_avg,
        name="K-콘텐츠 평균",
        mode="lines+markers",
        line=dict(color="#CD2E3A", width=3),
        marker=dict(size=9),
        hovertemplate="%{x}년  K-콘텐츠 평균 %{y:.2f}점<extra></extra>",
    )
)
fig_kr_r.update_layout(
    xaxis_title="연도",
    yaxis=dict(title="평균 IMDB 평점", range=[4, 10]),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    height=400,
)
st.plotly_chart(fig_kr_r, use_container_width=True)

st.markdown("---")

# ── 차트 3-3: 트리맵 ──────────────────────────────────────────────────────────
st.subheader("⑧ 한국 콘텐츠 플랫폼 × 장르 분포 (트리맵)")
st.caption(
    "한국 콘텐츠가 어느 플랫폼에 집중되어 있고, "
    "어떤 장르로 해외 시장을 공략하는지 면적 크기로 표현합니다."
)

kr_tree = (
    kr.groupby(["platform", "primary_genre"]).size().reset_index(name="count")
)
if kr_tree.empty:
    st.info("선택한 기간에 한국 콘텐츠 데이터가 없습니다.")
else:
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


st.markdown("---")

# ── TOP 10 K-콘텐츠 테이블 ────────────────────────────────────────────────────
st.subheader("🏆 K-콘텐츠 TOP 10 (IMDB 평점 기준)")
st.caption("선택한 기간 내 한국 콘텐츠 중 IMDB 평점이 가장 높은 작품 10편입니다.")

top_kr = (
    kr.dropna(subset=["imdb_rating"])
    .nlargest(10, "imdb_rating")[
        ["title", "platform", "primary_genre", "type", "release_year", "imdb_rating"]
    ]
    .rename(columns={
        "title": "제목",
        "platform": "플랫폼",
        "primary_genre": "장르",
        "type": "유형",
        "release_year": "출시 연도",
        "imdb_rating": "IMDB 평점",
    })
    .reset_index(drop=True)
)
top_kr.index += 1

if top_kr.empty:
    st.info("선택한 기간에 한국 콘텐츠 평점 데이터가 없습니다.")
else:
    st.dataframe(top_kr, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PART 4  IMDB 평점 예측기 (ML)
# ─────────────────────────────────────────────────────────────────────────────
st.header("PART 4. IMDB 평점 예측기 (머신러닝)")
st.caption(
    "플랫폼 · 장르 · 유형 · 출시 연도 · 런타임을 입력하면 "
    "RandomForest 모델이 예상 IMDB 평점을 예측합니다."
)

ML_BASE = ["platform", "primary_genre", "type", "release_year", "duration_minutes"]
ML_FEATURES = ML_BASE + ["duration_missing"]
ML_TARGET = "imdb_rating"
ML_CAT = ["platform", "primary_genre", "type"]


@st.cache_resource
def train_model(df):
    df_ml = df[ML_BASE + [ML_TARGET]].copy()

    # 결측 플래그 (imputation 전에 기록)
    df_ml["duration_missing"] = df_ml["duration_minutes"].isna().astype(int)

    # 장르별 평균으로 duration_minutes 대체, 그래도 NaN이면 전체 평균 사용
    genre_mean = df_ml.groupby("primary_genre")["duration_minutes"].transform("mean")
    overall_mean = df_ml["duration_minutes"].mean()
    df_ml["duration_minutes"] = df_ml["duration_minutes"].fillna(genre_mean).fillna(overall_mean)

    # imdb_rating 등 나머지 결측치 제거
    df_ml = df_ml.dropna(subset=ML_FEATURES + [ML_TARGET])

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

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, le_dict, mae, rmse, r2, len(X_train)


with st.spinner("모델 학습 중... (최초 1회만 실행됩니다)"):
    model, le_dict, mae, rmse, r2, n_train = train_model(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("학습 데이터", f"{n_train:,}건")
m2.metric("평균 절대 오차 (MAE)", f"{mae:.3f}점")
m3.metric("RMSE", f"{rmse:.3f}점")
m4.metric("결정계수 (R²)", f"{r2:.3f}")

st.markdown("---")

feat_imp = pd.DataFrame({
    "특성": ["플랫폼", "장르", "유형 (영화/TV)", "출시 연도", "런타임", "런타임 결측 여부"],
    "중요도": model.feature_importances_,
}).sort_values("중요도", ascending=True)

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

st.markdown("---")
st.subheader("🎯 나의 콘텐츠 평점 예측해보기")

col_a, col_b = st.columns(2)
with col_a:
    platform_in = st.selectbox("플랫폼", sorted(df["platform"].dropna().unique()))
    genre_in = st.selectbox("장르", sorted(df["primary_genre"].dropna().unique()))
    type_in = st.selectbox("유형", ["Movie", "TV Show"])
with col_b:
    year_in = st.slider("출시 연도", YEAR_MIN, YEAR_MAX, min(2020, YEAR_MAX))
    duration_in = st.slider("런타임 (분)", 10, 300, 100)

if st.button("⭐ 평점 예측하기", type="primary"):
    try:
        input_row = {
            "platform": le_dict["platform"].transform([platform_in])[0],
            "primary_genre": le_dict["primary_genre"].transform([genre_in])[0],
            "type": le_dict["type"].transform([type_in])[0],
            "release_year": year_in,
            "duration_minutes": duration_in,
            "duration_missing": 0,  # 사용자가 직접 입력하므로 결측 없음
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
