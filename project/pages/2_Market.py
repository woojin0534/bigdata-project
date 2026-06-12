import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import warnings

warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import COUNTRY_ISO, PLATFORM_COLORS, hex_to_rgba, load_data, render_sidebar


df, df_year = load_data()
year_range, selected_platforms = render_sidebar(df, df_year)
year_min_sel, year_max_sel = year_range

df_p = df[(df.release_year >= year_min_sel) & (df.release_year <= year_max_sel)]

st.title("📈 시장 전체 흐름 & 플랫폼 3파전 비교")
st.markdown(
    f"분석 기간: **{year_min_sel}–{year_max_sel}** | "
    f"선택 플랫폼: **{', '.join(selected_platforms)}** | "
    f"콘텐츠 수: **{len(df_p[df_p.platform.isin(selected_platforms)]):,}편**"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# PART 1  시장 전체의 흐름 읽기
# ─────────────────────────────────────────────────────────────────────────────
st.header("PART 1. 시장 전체의 흐름 읽기")

# 차트 ① 누적 영역 차트
st.subheader("① 연도별 콘텐츠 업로드 추이 — 영화 vs TV 시리즈")
st.caption(
    "전통적인 영화 중심 시장에서 OTT 오리지널 시리즈가 "
    "언제부터 급부상했는지 누적 면적으로 확인합니다."
)

_yearly_sel = (
    df_p[df_p.platform.isin(selected_platforms)]
    .groupby(["release_year", "type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
_yearly_sel.columns.name = None
for _c in ("Movie", "TV Show"):
    if _c not in _yearly_sel.columns:
        _yearly_sel[_c] = 0

fig1 = go.Figure()
fig1.add_trace(
    go.Scatter(
        x=_yearly_sel.release_year,
        y=_yearly_sel["Movie"],
        name="영화 (Movie)",
        stackgroup="one",
        fillcolor="rgba(78,121,167,0.75)",
        line=dict(color="#4E79A7", width=1.5),
        hovertemplate="%{x}년  영화 %{y}편<extra></extra>",
    )
)
fig1.add_trace(
    go.Scatter(
        x=_yearly_sel.release_year,
        y=_yearly_sel["TV Show"],
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
st.success(
    "💡 **인사이트**: 2015년을 기점으로 TV 시리즈 업로드량이 급증합니다. "
    "넷플릭스의 오리지널 드라마 투자 확대와 맞물려 OTT 플랫폼이 '드라마 제작사'로 변모한 시점입니다."
)

st.markdown("---")

# 차트 ② Bump Chart
st.subheader("② Top 5 장르 연도별 순위 변천사 (Bump Chart)")
st.caption(
    "10년 전 유행했던 장르와 최근 급부상한 장르의 "
    "세대교체를 순위 흐름 곡선으로 표현합니다."
)

_df_plat = df_p[df_p.platform.isin(selected_platforms)]
top5_genres = _df_plat["primary_genre"].value_counts().head(5).index.tolist()

genre_year = (
    _df_plat.groupby(["release_year", "primary_genre"])
    .size()
    .reset_index(name="count")
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
st.success(
    "💡 **인사이트**: Drama 장르는 전 기간에 걸쳐 1~2위를 유지하며 OTT의 핵심 콘텐츠임을 확인할 수 있습니다. "
    "반면 특정 장르는 시기에 따라 순위가 크게 변동하여 트렌드 변화를 보여줍니다."
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# PART 2  플랫폼별 3파전 비교
# ─────────────────────────────────────────────────────────────────────────────
st.header("PART 2. 플랫폼별 3파전 비교")

# 차트 ③ 100% 누적 막대
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
    st.success(
        "💡 **인사이트**: Disney+는 Animation·Family 비중이 다른 플랫폼 대비 현저히 높아 "
        "자사 IP(Pixar, Marvel, Disney) 중심의 패밀리 콘텐츠 전략을 확인할 수 있습니다."
    )

st.markdown("---")

# 차트 ④ 바이올린 플롯
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
    st.success(
        "💡 **인사이트**: Disney+의 영화 런타임 분포가 Netflix·Amazon보다 짧고 집중되어 있습니다. "
        "단편 애니메이션과 패밀리 영화 중심의 포트폴리오가 반영된 결과입니다."
    )

st.markdown("---")

# 차트 ⑤ 평균 IMDB 평점
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
st.success(
    "💡 **인사이트**: 편수가 가장 많은 플랫폼이 반드시 평점이 높지는 않습니다. "
    "오차 막대 크기는 플랫폼별 콘텐츠 품질 편차를 나타내며, 넓을수록 다양한 품질의 콘텐츠를 보유합니다."
)

st.markdown("---")

# 차트 ⑥ 장르 전략 변화
st.subheader("⑥ 플랫폼별 연도별 장르 전략 변화")
st.caption(
    "각 플랫폼이 5년 단위로 어떤 장르에 집중했는지 비율 변화로 확인합니다."
)

df_gt = df_p[df_p.platform.isin(selected_platforms)]
top5_g = df_gt["primary_genre"].value_counts().head(5).index.tolist()
df_gt = df_gt[df_gt.primary_genre.isin(top5_g)]

if df_gt.empty:
    st.info("선택한 기간·플랫폼에 해당하는 데이터가 없습니다.")
else:
    df_gt2 = df_gt.copy()
    df_gt2["period"] = df_gt2["release_year"].apply(
        lambda y: f"{(y // 5) * 5}~{(y // 5) * 5 + 4}"
    )

    tabs_gt = st.tabs(selected_platforms)
    for tab, platform in zip(tabs_gt, selected_platforms):
        with tab:
            pdata = (
                df_gt2[df_gt2.platform == platform]
                .groupby(["period", "primary_genre"])
                .size()
                .reset_index(name="count")
            )
            period_total = pdata.groupby("period")["count"].transform("sum")
            pdata["pct"] = (pdata["count"] / period_total * 100).round(1)
            period_order = sorted(pdata["period"].unique())

            fig_gt = px.bar(
                pdata,
                x="period",
                y="pct",
                color="primary_genre",
                barmode="stack",
                category_orders={"period": period_order},
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={"pct": "비율 (%)", "period": "기간", "primary_genre": "장르"},
                text=pdata["pct"].apply(lambda v: f"{v:.0f}%" if v >= 8 else ""),
            )
            fig_gt.update_layout(
                template="plotly_white",
                height=420,
                yaxis=dict(title="비율 (%)", range=[0, 100]),
                xaxis_title="기간",
                legend_title="장르",
                hovermode="x unified",
            )
            fig_gt.update_traces(textposition="inside", insidetextanchor="middle")
            st.plotly_chart(fig_gt, use_container_width=True)

    st.success(
        "💡 **인사이트**: 플랫폼마다 시기별 장르 집중도가 다릅니다. "
        "Disney+는 최근으로 올수록 Action 비중이 증가해 Marvel·Star Wars 확장을 반영합니다."
    )

st.markdown("---")

# 차트 ⑦ 세계지도
st.subheader("⑦ 국가별 콘텐츠 원산지 세계지도")
st.caption(
    "어느 나라 콘텐츠가 글로벌 OTT를 지배하는지 "
    "국가별 제작 편수를 색상 농도로 표현합니다."
)

country_cnt = (
    df_p[df_p.platform.isin(selected_platforms)]
    .dropna(subset=["country"])
    .groupby("country")
    .size()
    .reset_index(name="편수")
)
country_cnt["iso_a3"] = country_cnt["country"].map(COUNTRY_ISO)
country_cnt = country_cnt.dropna(subset=["iso_a3"])

if country_cnt.empty:
    st.info("선택한 기간에 국가 데이터가 없습니다.")
else:
    fig_map = px.choropleth(
        country_cnt,
        locations="iso_a3",
        color="편수",
        hover_name="country",
        hover_data={"편수": True, "iso_a3": False},
        color_continuous_scale="YlOrRd",
        labels={"편수": "콘텐츠 수"},
    )
    fig_map.update_layout(
        template="plotly_white",
        height=480,
        coloraxis_colorbar=dict(title="콘텐츠 수"),
        margin=dict(l=0, r=0, t=10, b=0),
        geo=dict(showframe=False, showcoastlines=True),
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.success(
        "💡 **인사이트**: 미국이 압도적이지만, 한국·인도·영국 등 비미국 콘텐츠 비중이 "
        "글로벌 OTT 시장의 다양성을 보여줍니다. 특히 아시아 국가들의 존재감이 두드러집니다."
    )

