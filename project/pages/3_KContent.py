import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data_loader import load_data, render_sidebar


df, df_year = load_data()
year_range, selected_platforms = render_sidebar(df, df_year)
year_min_sel, year_max_sel = year_range

df_p = df[(df.release_year >= year_min_sel) & (df.release_year <= year_max_sel)]
df_filtered = df_p[df_p.platform.isin(selected_platforms)]

st.title("🇰🇷 메인 하이라이트: K-콘텐츠의 글로벌 위상 분석")
st.caption(
    "넷플릭스·디즈니+·아마존 프라임에서 한국 콘텐츠의 양적 성장과 질적 경쟁력을 분석합니다."
)
st.divider()

kr = df_filtered[df_filtered.country == "South Korea"]
kr_avg = kr["imdb_rating"].dropna().mean()
global_avg = df_filtered["imdb_rating"].dropna().mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("한국 콘텐츠 총 편수", f"{len(kr):,}편")
c2.metric(
    "글로벌 점유율",
    f"{len(kr) / len(df_filtered) * 100:.1f}%"
    if len(df_filtered) > 0
    else "0.0%",
)
c3.metric("진출 플랫폼 수", f"{kr.platform.nunique()}개")
c4.metric(
    "평균 IMDB 평점",
    f"{kr_avg:.2f}점" if pd.notna(kr_avg) else "데이터 없음",
    f"{kr_avg - global_avg:+.2f} vs 글로벌"
    if pd.notna(kr_avg) and pd.notna(global_avg)
    else None,
)

st.markdown("---")

# 차트 ⑧ Bar + Line
st.subheader("⑧ 한국 콘텐츠 제작 개수 & 점유율 성장도")
st.caption(
    "막대로는 절대적인 편수 증가를, 선으로는 전체 대비 점유율(%) 우상향 곡선을 함께 읽습니다."
)

kr_yr = kr.groupby("release_year").size().reset_index(name="kr_count")
tot_yr = df_filtered.groupby("release_year").size().reset_index(name="total")
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
st.success(
    "💡 **인사이트**: 2019년 이후 K-콘텐츠 편수와 점유율이 동시에 급증합니다. "
    "'기생충' 아카데미 수상(2020)과 '오징어 게임'(2021) 글로벌 흥행이 "
    "OTT 플랫폼들의 K-콘텐츠 투자를 이끈 변곡점입니다."
)

st.markdown("---")

# 차트 ⑨ K-콘텐츠 vs 글로벌 평점 추이
st.subheader("⑨ K-콘텐츠 vs 글로벌 평균 IMDB 평점 추이")
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
    df_filtered.dropna(subset=["imdb_rating"])
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
st.success(
    "💡 **인사이트**: K-콘텐츠 평균 평점이 글로벌 평균을 꾸준히 상회합니다. "
    "단순히 편수가 많은 것이 아니라 '품질 있는 콘텐츠'로 글로벌 시장에서 인정받고 있음을 보여줍니다."
)

st.markdown("---")


# TOP 10
st.subheader("🏆 K-콘텐츠 TOP 10 (IMDB 평점 기준)")
st.caption("선택한 기간 내 한국 콘텐츠 중 IMDB 평점이 가장 높은 작품 10편입니다.")

top_kr = (
    kr.dropna(subset=["imdb_rating"])
    .nlargest(10, "imdb_rating")[
        ["title", "platform", "primary_genre", "type", "release_year", "imdb_rating"]
    ]
    .rename(
        columns={
            "title": "제목",
            "platform": "플랫폼",
            "primary_genre": "장르",
            "type": "유형",
            "release_year": "출시 연도",
            "imdb_rating": "IMDB 평점",
        }
    )
    .reset_index(drop=True)
)
top_kr.index += 1

if top_kr.empty:
    st.info("선택한 기간에 한국 콘텐츠 평점 데이터가 없습니다.")
else:
    st.dataframe(top_kr, use_container_width=True)

