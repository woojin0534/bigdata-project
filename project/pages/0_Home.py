import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import plotly.express as px
import streamlit as st

from utils import PLATFORM_COLORS, load_data


df, _ = load_data()

st.title("📺 글로벌 OTT 트렌드 시각화 스토리보드")
st.markdown(
    "**Netflix · Disney+ · Amazon Prime Video** 콘텐츠 데이터(TMDB API 기반)로 읽는 스트리밍 시장의 흐름"
)
st.divider()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("총 콘텐츠", f"{len(df):,}편")
c2.metric("분석 플랫폼", f"{df.platform.nunique()}개")
c3.metric("분석 기간", f"{int(df.release_year.min())}–{int(df.release_year.max())}")
c4.metric("수록 국가", f"{df.country.nunique()}개국")
c5.metric("평균 IMDB 평점", f"{df.imdb_rating.dropna().mean():.2f}점")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("플랫폼별 콘텐츠 분포")
    plat_cnt = df["platform"].value_counts().reset_index()
    plat_cnt.columns = ["platform", "count"]
    fig_pie = px.pie(
        plat_cnt,
        names="platform",
        values="count",
        color="platform",
        color_discrete_map=PLATFORM_COLORS,
        hole=0.4,
    )
    fig_pie.update_layout(height=340, template="plotly_white")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("영화 vs TV 시리즈 분포")
    type_cnt = df["type"].value_counts().reset_index()
    type_cnt.columns = ["type", "count"]
    fig_type = px.bar(
        type_cnt,
        x="type",
        y="count",
        color="type",
        text="count",
        color_discrete_sequence=["#4E79A7", "#F28E2B"],
        labels={"type": "유형", "count": "편수"},
    )
    fig_type.update_layout(height=340, template="plotly_white", showlegend=False)
    fig_type.update_traces(textposition="outside")
    st.plotly_chart(fig_type, use_container_width=True)

st.divider()

st.subheader("📑 페이지별 탐색 안내")
a, b, c, d, e = st.columns(5)
with a:
    st.info("**📊 1 EDA**\n\n데이터 구조·결측·이상치·분포 탐색 및 파생 특성 발견")
with b:
    st.success("**📈 2 Market**\n\n연도별 트렌드 & 플랫폼 3파전 비교 (차트 ①~⑦)")
with c:
    st.warning("**🇰🇷 3 KContent**\n\n한국 콘텐츠의 글로벌 위상 분석 (차트 ⑧~⑩)")
with d:
    st.error("**🤖 4 ML**\n\nIMDB 평점 예측 모델 & 3모델 성능 비교")
with e:
    st.markdown(
        "<div style='background:#f0f2f6;padding:12px;border-radius:8px;min-height:110px;'>"
        "<b>📝 5 Conclusion</b><br><br>핵심 발견 · 데이터 한계 · 향후 과제</div>",
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "데이터 출처: TMDB API (The Movie Database) — themoviedb.org  |  "
    "수집 기간: 2026년 6월  |  라이선스: CC BY-NC 4.0"
)

