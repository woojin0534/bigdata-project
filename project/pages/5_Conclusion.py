import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st

from utils import load_data


df, _ = load_data()

st.title("📝 결론 & 한계 & 향후 과제")
st.caption(
    "문제 정의부터 분석 결과, 데이터 한계까지 프로젝트 전체를 정리합니다."
)
st.divider()

# ── 1. 문제 정의 ───────────────────────────────────────────────────────────────
st.subheader("1. 문제 정의")
st.markdown(
    """
**글로벌 OTT 시장은 2015년 이후 폭발적으로 성장했습니다.**
Netflix·Disney+·Amazon Prime Video 3사가 주도하는 이 시장에서 다음 질문에 답하고자 했습니다.

> **"3대 OTT 플랫폼의 콘텐츠 전략은 어떻게 다르며, K-콘텐츠는 어떤 위상을 차지하고 있는가?"**

- 플랫폼별로 어떤 장르에 집중하는가?
- OTT 시장에서 영화와 TV 시리즈의 비중은 어떻게 변해왔는가?
- 한국 콘텐츠의 편수와 품질(평점)은 글로벌 대비 어떤 수준인가?
- 콘텐츠 메타데이터(플랫폼, 장르, 유형, 연도, 런타임)로 IMDB 평점을 예측할 수 있는가?

**데이터**: TMDB API로 수집한 3개 플랫폼 총 **{:,}편** (Netflix {:,}편 · Amazon Prime Video {:,}편 · Disney+ {:,}편)
""".format(
        len(df),
        len(df[df.platform == "Netflix"]),
        len(df[df.platform == "Amazon Prime Video"]),
        len(df[df.platform == "Disney+"]),
    )
)

st.markdown("---")

# ── 2. 핵심 발견 ───────────────────────────────────────────────────────────────
st.subheader("2. 핵심 발견 (Key Findings)")

findings = [
    (
        "📈 TV 시리즈의 부상",
        "2015년을 기점으로 OTT 플랫폼들의 TV 시리즈 업로드가 급증했습니다. "
        "넷플릭스 오리지널 드라마 투자 확대가 'OTT = 드라마 제작사' 시대를 열었습니다.",
    ),
    (
        "🎭 플랫폼별 장르 정체성",
        "Disney+는 Animation·Family 비중이 타 플랫폼 대비 현저히 높습니다 (Pixar·Marvel·Disney IP). "
        "Netflix와 Amazon은 Drama·Action 중심의 균형 잡힌 포트폴리오를 보입니다.",
    ),
    (
        "🇰🇷 K-콘텐츠 급성장",
        "2019년 이후 K-콘텐츠 편수와 글로벌 점유율이 동시에 급증했습니다. "
        "'기생충'(2020)과 '오징어 게임'(2021)이 OTT 플랫폼들의 K-콘텐츠 투자를 가속화한 변곡점입니다.",
    ),
    (
        "⭐ K-콘텐츠 품질 우위",
        "K-콘텐츠 평균 IMDB 평점이 글로벌 평균을 꾸준히 상회합니다. "
        "양적 성장과 동시에 질적 경쟁력을 확보했음을 객관적 지표로 확인했습니다.",
    ),
    (
        "🤖 ML 예측 결과",
        "랜덤 포레스트·XGBoost 모델이 선형 회귀보다 높은 예측 정확도를 보였습니다. "
        "IMDB 평점은 비선형 관계를 가지며, 파생 특성 `duration_missing`이 예측에 기여합니다.",
    ),
]

for title, body in findings:
    with st.expander(title, expanded=True):
        st.markdown(body)

st.markdown("---")

# ── 3. 데이터 한계 ─────────────────────────────────────────────────────────────
st.subheader("3. 데이터 한계 (Limitations)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 데이터 수집 한계")
    st.markdown(
        """
- **플랫폼 범위**: Netflix·Disney+·Amazon Prime 3개만 분석 — Hulu·Apple TV+·HBO Max 미포함
- **샘플 편향**: TMDB API는 투표 수(vote_count) 기준으로 콘텐츠를 제공하므로, 비주류 콘텐츠는 누락될 수 있음
- **런타임 결측**: TV 시리즈의 `duration_minutes` 결측률이 높음 — TMDB API 구조적 한계
- **지역 라이선스**: `country` 컬럼은 제작 국가이며, 실제 플랫폼 제공 지역과 다를 수 있음
"""
    )

with col2:
    st.markdown("#### 모델 한계")
    st.markdown(
        """
- **낮은 R²**: IMDB 평점은 감독·배우·스토리 등 메타데이터에 없는 요소에 크게 의존
- **피처 부족**: 실제 시청 데이터(조회 수, 완주율), 예산, 제작진 정보 미포함
- **시계열 미고려**: 출시 연도는 포함하나, 플랫폼 추가 날짜·시청 트렌드 변화 미반영
- **범주형 인코딩**: LabelEncoding 사용 — 장르·플랫폼 간 순서 관계가 없음에도 순서 부여됨
"""
    )

st.markdown("---")

# ── 4. 향후 과제 ───────────────────────────────────────────────────────────────
st.subheader("4. 향후 과제 (Future Work)")

st.markdown(
    """
| 과제 | 내용 |
|---|---|
| 플랫폼 확장 | Hulu·Apple TV+·HBO Max 등 추가 플랫폼 포함으로 더 넓은 시장 분석 |
| 실시간 데이터 | 플랫폼 추가/삭제 날짜 기반의 '현재 제공 콘텐츠' 실시간 업데이트 |
| 시청 데이터 통합 | Netflix의 공개 시청 데이터와 결합해 '제공 편수'와 '실제 시청' 관계 분석 |
"""
)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:13px;'>"
    "데이터 출처: TMDB API (The Movie Database) — themoviedb.org  |  "
    "수집 기간: 2026년 6월  |  라이선스: CC BY-NC 4.0"
    "</div>",
    unsafe_allow_html=True,
)

