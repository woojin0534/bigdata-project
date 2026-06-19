import os
import warnings

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLATFORM_COLORS = {
    "Netflix": "#E50914",
    "Amazon Prime Video": "#00A8E0",
    "Disney+": "#113CCF",
}
PLATFORMS = ["Netflix", "Amazon Prime Video", "Disney+"]

COUNTRY_ISO = {
    "United States": "USA", "South Korea": "KOR", "Japan": "JPN",
    "United Kingdom": "GBR", "India": "IND", "France": "FRA",
    "Germany": "DEU", "Italy": "ITA", "Spain": "ESP",
    "Canada": "CAN", "Australia": "AUS", "Brazil": "BRA",
    "Mexico": "MEX", "China": "CHN", "Nigeria": "NGA",
    "Argentina": "ARG", "Turkey": "TUR", "Thailand": "THA",
    "Philippines": "PHL", "Indonesia": "IDN", "Sweden": "SWE",
    "Israel": "ISR", "Denmark": "DNK", "Egypt": "EGY",
    "Hong Kong": "HKG", "Taiwan": "TWN", "New Zealand": "NZL",
    "South Africa": "ZAF", "Russia": "RUS", "Poland": "POL",
    "Netherlands": "NLD", "Belgium": "BEL", "Portugal": "PRT",
    "Norway": "NOR", "Finland": "FIN", "Switzerland": "CHE",
    "Austria": "AUT", "Czech Republic": "CZE", "Hungary": "HUN",
    "Romania": "ROU", "Chile": "CHL", "Colombia": "COL",
    "Peru": "PER", "Venezuela": "VEN", "Saudi Arabia": "SAU",
    "Iran": "IRN", "Pakistan": "PAK", "Bangladesh": "BGD",
    "Sri Lanka": "LKA", "Vietnam": "VNM", "Malaysia": "MYS",
    "Singapore": "SGP", "Morocco": "MAR", "Kenya": "KEN",
    "Ethiopia": "ETH", "Ghana": "GHA", "Senegal": "SEN",
    "Iceland": "ISL", "Luxembourg": "LUX", "Ireland": "IRL",
    "Greece": "GRC", "Ukraine": "UKR", "Slovakia": "SVK",
    "Bulgaria": "BGR", "United Arab Emirates": "ARE",
    "Dominican Republic": "DOM", "Jordan": "JOR",
    "Bahamas": "BHS", "Malawi": "MWI", "Puerto Rico": "PRI",
}


@st.cache_data
def load_data():
    # ▼ 데이터 파일 경로 — 여기 한 줄만 바꾸면 됩니다 ▼
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "streaming_catalog.csv"), encoding="utf-8-sig")
    df_year = pd.read_csv(os.path.join(BASE_DIR, "data", "yearly_release_trends.csv"), encoding="utf-8-sig")
    return df, df_year


def render_sidebar(df, df_year):
    year_min = int(df_year["release_year"].min()) if not df_year.empty else 1990
    year_max = int(df_year["release_year"].max()) if not df_year.empty else 2025
    with st.sidebar:
        st.header("🔧 필터")
        year_range = st.slider("분석 기간", year_min, year_max, (year_min, year_max))
        selected_platforms = st.multiselect(
            "플랫폼 선택", options=PLATFORMS, default=PLATFORMS
        )
        if not selected_platforms:
            selected_platforms = PLATFORMS
        st.caption("선택한 필터는 모든 차트에 실시간 반영됩니다.")
        st.markdown("---")
        df_p = df[
            (df.release_year >= year_range[0]) & (df.release_year <= year_range[1])
        ]
        _csv = (
            df_p[df_p.platform.isin(selected_platforms)]
            .to_csv(index=False)
            .encode("utf-8-sig")
        )
        st.download_button(
            "📥 필터 데이터 다운로드",
            data=_csv,
            file_name="ott_filtered.csv",
            mime="text/csv",
        )
    return year_range, selected_platforms


def hex_to_rgba(hex_color: str, alpha: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
