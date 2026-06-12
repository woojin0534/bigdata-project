import streamlit as st

st.set_page_config(
    page_title="글로벌 OTT 트렌드 분석",
    page_icon="📺",
    layout="wide",
)

pg = st.navigation(
    [
        st.Page("pages/0_Home.py",        title="홈",            icon="🏠", default=True),
        st.Page("pages/1_EDA.py",         title="EDA & 데이터 이해", icon="📊"),
        st.Page("pages/2_Market.py",      title="시장 분석",       icon="📈"),
        st.Page("pages/3_KContent.py",    title="K-콘텐츠 분석",   icon="🇰🇷"),
        st.Page("pages/4_ML.py",          title="ML 예측기",       icon="🤖"),
        st.Page("pages/5_Conclusion.py",  title="결론 & 한계",     icon="📝"),
    ]
)
pg.run()
