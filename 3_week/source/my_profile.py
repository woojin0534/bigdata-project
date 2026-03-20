import streamlit as st 
import pandas as pd 
st.title('자기소개') 
# 본인 정보 수정하기! 
st.write('## 기본 정보') 
st.write('**이름**: 장우진') 
st.write('**학과**: 인공지능소프트웨어학과') 
st.write('**학년**: 3학년') 
st.write('---')  # 구분선 
st.write('## 이번 학기 시간표') 
schedule = pd.DataFrame({ 
'요일': ['월', '화', '수', '목', '금'], 
'6교시': ['인공지능라이브러리', '인공지능캡스톤디자인', '-', '인공지능서비스프로젝트', '빅데이터분석프로젝트'], 
'7교시': ['인공지능라이브러리', '인공지능캡스톤디자인', '-', '인공지능서비스프로젝트', '빅데이터분석프로젝트'], 
'8교시': ['인공지능라이브러리', '인공지능캡스톤디자인', '-', '인공지능서비스프로젝트', '빅데이터분석프로젝트'],
'9교시': ['-','인공지능캡스톤디자인','-','-','-']
}) 
st.write(schedule) 
st.write('---') 
st.write('## 관심 분야') 
st.write('- 머신러닝') 
st.write('- 데이터 시각화')  

st.write('---')
st.write('## 이번 학기 목표') 
goals = pd.DataFrame({ 
'목표': ['Streamlit 마스터', 'HuggingFace 활용', '포트폴리오 완성'], 
'달성률': [10, 0, 5] 
}) 
st.write(goals) 
st.bar_chart(goals.set_index('목표'))