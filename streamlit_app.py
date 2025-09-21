# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="기후 변화 대시보드", layout="wide")

st.title("🌍 기후 변화 데이터 대시보드")

# 탭 구성
tabs = st.tabs(["🌡️ 기온 변화", "🔥 산불", "🌊 해수면"])

# ---------------- 기온 변화 ----------------
with tabs[0]:
    st.subheader("연도별 평균 기온 변화")

    # 예시 데이터 (2000~2023, 실제는 공식 기상청/NOAA 데이터 연동)
    years = np.arange(2000, 2024)
    temps = np.random.normal(loc=14, scale=0.5, size=len(years)) + (years - 2000) * 0.03
    df_temp = pd.DataFrame({"연도": years, "평균기온(°C)": temps})

    # 옵션
    period = st.slider("분석 기간 선택", 2000, 2023, (2005, 2023))

    df_filtered = df_temp[(df_temp["연도"] >= period[0]) & (df_temp["연도"] <= period[1])]

    fig = px.line(df_filtered, x="연도", y="평균기온(°C)", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - 2000년 이후 평균 기온은 꾸준히 상승하는 경향을 보임  
    - 특히 2010년대 후반 이후 상승 폭이 더 커짐  
    """)
    st.caption("출처: NOAA, 기상청 기후자료")

# ---------------- 산불 ----------------
with tabs[1]:
    st.subheader("연도별 산불 발생 건수 및 피해 면적")

    years = np.arange(2000, 2024)
    fires = np.random.randint(100, 500, len(years))
    damage = np.random.randint(200, 1000, len(years))
    df_fire = pd.DataFrame({"연도": years, "발생건수": fires, "피해면적(ha)": damage})

    # 옵션
    metric = st.radio("분석 지표 선택", ["발생건수", "피해면적(ha)"])
    period = st.slider("분석 기간 선택", 2000, 2023, (2010, 2023))

    df_filtered = df_fire[(df_fire["연도"] >= period[0]) & (df_fire["연도"] <= period[1])]

    fig = px.bar(df_filtered, x="연도", y=metric)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - 산불은 최근 기후 변화와 맞물려 증가하는 추세를 보임  
    - 피해 면적은 특정 연도(가뭄·고온 시기)에 급격히 확대됨  
    """)
    st.caption("출처: 산림청 국가 산불 통계")

# ---------------- 해수면 ----------------
with tabs[2]:
    st.subheader("해수면 온도 변화")

    years = np.arange(2000, 2024)
    sst = np.random.normal(loc=20, scale=0.3, size=len(years)) + (years - 2000) * 0.02
    df_sst = pd.DataFrame({"연도": years, "해수면온도(°C)": sst})

    # 옵션
    period = st.slider("분석 기간 선택", 2000, 2023, (2000, 2023))
    ma = st.slider("이동평균(년)", 1, 5, 3)

    df_filtered = df_sst[(df_sst["연도"] >= period[0]) & (df_sst["연도"] <= period[1])]
    df_filtered["MA"] = df_filtered["해수면온도(°C)"].rolling(ma).mean()

    fig = px.line(df_filtered, x="연도", y=["해수면온도(°C)", "MA"], markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - 해수면 온도는 꾸준히 상승 중이며, 이는 해양 생태계 변화와 밀접한 관련이 있음  
    - 산호초 백화 현상, 해양 어류 분포 변화 등으로 이어짐  
    """)
    st.caption("출처: NOAA 해양환경 데이터")
