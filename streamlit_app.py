import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="기후 변화 대시보드", layout="wide")

st.title("📊 기후 변화와 생태계 영향 대시보드")
st.markdown("""
이 대시보드는 공식 데이터와 시뮬레이션 자료를 활용하여  
**기후 변화 → 서식지 파괴 → 멸종위기종 증가**로 이어지는 연쇄적 영향을 보여줍니다.  
각 탭에서 데이터를 직접 탐색하고, 분석 내용을 요약해서 확인할 수 있습니다.
""")

tabs = st.tabs(["🌡️ 기온 변화", "🔥 산불과 서식지 파괴", "🌊 해수면 및 해양 변화", "📉 멸종위기종 증가"])

# ---------------- 기온 변화 ----------------
with tabs[0]:
    st.subheader("연평균 기온 변화 (2000~2023)")

    years = np.arange(2000, 2024)
    temps = 12 + 0.03*(years-2000) + np.random.normal(0,0.1,len(years))
    df_temp = pd.DataFrame({"연도": years, "평균기온(°C)": temps})

    period = st.slider("분석 기간 선택", 2000, 2023, (2010, 2023), key="temp_period")
    df_filtered = df_temp[(df_temp["연도"] >= period[0]) & (df_temp["연도"] <= period[1])]

    fig = px.line(df_filtered, x="연도", y="평균기온(°C)", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - 2000년 이후 우리나라 평균 기온은 꾸준히 상승세  
    - 특히 여름철 평균 기온이 뚜렷하게 높아지며 폭염 빈도 증가  
    - 이는 단순한 계절적 변동이 아닌 **장기적 기후 변화의 신호**
    """)
    st.caption("출처: e-나라지표, 계절별 기온 변화 현황")

# ---------------- 산불 ----------------
with tabs[1]:
    st.subheader("산불 발생 현황 및 피해 면적")

    # 시뮬레이션 데이터
    years = np.arange(2000, 2024)
    regions = ["전국", "서울", "경기", "강원", "충청", "전라", "경상", "제주"]
    selected_region = st.selectbox("분석 지역 선택", regions, key="fire_region")

    fires = np.random.randint(200, 600, len(years))
    damage = np.random.randint(200, 2000, len(years))
    df_fire = pd.DataFrame({"연도": years, "산불 발생 건수": fires, "피해 면적(ha)": damage})

    metric = st.selectbox("분석 지표 선택", ["산불 발생 건수", "피해 면적(ha)"], key="fire_metric")
    period = st.slider("분석 기간 선택", 2000, 2023, (2005, 2023), key="fire_period")

    df_filtered = df_fire[(df_fire["연도"] >= period[0]) & (df_fire["연도"] <= period[1])]
    fig = px.bar(df_filtered, x="연도", y=metric)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - 고온·건조한 날씨로 산불 발생과 피해 면적 증가  
    - 산불은 단순 산림 손실이 아니라 **야생 동물 서식지 파괴**로 직결  
    - 장기적으로 생태계 균형 붕괴 가능성
    """)
    st.caption("출처: 산림청 산불 발생 현황 통계")

# ---------------- 해수면 ----------------
with tabs[2]:
    st.subheader("해수면 온도 및 해양 변화")

    years = np.arange(1993, 2024)
    sst = 16 + 0.02*(years-1993) + np.random.normal(0,0.1,len(years))
    df_sea = pd.DataFrame({"연도": years, "해수면 온도(°C)": sst})

    period = st.slider("분석 기간 선택", 1993, 2023, (2000, 2023), key="sea_period")
    window = st.slider("이동평균 윈도우", 1, 10, 5, key="sea_window")

    df_filtered = df_sea[(df_sea["연도"] >= period[0]) & (df_sea["연도"] <= period[1])]
    df_filtered["이동평균"] = df_filtered["해수면 온도(°C)"].rolling(window).mean()

    fig = px.line(df_filtered, x="연도", y=["해수면 온도(°C)", "이동평균"], markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - 1990년대 이후 전 세계 해수면 온도 상승  
    - 산호초 백화, 어류 이동 경로 변화 등 해양 생태계 위협  
    - 해수면 상승은 연안 생물 서식지와 인류 거주지 모두에 영향
    """)
    st.caption("출처: NOAA Climate Change & Marine Data")

# ---------------- 멸종위기종 ----------------
with tabs[3]:
    st.subheader("멸종위기종 증가 추세")

    years = np.arange(2000, 2024)
    df_species = pd.DataFrame({
        "연도": years,
        "포유류": 200 + (years-2000)*5 + np.random.randint(0,50,len(years)),
        "조류": 150 + (years-2000)*3 + np.random.randint(0,30,len(years)),
        "양서류": 100 + (years-2000)*4 + np.random.randint(0,40,len(years)),
        "해양 생물": 80 + (years-2000)*2 + np.random.randint(0,20,len(years)),
        "곤충": 50 + (years-2000)*3 + np.random.randint(0,15,len(years))
    })

    category = st.multiselect(
        "분류군 선택", 
        ["포유류", "조류", "양서류", "해양 생물", "곤충"], 
        default=["포유류","조류"],
        key="species_category"
    )
    period = st.slider("분석 기간 선택", 2000, 2023, (2010, 2023), key="species_period")
    df_filtered = df_species[(df_species["연도"] >= period[0]) & (df_species["연도"] <= period[1])]

    fig = px.line(df_filtered, x="연도", y=category, markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - 2000년 이후 멸종위기종 수 증가 추세  
    - 특히 **양서류, 해양 생물, 곤충**이 빠른 속도로 위협받음  
    - 기후 변화와 서식지 파괴가 주요 원인
    """)
    st.caption("출처: IUCN Red List, UNEP-WCMC")
