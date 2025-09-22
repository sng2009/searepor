import streamlit as st
import pandas as pd
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
    st.subheader("연평균 기온 변화 (2000~2024)")
    df_temp_raw = pd.read_csv("/workspaces/searepor/datas/기온 추이_20250922110433.csv")
    df_temp = df_temp_raw.set_index('계절').loc['년평균'].reset_index()
    df_temp.columns = ['연도', '평균기온(°C)']
    df_temp['연도'] = df_temp['연도'].astype(int)
    df_temp['평균기온(°C)'] = df_temp['평균기온(°C)'].astype(float)

    period = st.slider("분석 기간 선택", 2000, 2024, (2010, 2024), key="temp_period")
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
    df_fire_total = pd.read_csv("/workspaces/searepor/datas/10년간 산불발생 현황 (연평균).csv")
    df_fire_total['면적(ha)'] = df_fire_total['면적(ha)'].replace({',':''}, regex=True).astype(float)
    df_fire_total['건수'] = df_fire_total['건수'].astype(int)

    st.markdown("### 전국 연평균 산불 현황")
    metric_total = st.selectbox("분석 지표 선택 (전국)", ["건수", "면적(ha)"], key="fire_metric_total")
    period_total = st.slider("분석 기간 선택 (전국)", 2015, 2024, (2015, 2024), key="fire_period_total")
    df_filtered_total = df_fire_total[(df_fire_total["구분"] >= period_total[0]) & (df_fire_total["구분"] <= period_total[1])]

    fig_total = px.bar(df_filtered_total, x="구분", y=metric_total)
    st.plotly_chart(fig_total, use_container_width=True)

    st.markdown("### 지역별 산불 현황")
    df_fire_region = pd.read_csv("/workspaces/searepor/datas/10년간 지역별 산불발생 현황.csv")
    df_fire_region.columns = [c.strip() for c in df_fire_region.columns]
    for col in df_fire_region.columns[1:]:
        df_fire_region[col] = df_fire_region[col].replace({',':''}, regex=True).astype(float)

    region_list = df_fire_region['구분'].tolist()
    selected_region = st.selectbox("지역 선택", region_list, key="fire_region_select")
    selected_metric_region = st.selectbox("분석 지표 선택 (지역별)", df_fire_region.columns[1:], key="fire_metric_region")

    df_region_filtered = df_fire_region[df_fire_region['구분'] == selected_region]
    region_value = df_region_filtered[selected_metric_region].values[0]
    st.write(f"{selected_region} 지역 {selected_metric_region} : {region_value}")

# ---------------- 해수면 ----------------
with tabs[2]:
    st.subheader("해수면 온도 및 해양 변화")
    df_sea = pd.read_csv("/workspaces/searepor/datas/지표및해양에8월달평균기온지표.csv")
    df_sea['Year'] = df_sea['Year'].astype(int)
    df_sea['Anomaly'] = df_sea['Anomaly'].astype(float)

    period = st.slider("분석 기간 선택", df_sea['Year'].min(), df_sea['Year'].max(), (2000, 2024), key="sea_period")
    window = st.slider("이동평균 윈도우", 1, 10, 5, key="sea_window")

    df_filtered = df_sea[(df_sea["Year"] >= period[0]) & (df_sea["Year"] <= period[1])]
    df_filtered["이동평균"] = df_filtered["Anomaly"].rolling(window).mean()

    fig = px.line(df_filtered, x="Year", y=["Anomaly", "이동평균"], markers=True,
                  labels={"value": "해수면 온도 편차 (°C)", "variable": "지표"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - 2000년 이후 해수면 온도 편차 추이  
    - 이상 고온/저온 현상 확인 가능  
    - 장기적으로 해양 생태계와 연안 지역 영향
    """)
    st.caption("출처: 지표 및 해양 8월 평균기온 지표 CSV")

# ---------------- 멸종위기종 ----------------
with tabs[3]:
    st.subheader("분류군별 멸종위기종 종 수")
    df_species = pd.read_csv("/workspaces/searepor/datas/환경부 국립생물자원관_한국의 멸종위기종_20241231..csv")
    df_species['분류군'] = df_species['분류군'].str.strip()

    species_count = df_species['분류군'].value_counts().reset_index()
    species_count.columns = ['분류군', '종 수']

    selected_groups = st.multiselect("분류군 선택", species_count['분류군'].tolist(),
                                     default=species_count['분류군'].tolist())

    df_filtered = species_count[species_count['분류군'].isin(selected_groups)]

    fig = px.bar(df_filtered, x='분류군', y='종 수', text='종 수')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - CSV 기준으로 현재 등록된 멸종위기종 종 수를 분류군별로 확인 가능  
    - 연도별 데이터는 없으므로 ‘연도 추세’ 대신 ‘분류군별 종 수’ 시각화
    """)
