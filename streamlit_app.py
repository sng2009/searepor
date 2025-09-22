# streamlit_app.py
# ------------------------------------------------------------
# 공개 데이터 출처 (URL)
# - NASA GISTEMP Global Annual Temperature Anomalies (CSV):
#   https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
# - World Bank Forest area (% of land area), Korea (JSON API):
#   https://api.worldbank.org/v2/country/KOR/indicator/AG.LND.FRST.ZS?format=json
# - 참고: 실패 시 예시 데이터로 자동 대체하며 화면에 안내 표시
# ------------------------------------------------------------

import os
import io
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib
from datetime import date

# 페이지 설정
st.set_page_config(page_title="기후 변화 대시보드", layout="wide")
st.title("📊 기후 변화와 생태계 영향 대시보드")
st.markdown("""
공식 공개 데이터와 사용자 제공 자료를 활용해  
**기후 변화 → 서식지 파괴 → 멸종위기종 증가**의 연쇄적 영향을 보여줍니다.
""")

# ------------------------------------------------------------
# 폰트 설정 (Pretendard-Bold이 있으면 적용, 없으면 생략)
# ------------------------------------------------------------
FONT_PATH = "/fonts/Pretendard-Bold.ttf"

def apply_fonts():
    try:
        if os.path.exists(FONT_PATH):
            matplotlib.font_manager.fontManager.addfont(FONT_PATH)
            matplotlib.rcParams['font.family'] = 'Pretendard Bold'
            st.markdown(f"""
                <style>
                    html, body, [class*="css"]  {{
                        font-family: "Pretendard Bold", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
                    }}
                </style>
            """, unsafe_allow_html=True)
            return "Pretendard Bold"
    except Exception:
        pass
    return None

FONT_FAMILY = apply_fonts()

def fig_set_font(fig):
    if FONT_FAMILY:
        fig.update_layout(font=dict(family=FONT_FAMILY))

# ------------------------------------------------------------
# 유틸: 미래 데이터 제거, 캐싱, 다운로드 헬퍼
# ------------------------------------------------------------
TODAY = pd.to_datetime(date.today())

def remove_future_years(df, year_col):
    max_year = TODAY.year
    return df[df[year_col] <= max_year]

def df_csv_download_button(df, filename, label="CSV 다운로드"):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")

# ------------------------------------------------------------
# 공개 데이터 불러오기 (캐싱)
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_nasa_gistemp():
    """
    NASA GISTEMP 글로벌 연평균 기온 이상치 (연도-이상치)
    출처: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
    실패 시 예시 데이터 반환 및 상태 메시지 포함
    """
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        content = r.text
        start_idx = None
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("Year"):
                start_idx = i
                break
        if start_idx is None:
            raise ValueError("데이터 헤더를 찾을 수 없음")

        csv_text = "\n".join(lines[start_idx:])
        df = pd.read_csv(io.StringIO(csv_text))
        if 'J-D' not in df.columns:
            raise ValueError("J-D 컬럼 없음")

        df_out = df[['Year', 'J-D']].dropna()
        df_out.rename(columns={'Year': '연도', 'J-D': '기온 이상치(°C)'}, inplace=True)
        df_out['연도'] = df_out['연도'].astype(int)
        df_out['기온 이상치(°C)'] = pd.to_numeric(df_out['기온 이상치(°C)'], errors='coerce')
        df_out = df_out.dropna()
        df_out = remove_future_years(df_out, '연도')
        status = None
        return df_out, status
    except Exception:
        years = list(range(2015, min(TODAY.year, 2025) + 1))
        sample = pd.DataFrame({
            '연도': years,
            '기온 이상치(°C)': np.linspace(0.84, 1.07, num=len(years))
        })
        status = "NASA GISTEMP 데이터를 불러오지 못해 예시 데이터로 대체했습니다."
        return sample, status

@st.cache_data(show_spinner=False)
def fetch_worldbank_forest(country="KOR"):
    """
    World Bank 숲 면적 비율(%), 국가별 연도 데이터
    출처: https://api.worldbank.org/v2/country/KOR/indicator/AG.LND.FRST.ZS?format=json
    실패 시 예시 데이터 반환 및 상태 메시지 포함
    """
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/AG.LND.FRST.ZS?format=json&per_page=20000"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2 or data[1] is None:
            raise ValueError("World Bank 응답 형식 오류")
        records = []
        for item in data[1]:
            year = item.get('date')
            value = item.get('value')
            if year is None:
                continue
            try:
                year = int(year)
            except:
                continue
            if value is None:
                continue
            records.append({'연도': year, '숲 면적 비율(%)': float(value)})
        df = pd.DataFrame(records).dropna()
        df = df.sort_values('연도')
        df = remove_future_years(df, '연도')
        status = None
        return df, status
    except Exception:
        years = list(range(2000, min(TODAY.year, 2024) + 1))
        vals = np.linspace(64.5, 62.0, num=len(years))
        sample = pd.DataFrame({'연도': years, '숲 면적 비율(%)': vals})
        status = "World Bank 데이터를 불러오지 못해 예시 데이터로 대체했습니다."
        return sample, status

# ------------------------------------------------------------
# 사용자 CSV 로더 (캐싱)
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv_temp():
    return pd.read_csv("/workspaces/searepor/datas/기온 추이_20250922110433.csv")

@st.cache_data(show_spinner=False)
def load_csv_fire_total():
    return pd.read_csv("/workspaces/searepor/datas/10년간 산불발생 현황 (연평균).csv")

@st.cache_data(show_spinner=False)
def load_csv_fire_region():
    return pd.read_csv("/workspaces/searepor/datas/10년간 지역별 산불발생 현황.csv")

@st.cache_data(show_spinner=False)
def load_csv_sea():
    return pd.read_csv("/workspaces/searepor/datas/지표및해양에8월달평균기온지표.csv")

@st.cache_data(show_spinner=False)
def load_csv_species():
    return pd.read_csv("/workspaces/searepor/datas/환경부 국립생물자원관_한국의 멸종위기종_20241231..csv")

# ------------------------------------------------------------
# 탭 구성
# ------------------------------------------------------------

tabs = st.tabs([
    "🛰️ 공식 공개 데이터 대시보드",
    "🌡️ 사용자: 기온 변화",
    "🔥 사용자: 산불과 서식지 파괴",
    "🌊 사용자: 해수면 및 해양 변화",
    "📉 사용자: 멸종위기종 증가"
])

# ---------------- 공식 공개 데이터 ----------------
with tabs[0]:
    st.subheader("공식 공개 데이터 기반 대시보드")

    # NASA GISTEMP
    st.markdown("#### 글로벌 연평균 기온 이상치 (NASA GISTEMP)")
    df_gistemp, status_gis = fetch_nasa_gistemp()
    if status_gis:
        st.info(status_gis)

    col1, col2 = st.columns([3, 1])
    with col1:
        year_min, year_max = int(df_gistemp['연도'].min()), int(df_gistemp['연도'].max())
        yrs = st.slider("분석 기간 선택 (연도)", year_min, year_max, (year_min, year_max), key="gistemp_year")
        df_gf = df_gistemp[(df_gistemp['연도'] >= yrs[0]) & (df_gistemp['연도'] <= yrs[1])].copy()
        window = st.slider("이동평균 윈도우 (연)", 1, 10, 5, key="gistemp_ma")
        df_gf['이동평균(°C)'] = df_gf['기온 이상치(°C)'].rolling(window).mean()

        fig = px.line(
            df_gf, x='연도', y=['기온 이상치(°C)', '이동평균(°C)'], markers=True,
            labels={"value": "기온 이상치(°C)", "variable": "지표"},
            title="글로벌 기온 이상치 추이"
        )
        fig_set_font(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        df_csv_download_button(df_gistemp, "nasa_gistemp_clean.csv", "NASA GISTEMP 정제 데이터 다운로드")

    st.caption("출처: NASA GISTEMP — https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv")

    st.markdown("---")

    # World Bank Forest Area
    st.markdown("#### 한국 숲 면적 비율 (World Bank)")
    df_wb, status_wb = fetch_worldbank_forest("KOR")
    if status_wb:
        st.info(status_wb)

    col3, col4 = st.columns([3, 1])
    with col3:
        year_min2, year_max2 = int(df_wb['연도'].min()), int(df_wb['연도'].max())
        yrs2 = st.slider("분석 기간 선택 (연도)", year_min2, year_max2, (max(year_min2, 1990), year_max2), key="wb_year")
        df_wbf = df_wb[(df_wb['연도'] >= yrs2[0]) & (df_wb['연도'] <= yrs2[1])].copy()

        # Y축 스케일 옵션: 0부터 vs 데이터 범위 맞춤
        yscale = st.radio("Y축 스케일", ["데이터 범위에 맞춤", "0부터 100"], horizontal=True, key="wb_yscale")
        fig2 = px.area(df_wbf, x='연도', y='숲 면적 비율(%)', markers=True, title="한국 숲 면적 비율 추이")
        if yscale == "데이터 범위에 맞춤":
            ymin, ymax = float(df_wbf['숲 면적 비율(%)'].min()), float(df_wbf['숲 면적 비율(%)'].max())
            pad = max((ymax - ymin) * 0.1, 0.5)
            fig2.update_yaxes(range=[ymin - pad, ymax + pad])
        else:
            fig2.update_yaxes(range=[0, 100])
        fig_set_font(fig2)
        st.plotly_chart(fig2, use_container_width=True)

        # 변화량 요약
        first_val = df_wbf.iloc[0]['숲 면적 비율(%)']
        last_val = df_wbf.iloc[-1]['숲 면적 비율(%)']
        delta_abs = last_val - first_val
        delta_pct = (delta_abs / first_val) * 100 if first_val else np.nan
        st.metric(label="기간 내 변화(종가 - 시가)", value=f"{last_val:.2f}%", delta=f"{delta_abs:.2f}p ( {delta_pct:+.2f}% )")
    with col4:
        df_csv_download_button(df_wb, "worldbank_forest_clean.csv", "World Bank 정제 데이터 다운로드")

    st.caption("출처: World Bank — https://api.worldbank.org/v2/country/KOR/indicator/AG.LND.FRST.ZS?format=json")

# ---------------- 사용자: 기온 변화 ----------------
with tabs[1]:
    st.subheader("연평균 기온 변화 (사용자 데이터)")

    df_temp_raw = load_csv_temp()
    df_temp = df_temp_raw.set_index('계절').loc['년평균'].reset_index()
    df_temp.columns = ['연도', '평균기온(°C)']
    df_temp['연도'] = pd.to_numeric(df_temp['연도'], errors='coerce').astype(int)
    df_temp['평균기온(°C)'] = pd.to_numeric(df_temp['평균기온(°C)'], errors='coerce')
    df_temp = df_temp.dropna()
    df_temp = remove_future_years(df_temp, '연도')

    colA, colB = st.columns([3, 1])
    with colA:
        period = st.slider(
            "분석 기간 선택",
            int(df_temp['연도'].min()), int(df_temp['연도'].max()),
            (int(df_temp['연도'].min()), int(df_temp['연도'].max())),
            key="temp_period_user"
        )
        df_filtered = df_temp[(df_temp["연도"] >= period[0]) & (df_temp["연도"] <= period[1])].copy()
        window_u = st.slider("이동평균 윈도우 (연)", 1, 10, 3, key="temp_ma")
        df_filtered['이동평균'] = df_filtered['평균기온(°C)'].rolling(window_u).mean()

        fig = px.line(
            df_filtered, x="연도", y=["평균기온(°C)", "이동평균"], markers=True,
            labels={"value": "기온(°C)", "variable": "지표"},
            title="연평균 기온 및 이동평균"
        )
        fig_set_font(fig)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        df_csv_download_button(df_temp, "user_temp_clean.csv", "정제 데이터 다운로드")

# ---------------- 사용자: 산불 ----------------
with tabs[2]:
    st.subheader("산불 발생 현황 및 피해 면적 (사용자 데이터)")

    # 전국 평균
    df_fire_total_raw = load_csv_fire_total().copy()
    df_fire_total_raw['면적(ha)'] = df_fire_total_raw['면적(ha)'].replace({',': ''}, regex=True)
    df_fire_total_raw['면적(ha)'] = pd.to_numeric(df_fire_total_raw['면적(ha)'], errors='coerce')
    df_fire_total_raw['건수'] = pd.to_numeric(df_fire_total_raw['건수'], errors='coerce').astype('Int64')
    df_fire_total_raw['구분'] = pd.to_numeric(df_fire_total_raw['구분'], errors='coerce').astype('Int64')
    df_fire_total = df_fire_total_raw.dropna().copy()
    df_fire_total['구분'] = df_fire_total['구분'].astype(int)
    df_fire_total = remove_future_years(df_fire_total.rename(columns={'구분': '연도'}), '연도').rename(columns={'연도': '구분'})

    colC, colD = st.columns([3, 1])
    with colC:
        metric_total = st.selectbox("전국 분석 지표 선택", ["건수", "면적(ha)"], key="fire_metric_total_user")
        period_total = st.slider(
            "전국 분석 기간 선택",
            int(df_fire_total['구분'].min()), int(df_fire_total['구분'].max()),
            (int(df_fire_total['구분'].min()), int(df_fire_total['구분'].max())),
            key="fire_period_total_user"
        )
        df_filtered_total = df_fire_total[(df_fire_total["구분"] >= period_total[0]) & (df_fire_total["구분"] <= period_total[1])].copy()
        fig_total = px.bar(
            df_filtered_total, x="구분", y=metric_total, text=metric_total,
            title=f"전국 {metric_total} 추이"
        )
        fig_set_font(fig_total)
        st.plotly_chart(fig_total, use_container_width=True)
    with colD:
        df_csv_download_button(df_fire_total, "user_fire_national_clean.csv", "정제 데이터 다운로드")

    st.markdown("---")

    # 지역별
    df_fire_region_raw = load_csv_fire_region().copy()
    df_fire_region_raw.columns = [c.strip() for c in df_fire_region_raw.columns]
    for col in df_fire_region_raw.columns[1:]:
        df_fire_region_raw[col] = df_fire_region_raw[col].replace({',': ''}, regex=True)
        df_fire_region_raw[col] = pd.to_numeric(df_fire_region_raw[col], errors='coerce')
    df_fire_region = df_fire_region_raw.dropna(subset=['구분']).copy()

    # 인터페이스
    colE, colF = st.columns([3, 1])
    with colE:
        selected_region = st.selectbox("지역 선택", df_fire_region['구분'].tolist(), key="fire_region_select_user")
        selected_metric_region = st.selectbox("분석 지표 선택 (지역별)", df_fire_region.columns[1:], key="fire_metric_region_user")

        df_region_filtered = df_fire_region[df_fire_region['구분'] == selected_region]

        # 1) 선택한 지표의 단일 값(제목 추가)
        single_title = f"{selected_region} — {selected_metric_region}"
        fig_region_single = px.bar(
            x=[selected_region],
            y=df_region_filtered[selected_metric_region],
            labels={'x': '지역', 'y': selected_metric_region},
            text=df_region_filtered[selected_metric_region],
            title=single_title
        )
        fig_set_font(fig_region_single)
        st.plotly_chart(fig_region_single, use_container_width=True)

        # 2) 같은 계열(건수/면적) 3항목 비교: 2025.09.22 / 10년평균 / 2024
        suffix = selected_metric_region.split('_')[-1]  # '건수' 또는 '면적'
        compare_cols = [c for c in df_fire_region.columns if c.endswith(suffix)]
        # 보기 좋게 라벨 정리
        label_map = {
            "2025.09.22_건수": "현재(건수)", "2025.09.22_면적": "현재(면적)",
            "10년평균_건수": "10년평균(건수)", "10년평균_면적": "10년평균(면적)",
            "2024_건수": "2024(건수)", "2024_면적": "2024(면적)"
        }
        comp_vals = df_region_filtered[compare_cols].iloc[0]
        df_comp = pd.DataFrame({
            '구분': [label_map.get(c, c) for c in compare_cols],
            '값': comp_vals.values
        })
        fig_region_comp = px.bar(
            df_comp, x='구분', y='값', text='값',
            title=f"{selected_region} — 지표 비교 ({suffix})"
        )
        fig_set_font(fig_region_comp)
        st.plotly_chart(fig_region_comp, use_container_width=True)

        # 3) 전 지역 랭킹 비교: 선택 지표 기준
        df_rank = df_fire_region[['구분', selected_metric_region]].sort_values(selected_metric_region, ascending=False)
        fig_rank = px.bar(
            df_rank, x='구분', y=selected_metric_region, text=selected_metric_region,
            title=f"전 지역 비교 — {selected_metric_region} 랭킹"
        )
        fig_rank.update_xaxes(categoryorder='array', categoryarray=df_rank['구분'].tolist())
        fig_set_font(fig_rank)
        st.plotly_chart(fig_rank, use_container_width=True)
    with colF:
        df_csv_download_button(df_fire_region, "user_fire_region_clean.csv", "정제 데이터 다운로드")

# ---------------- 사용자: 해수면 ----------------
with tabs[3]:
    st.subheader("해수면 온도 편차 및 해양 변화 (사용자 데이터)")

    df_sea_raw = load_csv_sea().copy()
    df_sea_raw['Year'] = pd.to_numeric(df_sea_raw['Year'], errors='coerce').astype(int)
    df_sea_raw['Anomaly'] = pd.to_numeric(df_sea_raw['Anomaly'], errors='coerce')
    df_sea = df_sea_raw.dropna().copy()
    df_sea = remove_future_years(df_sea.rename(columns={'Year': '연도'}), '연도').rename(columns={'연도': 'Year'})

    colG, colH = st.columns([3, 1])
    with colG:
        period = st.slider(
            "분석 기간 선택",
            int(df_sea['Year'].min()), int(df_sea['Year'].max()),
            (int(df_sea['Year'].min()), int(df_sea['Year'].max())),
            key="sea_period_user"
        )
        window = st.slider("이동평균 윈도우", 1, 10, 5, key="sea_window_user")

        df_filtered = df_sea[(df_sea["Year"] >= period[0]) & (df_sea["Year"] <= period[1])].copy()
        df_filtered["이동평균"] = df_filtered["Anomaly"].rolling(window).mean()

        fig = px.line(
            df_filtered, x="Year", y=["Anomaly", "이동평균"], markers=True,
            labels={"value": "해수면 온도 편차 (°C)", "variable": "지표"},
            title="해수면 온도 편차 및 이동평균"
        )
        fig_set_font(fig)
        st.plotly_chart(fig, use_container_width=True)
    with colH:
        df_csv_download_button(df_sea, "user_sea_anomaly_clean.csv", "정제 데이터 다운로드")

# ---------------- 사용자: 멸종위기종 ----------------
with tabs[4]:
    st.subheader("분류군별 멸종위기종 종 수 (사용자 데이터)")

    df_species_raw = load_csv_species().copy()
    df_species_raw['분류군'] = df_species_raw['분류군'].astype(str).str.strip()
    df_species = df_species_raw.dropna(subset=['분류군']).copy()

    species_count = df_species['분류군'].value_counts().reset_index()
    species_count.columns = ['분류군', '종 수']

    colI, colJ = st.columns([3, 1])
    with colI:
        selected_groups = st.multiselect(
            "분류군 선택",
            species_count['분류군'].tolist(),
            default=species_count['분류군'].tolist()
        )
        df_filtered = species_count[species_count['분류군'].isin(selected_groups)].copy()

        fig = px.bar(df_filtered, x='분류군', y='종 수', text='종 수', title="분류군별 멸종위기종 수")
        fig_set_font(fig)
        st.plotly_chart(fig, use_container_width=True)
    with colJ:
        df_csv_download_button(species_count, "user_endangered_species_counts.csv", "정제 데이터 다운로드")