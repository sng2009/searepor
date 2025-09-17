# streamlit_app.py
"""
Streamlit 앱: 해수 온난화 대시보드 (공식 공개 데이터 + 사용자 입력 데이터 기반)
- 한글 UI
- 자동 캐시, 전처리, CSV 내보내기
- 공개 데이터 로드 실패 시 예시 데이터로 대체 및 한국어 안내 표시

공식 공개 데이터(예시):
- NOAA OISST v2.1 (해수면 온도 그리드) - https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html
- NASA MUR / GHRSST (고해상도 SST) - https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1
- CSIRO / NOAA 해수면 상승 자료 / World Bank Sea-Level Dataset
  - CSIRO sea level summary: https://www.cmar.csiro.au/sealevel/sl_hist_last_decades.html
  - NOAA sea level trends: https://tidesandcurrents.noaa.gov/sltrends/
  - World Bank sea level datasets: https://datacatalog.worldbank.org/

주의: Codespaces / 로컬에서 실행 시 네트워크 접근이 필요함.
"""

import io
import datetime
from functools import partial

import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import requests
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# 폰트 시도: Pretendard (없으면 무시)
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"

# ---------- 유틸 ----------

def try_apply_pretendard():
    try:
        import matplotlib as mpl
        mpl.font_manager.fontManager.addfont(PRETENDARD_PATH)
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Pretendard', 'DejaVu Sans']
    except Exception:
        # 환경에 폰트가 없거나 실패해도 조용히 진행
        pass

try_apply_pretendard()

def download_text(url, timeout=20):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

# ---------- 데이터 로드 (공식 공개 데이터) ----------
# 시도: NOAA OISST (OPeNDAP 접근) - OISST v2.1
# 출처 주석:
# NOAA OISST v2.1: https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html
# NOAA OISST product page: https://www.ncei.noaa.gov/products/optimum-interpolation-sst

OISST_OPENDAP = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.nc"

@st.cache_data(show_spinner=False)
def load_noaa_oisst_subset(time_start=None, time_end=None, bbox=None, max_days=3650):
    """
    NOAA OISST를 OPeNDAP으로 열어 부분적으로 로드.
    bbox = (lon_min, lon_max, lat_min, lat_max) in degrees
    time_start/time_end: 'YYYY-MM-DD' strings or None
    max_days: cap to avoid huge downloads
    반환: pandas DataFrame(date, value, lon, lat)
    """
    try:
        ds = xr.open_dataset(OISST_OPENDAP)
        # 시간 범위 지정
        if time_end is None:
            time_end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        if time_start is None:
            # 기본: 최근 10년
            time_start = (pd.to_datetime(time_end) - pd.Timedelta(days=365*10)).strftime("%Y-%m-%d")
        # cap days
        t0 = pd.to_datetime(time_start)
        t1 = pd.to_datetime(time_end)
        if (t1 - t0).days > max_days:
            t0 = t1 - pd.Timedelta(days=max_days)
        ds_sub = ds.sel(time=slice(str(t0.date()), str(t1.date())))
        if bbox:
            lon_min, lon_max, lat_min, lat_max = bbox
            ds_sub = ds_sub.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        # take daily mean SST and compute regional avg
        sst = ds_sub['sst']
        # convert to pandas by averaging spatially to keep size small
        sst_mean = sst.mean(dim=['lon', 'lat']).to_series()
        df = sst_mean.reset_index()
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'])
        # remove future dates (요구사항)
        today_local = pd.Timestamp.now().normalize()
        df = df[df['date'] <= today_local]
        df['source'] = 'NOAA_OISST_mean'
        return df
    except Exception as e:
        raise e

# 예시 데이터 (공개 데이터 로드 실패 시 대체)
@st.cache_data(show_spinner=False)
def example_official_sst():
    # 간단한 예시: 연도별 전세계 평균 SST 추세 (임의 생성, 단위: °C anomaly)
    years = np.arange(1981, 2025)
    # 완만한 상승 곡선 + 잡음
    values = 0.02 * (years - 1981) + 0.1 * np.sin(np.linspace(0, 6.28, len(years))) + 14.0
    df = pd.DataFrame({'date': pd.to_datetime([f"{y}-07-15" for y in years]), 'value': values})
    df['source'] = 'EXAMPLE_OFFICIAL'
    return df

# ---------- 사용자 입력(프롬프트 제공 텍스트) 기반 데이터 ----------
# 입력(프롬프트)에서 사용 가능한 값들:
# - 1968~2017 우리나라 주변 수온 +1.23°C (탄소중립포털)
# - 전세계 평균 상승폭 +0.48°C (기간 동일 비교)
# - 10년마다 약 0.2°C 상승 (전세계 추세)
# - OCPC 2025년 7월: 동아시아 +1.2°C, 동해 +3.4°C, 황해 +2.7°C
# - 해수면: 1993~2010 우리나라 연안 +3.2 mm/yr (OCPC 등)
# - 일부 지역 최근 +5 mm/yr 이상
# - 2100년 예측 최대 +82 cm (KBS 보도자료 인용)
@st.cache_data(show_spinner=False)
def build_user_data_from_report():
    # 1) 연도 기반 수온 추이 (간단한 재구성)
    years = np.arange(1968, 2023)
    # create korea_nearby anomaly (start 0 in 1968, reach +1.23 by 2017 -> linear approx)
    korea_anom = np.interp(years, [1968, 2017], [0.0, 1.23])
    global_anom = np.interp(years, [1968, 2017], [0.0, 0.48])
    df = pd.DataFrame({
        'year': years,
        '한국_인근_수온_편차(℃)': korea_anom,
        '세계_평균_수온_편차(℃)': global_anom
    })
    # 2) 2025년 7월 지역별 이상치 (OCPC 인용 수치)
    region_df = pd.DataFrame({
        '지역': ['동아시아 평균', '동해(East Sea)', '황해(Yellow Sea)'],
        '7월_편차(℃)': [1.2, 3.4, 2.7],
        '위도': [35.0, 37.5, 35.0],
        '경도': [125.0, 131.0, 124.0]
    })
    # 3) 해수면 상승 추세 요약 (연도, mm/yr)
    sea_df = pd.DataFrame({
        'period': ['1993-2010', '최근 일부지역(예시)', '2100 예측(최대)'],
        'rate': [3.2, 5.0, 820.0],  # 마지막은 cm -> convert note
        'unit': ['mm/year', 'mm/year', 'mm (총 예측)']
    })
    return df, region_df, sea_df

# ---------- 전처리 공통 규칙 함수 ----------
def standardize_time_series(df, time_col, value_col, group_col=None):
    """
    표준화: date, value, group(optional)
    """
    df2 = df.copy()
    df2 = df2.rename(columns={time_col: 'date', value_col: 'value'}) if time_col != 'date' or value_col != 'value' else df2
    if 'date' in df2.columns:
        df2['date'] = pd.to_datetime(df2['date'])
    # drop future dates
    today_local = pd.Timestamp.now().normalize()
    if 'date' in df2.columns:
        df2 = df2[df2['date'] <= today_local]
    # 결측/중복 처리
    df2 = df2.drop_duplicates()
    df2['value'] = pd.to_numeric(df2['value'], errors='coerce')
    df2 = df2.dropna(subset=['value'])
    if group_col and group_col in df2.columns:
        df2 = df2.rename(columns={group_col: 'group'})
    return df2

# ---------- Streamlit UI ----------
st.set_page_config(page_title="해수 온난화 대시보드", layout="wide", initial_sidebar_state="expanded")

st.title("해수 온난화와 청소년 인식 — 데이터 대시보드")
st.markdown("공식 공개 데이터 기반 대시보드와 *입력된 보고서 내용* 기반 대시보드를 차례로 보여줘.")

# 사이드바: 범위 및 옵션 (공개 데이터)
st.sidebar.header("공개 데이터 옵션")
with st.sidebar.form("official_form"):
    bbox_choice = st.selectbox("지역 선택 (공개데이터 평균 계산 시)", ("전세계", "동아시아(대략)", "한국 주변(동해/황해)"))
    days_back = st.number_input("최근 N일(최대 3650):", min_value=30, max_value=3650, value=365*5)
    submit_off = st.form_submit_button("공개 데이터 불러오기")
if submit_off:
    st.sidebar.success("공개 데이터 로드 시도함")

# 공개 데이터 로드 시도
official_df = None
official_msg = ""
try:
    if bbox_choice == "전세계":
        bbox = None
    elif bbox_choice == "동아시아(대략)":
        bbox = (100, 150, 10, 50)  # lon_min, lon_max, lat_min, lat_max
    else:
        bbox = (120, 140, 30, 45)  # 한국 주변 대략
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.Timedelta(days=int(days_back))
    # 시도해서 불러오기
    official_df = load_noaa_oisst_subset(time_start=start_date.strftime("%Y-%m-%d"),
                                         time_end=end_date.strftime("%Y-%m-%d"),
                                         bbox=bbox,
                                         max_days=3650)
    official_msg = "NOAA OISST (공식)에서 시계열을 가져왔음"
except Exception as e:
    st.warning("공개 데이터(노아아 OISST) 로드 실패 - 예시 데이터로 대체함. (네트워크/OPeNDAP 접근 필요).")
    official_df = example_official_sst()
    official_msg = "예시 공개 데이터 사용(다운로드 실패 대체)"
    # 화면에 실패 사유 간단 표기
    st.sidebar.error(f"공개 데이터 로드 오류: {str(e)[:200]}")

# 전처리 표준화
if 'date' in official_df.columns:
    public_ts = standardize_time_series(official_df, 'date', 'value')
else:
    public_ts = standardize_time_series(official_df, 'date', 'value')

# 공개 데이터 UI 출력
st.subheader("공식 공개 데이터 기반 대시보드")
st.caption(f"데이터 출처(주요): NOAA OISST v2.1 (OPeNDAP) 등. ({official_msg})")
# show small metrics
col1, col2, col3 = st.columns([1,1,2])
with col1:
    st.metric("기간 시작", public_ts['date'].min().date() if not public_ts.empty else "N/A")
with col2:
    st.metric("기간 종료", public_ts['date'].max().date() if not public_ts.empty else "N/A")
with col3:
    st.metric("샘플 개수", int(len(public_ts)))

# 공개 데이터 시계열: smoothing 옵션
st.sidebar.markdown("----")
st.sidebar.header("시각화 옵션 (공개)")
smooth = st.sidebar.checkbox("이동평균(7일)", value=True)
resample = st.sidebar.selectbox("일간 / 주간 / 월간", ("일간", "주간", "월간"))

def resample_ts(df, rule):
    df2 = df.copy()
    df2 = df2.set_index('date').sort_index()
    if rule == "주간":
        df2 = df2.resample('W').mean()
    elif rule == "월간":
        df2 = df2.resample('M').mean()
    else:
        df2 = df2.resample('D').mean()
    df2 = df2.dropna().reset_index()
    return df2

plot_df = resample_ts(public_ts[['date','value']], resample)
if smooth:
    plot_df['value_sm'] = plot_df['value'].rolling(window=7, min_periods=1, center=False).mean()
    ycol = 'value_sm'
else:
    ycol = 'value'

fig1 = px.line(plot_df, x='date', y=ycol, title="공식 공개 데이터: 평균 해수면 온도(시계열)")
fig1.update_layout(xaxis_title="날짜", yaxis_title="값 (°C 또는 데이터 단위)")
st.plotly_chart(fig1, use_container_width=True)

# 공개 데이터 표 및 CSV 다운로드
st.markdown("#### 공개 데이터 표 (전처리 결과)")
st.dataframe(plot_df.head(200))
csv_buf = io.StringIO()
plot_df.to_csv(csv_buf, index=False)
st.download_button("공개 데이터 전처리 CSV 다운로드", csv_buf.getvalue(), file_name="official_sst_preprocessed.csv", mime="text/csv")

# ---------- 사용자 입력 대시보드 (프롬프트 텍스트 기반) ----------
st.markdown("---")
st.subheader("입력 데이터 기반 대시보드 (보고서 내용 재구성)")

user_ts, user_regions, user_sea = build_user_data_from_report()

# 표준화: 연도->date (중간 연도 날짜 사용)
user_ts_plot = user_ts.copy()
user_ts_plot['date'] = pd.to_datetime(user_ts_plot['year'].astype(str) + "-07-01")
user_ts_plot = user_ts_plot[['date', '한국_인근_수온_편차(℃)', '세계_평균_수온_편차(℃)']]

# 멀티라인 플롯
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=user_ts_plot['date'], y=user_ts_plot['한국_인근_수온_편차(℃)'],
                          mode='lines', name='한국 주변 수온 편차 (℃)'))
fig2.add_trace(go.Scatter(x=user_ts_plot['date'], y=user_ts_plot['세계_평균_수온_편차(℃)'],
                          mode='lines', name='세계 평균 수온 편차 (℃)'))
fig2.update_layout(title="1968~2022 재구성: 한국 인근 vs 세계 평균 수온 편차",
                   xaxis_title="연도", yaxis_title="편차 (℃)")
st.plotly_chart(fig2, use_container_width=True)

# 지역별 2025년 7월 편차 (지도 + 바)
st.markdown("##### 2025년 7월 지역별 수온 편차 (보고서 수치 기반)")
fig3 = px.bar(user_regions, x='지역', y='7월_편차(℃)', text='7월_편차(℃)')
fig3.update_layout(yaxis_title="편차 (℃)", xaxis_title="지역")
st.plotly_chart(fig3, use_container_width=True)

# 지도: 간단한 scatter_geo로 위치 표시
fig_map = px.scatter_geo(user_regions,
                         lat='위도', lon='경도',
                         hover_name='지역',
                         size='7월_편차(℃)',
                         projection="natural earth",
                         title="지역별 7월 편차 위치 (대략 좌표)")
st.plotly_chart(fig_map, use_container_width=True)

# 해수면 상승 요약
st.markdown("##### 해수면 상승 요약 (보고서 발췌)")
st.table(user_sea)

# 사용자 대시보드 사이드바 옵션 자동 구성 (기간 슬라이더, 단위 변환 등)
st.sidebar.header("사용자 데이터 옵션")
year_min = int(user_ts['year'].min())
year_max = int(user_ts['year'].max())
years_selected = st.sidebar.slider("연도 범위 선택 (보고서 기반 데이터)", min_value=year_min, max_value=year_max, value=(year_min, year_max))
smooth_user = st.sidebar.checkbox("사용자 데이터 이동평균(5년)", value=False)
# apply filters
mask = (user_ts_plot['date'].dt.year >= years_selected[0]) & (user_ts_plot['date'].dt.year <= years_selected[1])
user_plot_filtered = user_ts_plot.loc[mask].copy()
if smooth_user:
    user_plot_filtered['한국_sm'] = user_plot_filtered['한국_인근_수온_편차(℃)'].rolling(window=5, min_periods=1).mean()
    user_plot_filtered['세계_sm'] = user_plot_filtered['세계_평균_수온_편차(℃)'].rolling(window=5, min_periods=1).mean()
    fig_user = px.line(user_plot_filtered, x='date', y=['한국_sm', '세계_sm'], labels={'value':'편차(℃)','date':'연도'})
else:
    fig_user = px.line(user_plot_filtered, x='date', y=['한국_인근_수온_편차(℃)', '세계_평균_수온_편차(℃)'])
fig_user.update_layout(title="보고서 기반 수온 편차(선택한 범위)")
st.plotly_chart(fig_user, use_container_width=True)

# 사용자 데이터 전처리된 표 다운로드
buf2 = io.StringIO()
user_ts.to_csv(buf2, index=False)
st.download_button("보고서 기반 전처리 CSV 다운로드", buf2.getvalue(), file_name="user_report_reconstructed.csv", mime="text/csv")

# ---------- 추가 정보 / 출처 표시 ----------
st.markdown("---")
st.markdown("### 출처 및 참고 (코드 주석에 URL 명시)")
st.markdown("""
- NOAA OISST v2.1 (OPeNDAP): https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html  
- GHRSST / MUR (NASA JPL): https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1  
- CSIRO sea level summary: https://www.cmar.csiro.au/sealevel/sl_hist_last_decades.html  
- NOAA sea level trends: https://tidesandcurrents.noaa.gov/sltrends/  
- World Bank Sea-Level datasets: https://datacatalog.worldbank.org/
""")

st.markdown("앱 구현 노트: 공개 데이터 접근은 Codespaces/로컬 환경에서 네트워크가 허용되어야 함. OPeNDAP/THREDDS 접근 실패 시 예시 데이터로 자동 대체됨.")

# ---------- 끝 ----------
