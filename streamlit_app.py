# streamlit_app.py
"""
Streamlit 앱: 해수 온난화 대시보드 (공식 공개 데이터 + 사용자 입력 데이터 기반)
- 한글 UI
- 자동 캐시, 전처리, CSV 내보내기
- 공개 데이터 로드 실패 시 예시 데이터로 대체 및 한국어 안내 표시
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

# ---------- 유틸 ----------
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"

def try_apply_pretendard():
    try:
        import matplotlib as mpl
        mpl.font_manager.fontManager.addfont(PRETENDARD_PATH)
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Pretendard', 'DejaVu Sans']
    except Exception:
        pass

try_apply_pretendard()

def download_text(url, timeout=20):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

# ---------- 데이터 로드 ----------
OISST_OPENDAP = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.nc"

@st.cache_data(show_spinner=False)
def load_noaa_oisst_subset(time_start=None, time_end=None, bbox=None, max_days=3650):
    try:
        ds = xr.open_dataset(OISST_OPENDAP)
        if time_end is None:
            time_end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        if time_start is None:
            time_start = (pd.to_datetime(time_end) - pd.Timedelta(days=365*10)).strftime("%Y-%m-%d")
        t0 = pd.to_datetime(time_start)
        t1 = pd.to_datetime(time_end)
        if (t1 - t0).days > max_days:
            t0 = t1 - pd.Timedelta(days=max_days)
        ds_sub = ds.sel(time=slice(str(t0.date()), str(t1.date())))
        if bbox:
            lon_min, lon_max, lat_min, lat_max = bbox
            ds_sub = ds_sub.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        sst = ds_sub['sst']
        sst_mean = sst.mean(dim=['lon', 'lat']).to_series()
        df = sst_mean.reset_index()
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'])
        today_local = pd.Timestamp.now().normalize()
        df = df[df['date'] <= today_local]
        df['source'] = 'NOAA_OISST_mean'
        return df
    except Exception as e:
        raise e

@st.cache_data(show_spinner=False)
def example_official_sst():
    years = np.arange(1981, 2025)
    values = 0.02 * (years - 1981) + 0.1 * np.sin(np.linspace(0, 6.28, len(years))) + 14.0
    df = pd.DataFrame({'date': pd.to_datetime([f"{y}-07-15" for y in years]), 'value': values})
    df['source'] = 'EXAMPLE_OFFICIAL'
    return df

# ---------- 사용자 입력 데이터 ----------
@st.cache_data(show_spinner=False)
def build_user_data_from_report():
    years = np.arange(1968, 2023)
    korea_anom = np.interp(years, [1968, 2017], [0.0, 1.23])
    global_anom = np.interp(years, [1968, 2017], [0.0, 0.48])
    df = pd.DataFrame({
        'year': years,
        '한국_인근_수온_편차(℃)': korea_anom,
        '세계_평균_수온_편차(℃)': global_anom
    })
    region_df = pd.DataFrame({
        '지역': ['동아시아 평균', '동해(East Sea)', '황해(Yellow Sea)'],
        '7월_편차(℃)': [1.2, 3.4, 2.7],
        '위도': [35.0, 37.5, 35.0],
        '경도': [125.0, 131.0, 124.0]
    })
    sea_df = pd.DataFrame({
        'period': ['1993-2010', '최근 일부지역(예시)', '2100 예측(최대)'],
        'rate': [3.2, 5.0, 820.0],
        'unit': ['mm/year', 'mm/year', 'mm (총 예측)']
    })
    return df, region_df, sea_df

# ---------- 전처리 ----------
def standardize_time_series(df, time_col, value_col, group_col=None):
    df2 = df.copy()
    if time_col != 'date' or value_col != 'value':
        df2 = df2.rename(columns={time_col: 'date', value_col: 'value'})
    df2['date'] = pd.to_datetime(df2['date'])
    today_local = pd.Timestamp.now().normalize()
    df2 = df2[df2['date'] <= today_local]
    df2 = df2.drop_duplicates()
    df2['value'] = pd.to_numeric(df2['value'], errors='coerce')
    df2 = df2.dropna(subset=['value'])
    if group_col and group_col in df2.columns:
        df2 = df2.rename(columns={group_col: 'group'})
    return df2

# ---------- Streamlit UI ----------
st.set_page_config(page_title="해수 온난화 대시보드", layout="wide", initial_sidebar_state="expanded")

st.title("해수 온난화와 청소년 인식 — 데이터 대시보드")

st.sidebar.header("공개 데이터 옵션")
with st.sidebar.form("official_form"):
    bbox_choice = st.selectbox("지역 선택", ("전세계", "동아시아(대략)", "한국 주변"))
    days_back = st.number_input("최근 N일(최대 3650):", min_value=30, max_value=3650, value=365*5)
    submit_off = st.form_submit_button("공개 데이터 불러오기")

if submit_off:
    st.sidebar.success("공개 데이터 로드 시도함")

official_df = None
try:
    if bbox_choice == "전세계":
        bbox = None
    elif bbox_choice == "동아시아(대략)":
        bbox = (100, 150, 10, 50)
    else:
        bbox = (120, 140, 30, 45)
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.Timedelta(days=int(days_back))
    official_df = load_noaa_oisst_subset(time_start=start_date.strftime("%Y-%m-%d"),
                                         time_end=end_date.strftime("%Y-%m-%d"),
                                         bbox=bbox,
                                         max_days=3650)
    official_msg = "NOAA OISST (공식) 데이터"
except Exception as e:
    st.warning("공개 데이터 로드 실패 - 예시 데이터 사용")
    official_df = example_official_sst()
    official_msg = "예시 공개 데이터"

public_ts = standardize_time_series(official_df, 'date', 'value')

st.subheader("공식 공개 데이터")
col1, col2, col3 = st.columns([1,1,2])
with col1:
    st.metric("기간 시작", public_ts['date'].min().strftime("%Y-%m-%d") if not public_ts.empty else "N/A")
with col2:
    st.metric("기간 종료", public_ts['date'].max().strftime("%Y-%m-%d") if not public_ts.empty else "N/A")
with col3:
    st.metric("샘플 개수", int(len(public_ts)))

def resample_ts(df, rule):
    df2 = df.set_index('date').sort_index()
    if rule == "주간":
        df2 = df2.resample('W').mean()
    elif rule == "월간":
        df2 = df2.resample('M').mean()
    else:
        df2 = df2.resample('D').mean()
    return df2.dropna().reset_index()

st.sidebar.header("시각화 옵션")
smooth = st.sidebar.checkbox("이동평균(7일)", value=True)
resample = st.sidebar.selectbox("주기", ("일간", "주간", "월간"))

plot_df = resample_ts(public_ts[['date','value']], resample)
if smooth:
    plot_df['value_sm'] = plot_df['value'].rolling(window=7, min_periods=1).mean()
    ycol = 'value_sm'
else:
    ycol = 'value'

fig1 = px.line(plot_df, x='date', y=ycol, title="공식 공개 데이터 시계열")
st.plotly_chart(fig1, use_container_width=True)

# ---------- 사용자 데이터 ----------
st.subheader("보고서 기반 데이터")
user_ts, user_regions, user_sea = build_user_data_from_report()
user_ts_plot = user_ts.copy()
user_ts_plot['date'] = pd.to_datetime(user_ts_plot['year'].astype(str) + "-07-01")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=user_ts_plot['date'], y=user_ts_plot['한국_인근_수온_편차(℃)'], mode='lines', name='한국 주변'))
fig2.add_trace(go.Scatter(x=user_ts_plot['date'], y=user_ts_plot['세계_평균_수온_편차(℃)'], mode='lines', name='세계 평균'))
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.bar(user_regions, x='지역', y='7월_편차(℃)', text='7월_편차(℃)')
st.plotly_chart(fig3, use_container_width=True)

fig_map = px.scatter_geo(user_regions, lat='위도', lon='경도',
                         hover_name='지역', size='7월_편차(℃)',
                         projection="natural earth")
st.plotly_chart(fig_map, use_container_width=True)

st.table(user_sea)

# ---------- 출처 ----------
st.markdown("---")
st.markdown("출처: NOAA OISST v2.1, NASA GHRSST, CSIRO, World Bank 등")
