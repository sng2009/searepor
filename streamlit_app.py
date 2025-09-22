import os
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib
from datetime import date

# ---------------- 기본 설정 ----------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = BASE_DIR  # CSV가 루트에 있을 경우
TODAY = pd.to_datetime(date.today())

st.set_page_config(page_title="기후 변화 대시보드", layout="wide")
st.title("📊 기후 변화와 생태계 영향 대시보드")
st.markdown("""
공식 공개 데이터와 사용자 제공 자료를 활용해  
**기후 변화 → 서식지 파괴 → 멸종위기종 증가**의 연쇄적 영향을 보여줍니다.
""")

# ---------------- 폰트 설정 ----------------
FONT_PATH = os.path.join(BASE_DIR, "fonts", "Pretendard-Bold.ttf")
def apply_fonts():
    try:
        if os.path.exists(FONT_PATH):
            matplotlib.font_manager.fontManager.addfont(FONT_PATH)
            matplotlib.rcParams['font.family'] = 'Pretendard Bold'
            st.markdown(f"""
                <style>
                    html, body, [class*="css"]  {{
                        font-family: "Pretendard Bold", sans-serif;
                    }}
                </style>
            """, unsafe_allow_html=True)
            return "Pretendard Bold"
    except:
        pass
    return None
FONT_FAMILY = apply_fonts()
def fig_set_font(fig):
    if FONT_FAMILY:
        fig.update_layout(font=dict(family=FONT_FAMILY))

# ---------------- 유틸 ----------------
def remove_future_years(df, year_col):
    return df[df[year_col] <= TODAY.year]

def df_csv_download_button(df, filename, label="CSV 다운로드"):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")

# ---------------- 공개 데이터 ----------------
@st.cache_data
def fetch_nasa_gistemp():
    backup_path = os.path.join(DATA_DIR, "nasa_gistemp_backup.csv")
    try:
        # 1차 시도: NASA API
        url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        lines = r.text.splitlines()
    except:
        # 실패 시 로컬 백업 사용
        if os.path.exists(backup_path):
            with open(backup_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        else:
            years = list(range(2015, TODAY.year + 1))
            df = pd.DataFrame({'연도': years, '기온 이상치(°C)': np.linspace(0.84, 1.07, len(years))})
            return df, "NASA API 및 로컬 백업 모두 실패 — 예시 데이터 사용"

    # 헤더 행 찾기
    start_idx = next(i for i, line in enumerate(lines) if line.startswith("Year"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start_idx:])))

    # *** → NaN 처리
    df = df.replace("***", np.nan)

    # 연도와 J-D만 추출, J-D 없는 행 제거
    df = df[['Year', 'J-D']].dropna(subset=['J-D'])
    df.columns = ['연도', '기온 이상치(°C)']
    df['연도'] = df['연도'].astype(int)
    df['기온 이상치(°C)'] = pd.to_numeric(df['기온 이상치(°C)'], errors='coerce')

    return remove_future_years(df, '연도'), None

@st.cache_data
def fetch_worldbank_forest():
    url = "https://api.worldbank.org/v2/country/KOR/indicator/AG.LND.FRST.ZS?format=json&per_page=20000"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()[1]
        df = pd.DataFrame([{'연도': int(d['date']), '숲 면적 비율(%)': d['value']} for d in data if d['value']])
        return remove_future_years(df, '연도'), None
    except:
        years = list(range(2000, TODAY.year + 1))
        df = pd.DataFrame({'연도': years, '숲 면적 비율(%)': np.linspace(64.5, 62.0, len(years))})
        return df, "World Bank API 실패로 예시 데이터 사용"

# ---------------- 사용자 CSV 로더 ----------------
@st.cache_data
def load_csv(filename):
    return pd.read_csv(os.path.join(DATA_DIR, filename))
# ---------------- 탭 구성 ----------------
tabs = st.tabs([
    "🛰️ 공식 공개 데이터 대시보드",
    "🌡️ 사용자: 기온 변화",
    "🔥 사용자: 산불과 서식지 파괴",
    "🌊 사용자: 해수면 및 해양 변화",
    "📉 사용자: 멸종위기종 증가"
])

# --- 공식 데이터 ---
with tabs[0]:
    st.subheader("글로벌 연평균 기온 이상치 (NASA GISTEMP)")
    df_gistemp, status_gis = fetch_nasa_gistemp()
    if status_gis:
        st.info(status_gis)

    # 기간 선택
    yrs = st.slider(
        "기간",
        int(df_gistemp['연도'].min()), int(df_gistemp['연도'].max()),
        (int(df_gistemp['연도'].min()), int(df_gistemp['연도'].max()))
    )
    df_gf = df_gistemp[(df_gistemp['연도'] >= yrs[0]) & (df_gistemp['연도'] <= yrs[1])]

    # 통계 계산
    avg_val = df_gf['기온 이상치(°C)'].mean()
    max_row = df_gf.loc[df_gf['기온 이상치(°C)'].idxmax()]
    min_row = df_gf.loc[df_gf['기온 이상치(°C)'].idxmin()]
    last_val = df_gf.iloc[-1, 1]

    # 변화율 계산 (평균 기준)
    기준값 = avg_val
    if abs(기준값) < 0.1:
        change_display = f"{last_val - 기준값:+.2f}°C"
    else:
        rate = ((last_val - 기준값) / abs(기준값)) * 100
        change_display = f"{rate:+.1f}%"

    # 카드 UI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("평균", f"{avg_val:.2f}°C")
    col2.metric("최댓값", f"{max_row['기온 이상치(°C)']:.2f}°C", f"{int(max_row['연도'])}년")
    col3.metric("최솟값", f"{min_row['기온 이상치(°C)']:.2f}°C", f"{int(min_row['연도'])}년")
    col4.metric("변화율(평균 대비)", change_display)

    # 그래프
    fig = px.line(df_gf, x='연도', y='기온 이상치(°C)', markers=True)
    fig.add_hline(y=avg_val, line_dash="dot", line_color="green", annotation_text=f"평균 {avg_val:.2f}°C")
    fig.add_scatter(
        x=[max_row['연도']], y=[max_row['기온 이상치(°C)']],
        mode="markers+text", text=[f"최댓값 {max_row['기온 이상치(°C)']:.2f}°C"], textposition="top center"
    )
    fig.add_scatter(
        x=[min_row['연도']], y=[min_row['기온 이상치(°C)']],
        mode="markers+text", text=[f"최솟값 {min_row['기온 이상치(°C)']:.2f}°C"], textposition="bottom center"
    )
    fig_set_font(fig)
    st.plotly_chart(fig, use_container_width=True)

# --- 사용자: 기온 ---
with tabs[1]:
    st.subheader("연평균 기온 변화")
    df_temp_raw = load_csv("기온 추이_20250922110433.csv")
    df_temp = df_temp_raw.set_index('계절').loc['년평균'].reset_index()
    df_temp.columns = ['연도', '평균기온(°C)']
    df_temp['연도'] = df_temp['연도'].astype(int)
    df_temp['평균기온(°C)'] = df_temp['평균기온(°C)'].astype(float)

    # 기간 및 이동평균 윈도우 선택
    yrs = st.slider(
        "기간",
        int(df_temp['연도'].min()), int(df_temp['연도'].max()),
        (int(df_temp['연도'].min()), int(df_temp['연도'].max()))
    )
    window = st.slider("이동평균 윈도우", 1, 10, 3)

    # 데이터 필터링 및 이동평균 계산
    df_f = df_temp[(df_temp['연도'] >= yrs[0]) & (df_temp['연도'] <= yrs[1])]
    df_f['이동평균'] = df_f['평균기온(°C)'].rolling(window).mean()

    # 통계 계산
    avg_val = df_f['평균기온(°C)'].mean()
    max_row = df_f.loc[df_f['평균기온(°C)'].idxmax()]
    min_row = df_f.loc[df_f['평균기온(°C)'].idxmin()]
    last_val = df_f.iloc[-1, 1]

    # 변화율 계산 (평균 기준)
    기준값 = avg_val
    if abs(기준값) < 0.1:
        change_display = f"{last_val - 기준값:+.2f}°C"
    else:
        rate = ((last_val - 기준값) / abs(기준값)) * 100
        change_display = f"{rate:+.1f}%"

    # 카드 UI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("평균", f"{avg_val:.2f}°C")
    col2.metric("최댓값", f"{max_row['평균기온(°C)']:.2f}°C", f"{int(max_row['연도'])}년")
    col3.metric("최솟값", f"{min_row['평균기온(°C)']:.2f}°C", f"{int(min_row['연도'])}년")
    col4.metric("변화율(평균 대비)", change_display)

    # 그래프
    fig = px.line(df_f, x='연도', y=['평균기온(°C)', '이동평균'], markers=True)
    fig.add_hline(y=avg_val, line_dash="dot", line_color="green", annotation_text=f"평균 {avg_val:.2f}°C")
    fig.add_scatter(
        x=[max_row['연도']], y=[max_row['평균기온(°C)']],
        mode="markers+text", text=[f"최댓값 {max_row['평균기온(°C)']:.2f}°C"], textposition="top center"
    )
    fig.add_scatter(
        x=[min_row['연도']], y=[min_row['평균기온(°C)']],
        mode="markers+text", text=[f"최솟값 {min_row['평균기온(°C)']:.2f}°C"], textposition="bottom center"
    )
    fig_set_font(fig)
    st.plotly_chart(fig, use_container_width=True)

# --- 사용자: 산불과 서식지 파괴 ---
with tabs[2]:
    st.subheader("산불 발생 현황 및 피해 면적")

    # 전국 평균 데이터 불러오기
    df_fire_total = load_csv("10년간 산불발생 현황 (연평균).csv")
    df_fire_total['면적(ha)'] = df_fire_total['면적(ha)'].replace({',':''}, regex=True).astype(float)
    df_fire_total['건수'] = df_fire_total['건수'].astype(int)

    # 전국 분석
    metric_total = st.selectbox("전국 분석 지표 선택", ["건수", "면적(ha)"])
    yrs_fire = st.slider(
        "전국 분석 기간",
        int(df_fire_total['구분'].min()), int(df_fire_total['구분'].max()),
        (int(df_fire_total['구분'].min()), int(df_fire_total['구분'].max()))
    )
    df_fire_filtered = df_fire_total[(df_fire_total['구분'] >= yrs_fire[0]) & (df_fire_total['구분'] <= yrs_fire[1])]

    # 통계 계산
    avg_val = df_fire_filtered[metric_total].mean()
    max_row = df_fire_filtered.loc[df_fire_filtered[metric_total].idxmax()]
    min_row = df_fire_filtered.loc[df_fire_filtered[metric_total].idxmin()]
    last_val = df_fire_filtered.iloc[-1][metric_total]

    # 변화율 계산 (평균 기준)
    기준값 = avg_val
    if abs(기준값) < 0.1:
        change_display = f"{last_val - 기준값:+.0f}"
    else:
        rate = ((last_val - 기준값) / abs(기준값)) * 100
        change_display = f"{rate:+.1f}%"

    # 카드 UI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("평균", f"{avg_val:.0f}")
    col2.metric("최댓값", f"{max_row[metric_total]:.0f}", f"{int(max_row['구분'])}년")
    col3.metric("최솟값", f"{min_row[metric_total]:.0f}", f"{int(min_row['구분'])}년")
    col4.metric("변화율(평균 대비)", change_display)

    # 그래프 (전국)
    fig_fire_total = px.bar(df_fire_filtered, x="구분", y=metric_total, text=metric_total,
                            title=f"전국 {metric_total} 추이 (막대+선)")
    fig_fire_total.add_scatter(x=df_fire_filtered["구분"], y=df_fire_filtered[metric_total],
                               mode="lines+markers", name="추이선", line=dict(color="red"))
    fig_fire_total.update_traces(texttemplate='%{text:.0f}', textposition='outside', selector=dict(type='bar'))
    fig_set_font(fig_fire_total)
    st.plotly_chart(fig_fire_total, use_container_width=True)

    st.markdown("---")

    # 지역별 데이터 불러오기
    df_fire_region = load_csv("10년간 지역별 산불발생 현황.csv")
    df_fire_region.columns = [c.strip() for c in df_fire_region.columns]
    for col in df_fire_region.columns[1:]:
        df_fire_region[col] = df_fire_region[col].replace({',':''}, regex=True).astype(float)

    # 지역 선택
    selected_region = st.selectbox("지역 선택", df_fire_region['구분'].tolist())

    # 선택한 지역의 모든 지표 데이터
    df_region_filtered = df_fire_region[df_fire_region['구분'] == selected_region].drop(columns=['구분'])
    df_long = df_region_filtered.melt(var_name="지표", value_name="값")

    # 통계 계산 (지역별)
    avg_val_r = df_long['값'].mean()
    max_row_r = df_long.loc[df_long['값'].idxmax()]
    min_row_r = df_long.loc[df_long['값'].idxmin()]
    last_val_r = df_long.iloc[-1]['값']

    # 변화율 계산 (평균 기준)
    기준값_r = avg_val_r
    if abs(기준값_r) < 0.1:
        change_display_r = f"{last_val_r - 기준값_r:+.0f}"
    else:
        rate_r = ((last_val_r - 기준값_r) / abs(기준값_r)) * 100
        change_display_r = f"{rate_r:+.1f}%"

    # 카드 UI (지역별)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("평균", f"{avg_val_r:.0f}")
    col2.metric("최댓값", f"{max_row_r['값']:.0f}", max_row_r['지표'])
    col3.metric("최솟값", f"{min_row_r['값']:.0f}", min_row_r['지표'])
    col4.metric("변화율(평균 대비)", change_display_r)

    # 그래프 (지역별)
    fig_bar_all = px.bar(
        df_long,
        x="지표",
        y="값",
        text="값",
        color="값",
        color_continuous_scale="Reds",
        title=f"{selected_region} — 모든 지표 비교"
    )
    fig_bar_all.update_xaxes(tickangle=45)
    fig_bar_all.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig_bar_all.update_layout(
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        yaxis_title="값",
        xaxis_title="지표"
    )
    fig_set_font(fig_bar_all)
    st.plotly_chart(fig_bar_all, use_container_width=True)
    # --- 사용자: 해수면 ---
with tabs[3]:
    st.subheader("해수면 온도 편차 및 해양 변화")
    df_sea = load_csv("지표및해양에8월달평균기온지표.csv")
    df_sea['Year'] = df_sea['Year'].astype(int)
    df_sea['Anomaly'] = df_sea['Anomaly'].astype(float)

    # 기간 및 이동평균 윈도우 선택
    yrs_sea = st.slider(
        "기간",
        int(df_sea['Year'].min()), int(df_sea['Year'].max()),
        (int(df_sea['Year'].min()), int(df_sea['Year'].max()))
    )
    window = st.slider("이동평균 윈도우", 1, 10, 5)

    # 데이터 필터링 및 이동평균 계산
    df_sea_filtered = df_sea[(df_sea['Year'] >= yrs_sea[0]) & (df_sea['Year'] <= yrs_sea[1])]
    df_sea_filtered["이동평균"] = df_sea_filtered["Anomaly"].rolling(window).mean()

    # 통계 계산
    avg_val = df_sea_filtered['Anomaly'].mean()
    max_row = df_sea_filtered.loc[df_sea_filtered['Anomaly'].idxmax()]
    min_row = df_sea_filtered.loc[df_sea_filtered['Anomaly'].idxmin()]
    last_val = df_sea_filtered.iloc[-1]['Anomaly']

    # 변화율 계산 (평균 기준)
    기준값 = avg_val
    if abs(기준값) < 0.1:
        change_display = f"{last_val - 기준값:+.2f}°C"
    else:
        rate = ((last_val - 기준값) / abs(기준값)) * 100
        change_display = f"{rate:+.1f}%"

    # 카드 UI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("평균", f"{avg_val:.2f}°C")
    col2.metric("최댓값", f"{max_row['Anomaly']:.2f}°C", f"{int(max_row['Year'])}년")
    col3.metric("최솟값", f"{min_row['Anomaly']:.2f}°C", f"{int(min_row['Year'])}년")
    col4.metric("변화율(평균 대비)", change_display)

    # 그래프
    fig_sea = px.line(
        df_sea_filtered, x="Year", y=["Anomaly", "이동평균"], markers=True,
        labels={"value": "해수면 온도 편차 (°C)", "variable": "지표"}
    )
    fig_sea.add_hline(y=avg_val, line_dash="dot", line_color="green", annotation_text=f"평균 {avg_val:.2f}°C")
    fig_sea.add_scatter(
        x=[max_row['Year']], y=[max_row['Anomaly']],
        mode="markers+text", text=[f"최댓값 {max_row['Anomaly']:.2f}°C"], textposition="top center"
    )
    fig_sea.add_scatter(
        x=[min_row['Year']], y=[min_row['Anomaly']],
        mode="markers+text", text=[f"최솟값 {min_row['Anomaly']:.2f}°C"], textposition="bottom center"
    )
    fig_set_font(fig_sea)
    st.plotly_chart(fig_sea, use_container_width=True)

# --- 사용자: 멸종위기종 ---
with tabs[4]:
    st.subheader("분류군별 멸종위기종 종 수")
    
    # 데이터 불러오기
    df_species = load_csv("환경부 국립생물자원관_한국의 멸종위기종_20241231..csv")
    df_species['분류군'] = df_species['분류군'].str.strip()
    species_count = df_species['분류군'].value_counts().reset_index()
    species_count.columns = ['분류군', '종 수']

    # 분류군 선택
    selected_groups = st.multiselect(
        "분류군 선택",
        species_count['분류군'].tolist(),
        default=species_count['분류군'].tolist()
    )
    df_species_filtered = species_count[species_count['분류군'].isin(selected_groups)]

    # 통계 계산
    max_row = df_species_filtered.loc[df_species_filtered['종 수'].idxmax()]
    min_row = df_species_filtered.loc[df_species_filtered['종 수'].idxmin()]

    # 카드 UI (평균 제거)
    col1, col2 = st.columns(2)
    col1.metric("최댓값", f"{max_row['종 수']:.0f} 종", max_row['분류군'])
    col2.metric("최솟값", f"{min_row['종 수']:.0f} 종", min_row['분류군'])

    # 그래프
    fig_species = px.bar(
        df_species_filtered,
        x='분류군',
        y='종 수',
        text='종 수',
        color='종 수',
        color_continuous_scale="Reds"
    )
    fig_species.add_scatter(
        x=[max_row['분류군']], y=[max_row['종 수']],
        mode="markers+text", text=[f"최댓값 {max_row['종 수']}"], textposition="top center"
    )
    fig_species.add_scatter(
        x=[min_row['분류군']], y=[min_row['종 수']],
        mode="markers+text", text=[f"최솟값 {min_row['종 수']}"], textposition="bottom center"
    )
    fig_set_font(fig_species)
    st.plotly_chart(fig_species, use_container_width=True)
    