import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Streamlit 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

st.title("품목별 수치 변수 간 상관관계 분석")

# 1. 파일 경로 설정 (streamlit_app.py와 동일하게)
price_file = 'region_price.xlsx'
weather_file = 'region_weather.csv'
cold_file = 'mon_cold.xlsx'
hot_file = 'mon_hot.xlsx'
wind_file = 'mon_wind.xlsx'

# 2. 데이터 로드 및 전처리 함수 (streamlit_app.py와 동일하게 복사)
@st.cache_data # 데이터 로드 및 전처리는 한 번만 실행되도록 캐싱
def load_and_preprocess_data():
    df_price = pd.read_excel(price_file)
    df_weather = pd.read_csv(weather_file)
    df_cold = pd.read_excel(cold_file)
    df_hot = pd.read_excel(hot_file)
    df_wind = pd.read_excel(wind_file)

    # 날짜 컬럼을 datetime 형식으로 변환
    df_price['날짜'] = pd.to_datetime(df_price['날짜'])
    df_weather['날짜'] = pd.to_datetime(df_weather['날짜'])

    # 가격 데이터의 '평균가격'과 '총거래물량' 컬럼을 숫자형으로 변환 (오류 발생 시 NaN으로 처리)
    df_price['평균가격'] = pd.to_numeric(df_price['평균가격'], errors='coerce')
    df_price['총거래물량'] = pd.to_numeric(df_price['총거래물량'], errors='coerce')

    # 날씨 데이터 컬럼명 정리
    weather_cols = ['지역', '날짜', '평균기온(°C)', '월합강수량(00~24h만)(mm)', '평균풍속(m/s)', '최심적설(cm)']
    if len(df_weather.columns) == len(weather_cols):
        df_weather.columns = weather_cols

    # 이벤트 데이터 전처리 함수
    def preprocess_event_df(df, event_name):
        df_melted = df.melt(id_vars=[df.columns[0]], var_name="날짜", value_name=f"{event_name}_발생")
        df_melted.rename(columns={df.columns[0]: "지역"}, inplace=True)
        df_melted["날짜"] = pd.to_datetime(df_melted["날짜"], format="%Y-%m")
        df_melted["날짜"] = df_melted["날짜"].dt.to_period('M')
        return df_melted

    df_cold_processed = preprocess_event_df(df_cold, "한파")
    df_hot_processed = preprocess_event_df(df_hot, "폭염")
    df_wind_processed = preprocess_event_df(df_wind, "태풍")

    # 날짜를 월 단위로 통일
    df_weather['날짜'] = df_weather['날짜'].dt.to_period('M')
    df_price['날짜'] = df_price['날짜'].dt.to_period('M')

    # 가격 데이터와 날씨 데이터를 병합
    merged_df = pd.merge(df_price, df_weather, left_on=['지역', '날짜'], right_on=['지역', '날짜'], how='left')

    # 이벤트 데이터 병합
    merged_df = pd.merge(merged_df, df_cold_processed, on=['지역', '날짜'], how='left')
    merged_df = pd.merge(merged_df, df_hot_processed, on=['지역', '날짜'], how='left')
    merged_df = pd.merge(merged_df, df_wind_processed, on=['지역', '날짜'], how='left')

    # 최심적설, 한파, 폭염, 태풍의 NaN 값은 이벤트가 없었던 것으로 간주하여 0으로 채움
    merged_df['최심적설(cm)'] = merged_df['최심적설(cm)'].fillna(0)
    merged_df['한파_발생'] = merged_df['한파_발생'].fillna(0)
    merged_df['폭염_발생'] = merged_df['폭염_발생'].fillna(0)
    merged_df['태풍_발생'] = merged_df['태풍_발생'].fillna(0)

    # 모든 숫자형 컬럼을 다시 숫자형으로 변환하고, 변환할 수 없는 값(NaN)이 있는 행 제거
    numeric_cols = ['평균가격', '총거래물량', '평균기온(°C)', '월합강수량(00~24h만)(mm)', '평균풍속(m/s)', '최심적설(cm)']
    for col in numeric_cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
    merged_df.dropna(subset=numeric_cols, inplace=True)

    # 이상치 탐지 및 제거 (평균가격, 총거래물량)
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    columns_to_check = ['평균가격', '총거래물량']
    for col in columns_to_check:
        if col in merged_df.columns:
            merged_df = remove_outliers_iqr(merged_df, col)

    return merged_df

merged_df = load_and_preprocess_data()

# 상관관계를 보고 싶은 수치형 열
num_cols = ['평균가격', '총거래물량', '평균기온(°C)', 
            '월합강수량(00~24h만)(mm)', '평균풍속(m/s)', '최심적설(cm)', 
            '한파_발생', '폭염_발생', '태풍_발생']

# 품목 선택 드롭다운
selected_item = st.selectbox(
    "상관관계를 볼 품목을 선택하세요:",
    merged_df['품목'].unique()
)

# 선택된 품목에 대한 데이터 필터링
subset = merged_df[merged_df['품목'] == selected_item]

# 수치형 변수만 선택하여 상관관계 행렬 계산
# 모든 num_cols가 subset에 있는지 확인하고 없는 컬럼은 제외
actual_num_cols = [col for col in num_cols if col in subset.columns]

if not subset[actual_num_cols].empty:
    corr_matrix = subset[actual_num_cols].corr()

    # 히트맵 그리기
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"[{selected_item}] 수치 변수 간 상관관계 히트맵")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.write(f"{selected_item}에 대한 상관관계 분석을 위한 데이터가 부족합니다.")