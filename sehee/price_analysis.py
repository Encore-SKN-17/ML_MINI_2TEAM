import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Streamlit 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

st.title("농작물 가격과 이상기후 분석")

# 1. 파일 경로 설정
price_file = 'region_price.xlsx'
weather_file = 'region_weather.csv'
cold_file = 'mon_cold.xlsx'
hot_file = 'mon_hot.xlsx'
wind_file = 'mon_wind.xlsx'

# 2. 데이터 로드 및 전처리 함수
@st.cache_data # 데이터 로드 및 전처리는 한 번만 실행되도록 캐싱
def load_and_preprocess_data():
    try:
        df_price = pd.read_excel(price_file)
        df_weather = pd.read_csv(weather_file)
        df_cold = pd.read_excel(cold_file)
        df_hot = pd.read_excel(hot_file)
        df_wind = pd.read_excel(wind_file)
    except FileNotFoundError:
        st.error("필요한 데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        st.stop()

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

# 월별 평균 가격 계산 (이벤트 전후 월 계산을 위해)
monthly_avg_price_national = merged_df.groupby(['품목', '날짜'])['평균가격'].mean().reset_index()
monthly_avg_price_national['날짜'] = monthly_avg_price_national['날짜'].dt.to_timestamp()

# 지역별 월별 평균 가격 계산
monthly_avg_price_regional = merged_df.groupby(['품목', '지역', '날짜'])['평균가격'].mean().reset_index()
monthly_avg_price_regional['날짜'] = monthly_avg_price_regional['날짜'].dt.to_timestamp()

# 3. 사이드바 UI
st.sidebar.header("분석 설정")

# 지역 선택
unique_regions = merged_df['지역'].unique()
selected_region = st.sidebar.selectbox(
    "지역 선택:",
    unique_regions
)

event_type = st.sidebar.radio(
    "날씨 이벤트 선택:",
    ("한파", "폭염", "태풍")
)

# 선택된 이벤트에 따른 컬럼명 매핑
event_col_map = {
    "한파": "한파_발생",
    "폭염": "폭염_발생",
    "태풍": "태풍_발생"
}
selected_event_col = event_col_map[event_type]

# 선택된 지역과 이벤트 유형에 해당하는 월 목록 가져오기
event_months_available = merged_df[(merged_df[selected_event_col] == 1) & 
                                     (merged_df['지역'] == selected_region)]['날짜'].unique()

# Period 객체를 'YYYY-MM' 형식의 문자열로 변환
event_months_str = [str(month) for month in event_months_available]

if not event_months_str:
    st.sidebar.write(f"{selected_region}에서 {event_type} 발생 데이터가 없습니다.")
    selected_event_month_str = None
else:
    selected_event_month_str = st.sidebar.selectbox(
        f"{selected_region}의 {event_type} 발생 월 선택:",
        event_months_str
    )

# 4. 시각화 로직
st.header(f"{selected_region} - {event_type} 발생 시 품목별 가격 변화")

def plot_event_impact_streamlit(event_col, event_name, selected_region, selected_event_month_str, df, monthly_avg_df_regional, monthly_avg_df_national):
    if selected_event_month_str is None:
        st.write("이벤트 발생 월을 선택해주세요.")
        return

    # 선택된 월 문자열을 Period('M') 객체로 변환
    selected_event_month_period = pd.Period(selected_event_month_str, freq='M')

    # 선택된 지역, 이벤트 월에 이벤트가 발생한 품목들 필터링
    event_data_for_month_region = df[(df[event_col] == 1) & 
                                     (df['날짜'] == selected_event_month_period) & 
                                     (df['지역'] == selected_region)].copy()
    
    if event_data_for_month_region.empty:
        st.write(f"선택하신 {selected_region}의 {selected_event_month_str}에 {event_name} 발생 데이터가 없습니다.")
        return

    # 해당 월에 이벤트가 발생한 고유 품목들
    items_that_experienced_event = event_data_for_month_region['품목'].unique()

    if len(items_that_experienced_event) == 0:
        st.write(f"선택하신 {selected_region}의 {selected_event_month_str}에 {event_name}을 경험한 품목이 없습니다.")
        return

    # 전국 평균 가격 그래프
    st.subheader(f"{selected_event_month_str} {event_name} 발생 시 품목별 전국 평균 가격 변화")
    fig_national, ax_national = plt.subplots(figsize=(10, 6))
    plot_data_national = pd.DataFrame()
    for item in items_that_experienced_event:
        event_month_ts = selected_event_month_period.to_timestamp()
        prev_month_ts = (selected_event_month_period - 1).to_timestamp()
        next_month_ts = (selected_event_month_period + 1).to_timestamp()

        temp_df_national = monthly_avg_df_national[(monthly_avg_df_national['품목'] == item) & 
                                                    (monthly_avg_df_national['날짜'].isin([prev_month_ts, event_month_ts, next_month_ts]))].copy()
        if not temp_df_national.empty:
            temp_df_national['event_occurrence'] = event_month_ts
            temp_df_national['relative_month'] = temp_df_national.apply(lambda row: 
                                                                        (row['날짜'].year - row['event_occurrence'].year) * 12 + 
                                                                        (row['날짜'].month - row['event_occurrence'].month),
                                                                        axis=1)
            plot_data_national = pd.concat([plot_data_national, temp_df_national])

    if not plot_data_national.empty:
        sns.lineplot(data=plot_data_national, x="relative_month", y="평균가격", hue="품목", ax=ax_national, ci=None)
        ax_national.set_title(f"전국 평균 가격 변화")
        ax_national.set_xlabel("상대적 월 (0: 이벤트 발생 월)")
        ax_national.set_ylabel("평균 가격")
        ax_national.set_xticks([-1, 0, 1])
        ax_national.set_xticklabels(["전월", "당월", "익월"])
        ax_national.grid(True)
        ax_national.legend(title="품목")
        st.pyplot(fig_national)
    else:
        st.write("전국 평균 가격 데이터를 찾을 수 없습니다.")

    # 지역별 가격 그래프 (하나의 그래프에 병합)
    st.subheader(f"{selected_region} - {selected_event_month_str} {event_name} 발생 시 품목별 지역 가격 변화")
    fig_regional, ax_regional = plt.subplots(figsize=(10, 6))
    plot_data_regional = pd.DataFrame()
    for item in items_that_experienced_event:
        event_month_ts = selected_event_month_period.to_timestamp()
        prev_month_ts = (selected_event_month_period - 1).to_timestamp()
        next_month_ts = (selected_event_month_period + 1).to_timestamp()

        temp_df_regional = monthly_avg_df_regional[(monthly_avg_df_regional['품목'] == item) & 
                                                     (monthly_avg_df_regional['지역'] == selected_region) & 
                                                     (monthly_avg_df_regional['날짜'].isin([prev_month_ts, event_month_ts, next_month_ts]))].copy()
        if not temp_df_regional.empty:
            temp_df_regional['event_occurrence'] = event_month_ts
            temp_df_regional['relative_month'] = temp_df_regional.apply(lambda row: 
                                                                        (row['날짜'].year - row['event_occurrence'].year) * 12 + 
                                                                        (row['날짜'].month - row['event_occurrence'].month),
                                                                        axis=1)
            plot_data_regional = pd.concat([plot_data_regional, temp_df_regional])

    if not plot_data_regional.empty:
        sns.lineplot(data=plot_data_regional, x="relative_month", y="평균가격", hue="품목", ax=ax_regional, ci=None)
        ax_regional.set_title(f"{selected_region} 지역 가격 변화")
        ax_regional.set_xlabel("상대적 월 (0: 이벤트 발생 월)")
        ax_regional.set_ylabel("평균 가격")
        ax_regional.set_xticks([-1, 0, 1])
        ax_regional.set_xticklabels(["전월", "당월", "익월"])
        ax_regional.grid(True)
        ax_regional.legend(title="품목")
        st.pyplot(fig_regional)
    else:
        st.write(f"{selected_region} 지역 가격 데이터를 찾을 수 없습니다.")

# 선택된 이벤트와 품목에 대한 그래프 그리기
plot_event_impact_streamlit(selected_event_col, event_type, selected_region, selected_event_month_str, merged_df, monthly_avg_price_regional, monthly_avg_price_national)
