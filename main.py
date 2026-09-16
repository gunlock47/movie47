import math

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()
total_count = len(df)

# ---------------------------------------------------------------
# 네 가지 속성 계산에 필요한 값이 없거나 첫 주 관객이 0인 영화 제외
# ---------------------------------------------------------------
valid_mask = (
    df["first_scrn"].notna()
    & (df["first_scrn"] > 0)
    & df["total_audi"].notna()
    & (df["total_audi"] > 0)
    & df["first_week_audi"].notna()
    & (df["first_week_audi"] > 0)
    & df["days_in_top10"].notna()
)
data = df[valid_mask].copy()

# 스크린 수, 누적 관객: 상용로그
data["log_scrn"] = data["first_scrn"].apply(lambda x: math.log10(x))
data["log_audi"] = data["total_audi"].apply(lambda x: math.log10(x))
# 10위권 일수: 그대로
data["top10_days"] = data["days_in_top10"]
# 롱런 지수: 누적 관객 / 첫 주 관객, 20 초과 시 20으로 절단
data["long_run"] = (data["total_audi"] / data["first_week_audi"]).clip(upper=20)

used_count = len(data)
st.write(f"전체 {total_count}편 중 유형 묶기에 사용한 영화는 {used_count}편입니다.")

# ---------------------------------------------------------------
# 군집화에 사용할 속성 선택 (2개 이상, 기본은 네 개 다)
# ---------------------------------------------------------------
feature_options = {
    "스크린 수 (log10)": "log_scrn",
    "누적 관객 수 (log10)": "log_audi",
    "10위권 유지 일수": "top10_days",
    "롱런 지수 (누적/첫주, 최대 20)": "long_run",
}

selected_labels = st.multiselect(
    "묶는 데 사용할 속성을 선택하세요 (2개 이상)",
    options=list(feature_options.keys()),
    default=list(feature_options.keys()),
)

if len(selected_labels) < 2:
    st.warning("속성을 2개 이상 선택해야 유형을 나눌 수 있습니다.")
    st.stop()

selected_cols = [feature_options[label] for label in selected_labels]

# ---------------------------------------------------------------
# 표준화 후 k-평균으로 3개 묶음으로 분리 (난수 고정)
# ---------------------------------------------------------------
X = data[selected_cols]
X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data["cluster"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 큰 묶음부터 ㉮, ㉯, ㉰
cluster_order = (
    data.groupby("cluster")["total_audi"].mean().sort_values(ascending=False).index.tolist()
)
symbols = ["㉮", "㉯", "㉰"]
symbol_map = {cluster_id: sym for cluster_id, sym in zip(cluster_order, symbols)}
data["묶음"] = data["cluster"].map(symbol_map)

color_map = {"㉮": "#EF553B", "㉯": "#636EFA", "㉰": "#00CC96"}

st.divider()

# ---------------------------------------------------------------
# 2차원 산점도
# ---------------------------------------------------------------
st.subheader("2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_label_2d = st.selectbox("가로축 속성", selected_labels, index=0, key="x2d")
with col2:
    default_y_idx = 1 if len(selected_labels) > 1 else 0
    y_label_2d = st.selectbox("세로축 속성", selected_labels, index=default_y_idx, key="y2d")

fig2d = px.scatter(
    data,
    x=feature_options[x_label_2d],
    y=feature_options[y_label_2d],
    color="묶음",
    color_discrete_map=color_map,
    hover_name="movieNm",
    labels={feature_options[x_label_2d]: x_label_2d, feature_options[y_label_2d]: y_label_2d},
    category_orders={"묶음": symbols},
)
st.plotly_chart(fig2d, use_container_width=True)

# ---------------------------------------------------------------
# 3차원 산점도
# ---------------------------------------------------------------
st.subheader("3차원 산점도")
if len(selected_labels) < 3:
    st.info("3차원 산점도를 보려면 속성을 3개 이상 선택하세요.")
else:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_label_3d = st.selectbox("x축 속성", selected_labels, index=0, key="x3d")
    with col4:
        y_label_3d = st.selectbox("y축 속성", selected_labels, index=1, key="y3d")
    with col5:
        z_label_3d = st.selectbox("z축 속성", selected_labels, index=2, key="z3d")

    fig3d = px.scatter_3d(
        data,
        x=feature_options[x_label_3d],
        y=feature_options[y_label_3d],
        z=feature_options[z_label_3d],
        color="묶음",
        color_discrete_map=color_map,
        hover_name="movieNm",
        labels={
            feature_options[x_label_3d]: x_label_3d,
            feature_options[y_label_3d]: y_label_3d,
            feature_options[z_label_3d]: z_label_3d,
        },
        category_orders={"묶음": symbols},
    )
    fig3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig3d, use_container_width=True)

st.divider()

# ---------------------------------------------------------------
# 묶음별 편수와 네 속성 평균(원래 단위)
# ---------------------------------------------------------------
st.subheader("묶음별 편수와 속성 평균 (원래 단위)")
summary = (
    data.groupby("묶음")
    .agg(
        편수=("movieNm", "count"),
        평균_스크린수=("first_scrn", "mean"),
        평균_누적관객=("total_audi", "mean"),
        평균_10위권일수=("days_in_top10", "mean"),
        평균_롱런지수=("long_run", "mean"),
    )
    .reindex(symbols)
    .round(1)
)
st.dataframe(summary, use_container_width=True)

# ---------------------------------------------------------------
# 묶음별 누적 관객 상위 5편
# ---------------------------------------------------------------
st.subheader("묶음별 누적 관객 상위 5편")
cols = st.columns(3)
for sym, col in zip(symbols, cols):
    with col:
        st.markdown(f"**{sym} 묶음**")
        top5 = (
            data[data["묶음"] == sym]
            .sort_values("total_audi", ascending=False)
            .head(5)["movieNm"]
            .tolist()
        )
        for rank, name in enumerate(top5, start=1):
            st.write(f"{rank}. {name}")
