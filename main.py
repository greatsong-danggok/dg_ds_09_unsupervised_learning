import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")

# -----------------------------
# 데이터 불러오기
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df_raw = load_data()

# -----------------------------
# 전처리
# -----------------------------
df = df_raw.copy()

# 네 가지 속성에 필요한 원본 열이 없거나 비어있는 행 제거
required_cols = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
df = df.dropna(subset=required_cols)

# 첫 주 관객이 0인 영화 제외 (나눗셈 문제 방지)
df = df[df["first_week_audi"] > 0]

# 속성 계산
df["log_first_scrn"] = np.log10(df["first_scrn"].clip(lower=1))
df["log_total_audi"] = np.log10(df["total_audi"].clip(lower=1))
df["days_in_top10_feat"] = df["days_in_top10"]
df["longrun_index"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20)

# 계산 후 혹시 생길 수 있는 결측치 제거
feature_cols_all = ["log_first_scrn", "log_total_audi", "days_in_top10_feat", "longrun_index"]
df = df.dropna(subset=feature_cols_all)

total_count = len(df_raw)
used_count = len(df)

# -----------------------------
# 속성 선택 UI
# -----------------------------
st.subheader("1. 군집에 사용할 속성 선택")

feature_labels = {
    "log_first_scrn": "스크린 수 (로그)",
    "log_total_audi": "누적 관객 (로그)",
    "days_in_top10_feat": "10위권 일수",
    "longrun_index": "롱런 지수",
}

selected_features = st.multiselect(
    "두 개 이상 선택하세요 (기본값: 네 개 모두)",
    options=list(feature_labels.keys()),
    default=list(feature_labels.keys()),
    format_func=lambda x: feature_labels[x],
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해야 합니다.")
    st.stop()

st.info(f"전체 편수: {total_count}편 · 군집에 사용된 편수: {used_count}편")

# -----------------------------
# 묶음 수 선택 UI
# -----------------------------
st.subheader("2. 묶음 수 선택")

n_clusters = st.slider("나눌 묶음 수를 고르세요", min_value=2, max_value=7, value=3, step=1)

# 묶음 기호 (최대 7개까지)
CLUSTER_SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]
cluster_order_labels = CLUSTER_SYMBOLS[:n_clusters]

# -----------------------------
# 표준화 + KMeans
# -----------------------------
X = df[selected_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 큰 순서로 군집 재정렬 -> 기호 부여
cluster_order = (
    df.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
label_map = {cluster_id: CLUSTER_SYMBOLS[i] for i, cluster_id in enumerate(cluster_order)}
df["cluster_label"] = df["cluster_raw"].map(label_map)

# -----------------------------
# 2D 산점도
# -----------------------------
st.subheader("3. 2차원 산점도")

col1, col2 = st.columns(2)
with col1:
    x_axis_2d = st.selectbox(
        "가로축 속성",
        options=selected_features,
        format_func=lambda x: feature_labels[x],
        key="x2d",
    )
with col2:
    remaining_for_y = [f for f in selected_features if f != x_axis_2d] or selected_features
    y_axis_2d = st.selectbox(
        "세로축 속성",
        options=remaining_for_y,
        format_func=lambda x: feature_labels[x],
        key="y2d",
    )

fig2d = px.scatter(
    df,
    x=x_axis_2d,
    y=y_axis_2d,
    color="cluster_label",
    category_orders={"cluster_label": cluster_order_labels},
    hover_name="movieNm",
    labels={x_axis_2d: feature_labels[x_axis_2d], y_axis_2d: feature_labels[y_axis_2d]},
    title="영화 유형 2차원 분포",
)
st.plotly_chart(fig2d, use_container_width=True)

# -----------------------------
# 3D 산점도
# -----------------------------
st.subheader("4. 3차원 산점도")

if len(selected_features) < 3:
    st.info("3차원 산점도를 그리려면 속성을 세 개 이상 선택해야 합니다.")
else:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_axis_3d = st.selectbox(
            "X축 속성",
            options=selected_features,
            format_func=lambda x: feature_labels[x],
            key="x3d",
        )
    with col4:
        y_options = [f for f in selected_features if f != x_axis_3d]
        y_axis_3d = st.selectbox(
            "Y축 속성",
            options=y_options,
            format_func=lambda x: feature_labels[x],
            key="y3d",
        )
    with col5:
        z_options = [f for f in selected_features if f not in (x_axis_3d, y_axis_3d)]
        z_axis_3d = st.selectbox(
            "Z축 속성",
            options=z_options,
            format_func=lambda x: feature_labels[x],
            key="z3d",
        )

    fig3d = px.scatter_3d(
        df,
        x=x_axis_3d,
        y=y_axis_3d,
        z=z_axis_3d,
        color="cluster_label",
        category_orders={"cluster_label": cluster_order_labels},
        hover_name="movieNm",
        labels={
            x_axis_3d: feature_labels[x_axis_3d],
            y_axis_3d: feature_labels[y_axis_3d],
            z_axis_3d: feature_labels[z_axis_3d],
        },
        title="영화 유형 3차원 분포",
    )
    st.plotly_chart(fig3d, use_container_width=True)

# -----------------------------
# 군집별 통계표
# -----------------------------
st.subheader("5. 묶음별 편수와 평균값 (원래 단위)")

summary_rows = []
for label in cluster_order_labels:
    sub = df[df["cluster_label"] == label]
    row = {
        "묶음": label,
        "편수": len(sub),
        "스크린 수 평균": sub["first_scrn"].mean(),
        "누적 관객 평균": sub["total_audi"].mean(),
        "10위권 일수 평균": sub["days_in_top10"].mean(),
        "롱런 지수 평균": sub["longrun_index"].mean(),
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("묶음")
st.dataframe(summary_df.style.format({
    "스크린 수 평균": "{:.1f}",
    "누적 관객 평균": "{:,.0f}",
    "10위권 일수 평균": "{:.1f}",
    "롱런 지수 평균": "{:.2f}",
}))

# -----------------------------
# 묶음별 대표 영화 (누적 관객 상위 5편)
# -----------------------------
st.subheader("6. 묶음별 누적 관객 상위 5편")

for label in cluster_order_labels:
    sub = df[df["cluster_label"] == label].sort_values("total_audi", ascending=False)
    top5 = sub.head(5)["movieNm"].tolist()
    st.markdown(f"**{label} 묶음**: " + ", ".join(top5))

# -----------------------------
# 엘보우 방법 (관성 값 변화)
# -----------------------------
st.subheader("7. 묶음 수에 따른 관성 값 변화 (엘보우 방법)")

k_range = list(range(1, 8))
inertia_list = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia_list.append(km.inertia_)

elbow_df = pd.DataFrame({"묶음 수": k_range, "관성 값": inertia_list})

fig_elbow = go.Figure()
fig_elbow.add_trace(
    go.Scatter(
        x=elbow_df["묶음 수"],
        y=elbow_df["관성 값"],
        mode="lines+markers",
        name="관성 값",
    )
)
# 현재 선택한 묶음 수 위치에 세로선 추가
fig_elbow.add_vline(
    x=n_clusters,
    line_dash="dash",
    line_color="red",
    annotation_text=f"선택한 묶음 수: {n_clusters}",
    annotation_position="top",
)
fig_elbow.update_layout(
    xaxis_title="묶음 수",
    yaxis_title="관성 값 (군집 내 거리 제곱 합)",
    title="묶음 수에 따른 관성 값 변화",
)
st.plotly_chart(fig_elbow, use_container_width=True)

# -----------------------------
# 관성 값 감소폭 표
# -----------------------------
st.subheader("8. 묶음 수별 관성 값과 감소폭")

decrease_list = [None]  # 첫 줄은 비교 대상 없음
for i in range(1, len(inertia_list)):
    decrease = inertia_list[i - 1] - inertia_list[i]
    decrease_list.append(decrease)

inertia_table = pd.DataFrame({
    "묶음 수": k_range,
    "관성 값": inertia_list,
    "직전 대비 감소량": decrease_list,
})

st.dataframe(
    inertia_table.style.format({
        "관성 값": "{:.2f}",
        "직전 대비 감소량": lambda v: "" if pd.isna(v) else f"{v:.2f}",
    })
)

# -----------------------------
# 실루엣 점수
# -----------------------------
st.subheader("9. 현재 선택한 묶음 수의 실루엣 점수")

if n_clusters >= 2:
    sil_score = silhouette_score(X_scaled, df["cluster_raw"])
    st.info(f"묶음 수 {n_clusters}개일 때 실루엣 점수: {sil_score:.3f} (−1~1 사이, 1에 가까울수록 묶음이 뚜렷함)")
else:
    st.info("실루엣 점수는 묶음 수가 2개 이상일 때 계산할 수 있습니다.")
