# main.py — 선수 유형 나누기: 고른 능력치로 묶고, 묶음 수도 바꿔 가며 정한다
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="선수 유형 나누기", page_icon="⚽", layout="wide")
st.title("⚽ 선수 유형 나누기")
st.caption("포지션을 사용하지 않고, 고른 능력치가 비슷한 선수끼리 묶습니다.")

DATA = ("https://raw.githubusercontent.com/greatsong/modudata/"
        "main/data/eafc25_top100.csv")
능력치 = {"pace": "속도", "shooting": "슈팅", "passing": "패스",
         "dribbling": "드리블", "defending": "수비", "physic": "몸싸움"}
기호 = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳"]
공격 = {"ST", "CF", "LW", "RW"}
미드 = {"CAM", "CM", "CDM", "LM", "RM"}


def 포지션그룹(칸):
    앞 = 칸.split(",")[0].strip().upper()
    if 앞 in 공격:
        return "공격수"
    if 앞 in 미드:
        return "미드필더"
    return "수비수"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA, encoding="utf-8")
    df = df.dropna(subset=list(능력치) + ["name_ko", "positions"]).reset_index(drop=True)
    df = df.rename(columns=능력치)
    df["포지션"] = df["positions"].map(포지션그룹)
    return df


df = load_data()
항목 = list(능력치.values())

고른항목 = st.multiselect("묶는 데 사용할 능력치 (둘 이상)", 항목, default=항목)
if len(고른항목) < 2:
    st.warning("능력치를 둘 이상 골라 주세요.")
    st.stop()
k = st.slider("몇 묶음으로 나눌까요", 2, 6, 3)

Xs = StandardScaler().fit_transform(df[고른항목])          # 항목마다 변화 폭이 다르므로 표준화한다



def 나누기(개수):
    """표준화한 값을 k-평균으로 나눈다. 난수를 고정해 다시 실행해도 결과가 같다."""
    return KMeans(n_clusters=개수, n_init=10, random_state=42).fit(Xs)


번호 = 나누기(k).labels_
순서 = pd.Series(df["슈팅"].to_numpy()).groupby(번호).mean().sort_values(ascending=False).index
자리 = {g: i for i, g in enumerate(순서)}
df["묶음"] = [기호[자리[g]] for g in 번호]

st.info(f"선수 {len(df):,}명을 {k}묶음으로 나눴습니다. 사용한 능력치: {', '.join(고른항목)}")

st.subheader("묶음 지도 · 2차원")
c1, c2 = st.columns(2)
x축 = c1.selectbox("가로축", 고른항목, index=0)
y축 = c2.selectbox("세로축", 고른항목, index=min(1, len(고른항목) - 1))
fig = px.scatter(df, x=x축, y=y축, color="묶음", hover_name="name_ko",
                 category_orders={"묶음": 기호[:k]})
fig.update_traces(marker=dict(size=7, opacity=0.75))
fig.update_layout(height=460)
st.plotly_chart(fig, width="stretch")
st.caption("점 하나가 선수 한 명입니다. 축을 바꿔 보면 묶음이 나뉘는 축과 섞이는 축이 보입니다.")

st.subheader("묶음 지도 · 3차원")
if len(고른항목) < 3:
    st.info("능력치를 셋 이상 고르면 3차원 그림이 표시됩니다.")
else:
    d1, d2, d3 = st.columns(3)
    x3 = d1.selectbox("x축", 고른항목, index=0, key="x3")
    y3 = d2.selectbox("y축", 고른항목, index=1, key="y3")
    z3 = d3.selectbox("z축", 고른항목, index=2, key="z3")
    fig3 = px.scatter_3d(df, x=x3, y=y3, z=z3, color="묶음", hover_name="name_ko",
                         category_orders={"묶음": 기호[:k]})
    fig3.update_traces(marker=dict(size=2.5, opacity=0.8))
    fig3.update_layout(height=560, legend=dict(orientation="h"))
    st.plotly_chart(fig3, width="stretch")
    st.caption("마우스로 끌면 돌아갑니다. 점에 마우스를 올리면 선수 이름이 보입니다.")

st.subheader("묶음별 인원과 능력치 평균")
요약 = df.groupby("묶음")[항목].mean().round(1)
요약.insert(0, "인원", df.groupby("묶음").size())
st.dataframe(요약.reindex(기호[:k]), width="stretch")

st.subheader("묶음마다 종합 능력치가 높은 다섯 명")
열 = st.columns(k)
for i, 이름 in enumerate(기호[:k]):
    묶음 = df[df["묶음"] == 이름].sort_values("overall", ascending=False).head(5)
    열[i].markdown(f"**{이름} 묶음 ({int((df['묶음'] == 이름).sum()):,}명)**")
    열[i].dataframe(pd.DataFrame({"선수": 묶음["name_ko"].to_numpy(),
                                "종합": 묶음["overall"].to_numpy()}),
                   width="stretch", hide_index=True)

st.subheader("묶음과 포지션은 얼마나 겹치는가")
대조 = pd.crosstab(df["포지션"], df["묶음"]).reindex(columns=기호[:k], fill_value=0)
st.dataframe(대조, width="stretch")
st.caption("포지션은 묶는 데 사용하지 않았습니다. 두 구분이 겹치기는 하지만 일치하지는 않습니다.")

# 여기부터 ③ — 묶음 수를 몇으로 할지 정하는 화면
st.divider()
st.subheader("묶음 수는 몇이 좋을까")

합 = pd.DataFrame({"묶음 수": list(range(1, 8)),
                   "거리 제곱의 합": [round(float(나누기(개수).inertia_), 1) for 개수 in range(1, 8)]})
합["앞보다 줄어든 값"] = ["—" if pd.isna(v) else f"{v:,.1f}" for v in -합["거리 제곱의 합"].diff()]

꺾은선 = px.line(합, x="묶음 수", y="거리 제곱의 합", markers=True)
꺾은선.add_vline(x=k, line_dash="dash", annotation_text=f"지금 고른 묶음 수 {k}")
꺾은선.update_layout(height=380, xaxis_title="묶음 수(k)", yaxis_title="묶음 안 거리 제곱의 합")
st.plotly_chart(꺾은선, width="stretch")
st.caption("묶음 수를 늘리면 값은 반드시 줄어듭니다. 줄어드는 폭이 크게 꺾이는 자리를 찾습니다.")
st.dataframe(합, width="stretch", hide_index=True)

점수 = pd.DataFrame([{"묶음 수": kk,
                    "실루엣": round(float(silhouette_score(Xs, 나누기(kk).labels_)), 3)}
                   for kk in range(2, 8)])
st.plotly_chart(px.line(점수, x="묶음 수", y="실루엣", markers=True), width="stretch")
st.dataframe(점수, width="stretch", hide_index=True)
st.caption("실루엣 점수가 가장 높은 묶음 수와 사람이 이해하기 좋은 묶음 수가 늘 같지는 않습니다.")
