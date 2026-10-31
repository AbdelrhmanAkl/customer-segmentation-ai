from html import escape
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Customer Segmentation AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "customer_segmentation_model.joblib"
DATA_PATH = BASE_DIR / "data" / "Mall_Customers.csv"

INK, MUTED, LINE, ACCENT = "#0B1020", "#64748B", "#E6E9F2", "#4F46E5"

# name -> (persona, color, goal, action, KPI, sample income, sample score)
SEGMENTS = {
    "High Income - High Spending": ("Champions", "#E11D48", "Protect and grow",
        "Loyalty programs and premium offers.", "Repeat purchase rate", 88, 82),
    "High Income - Low Spending": ("Untapped Potential", "#F59E0B", "Win more of their wallet",
        "Personalized campaigns. Look into what holds them back from buying.", "Conversion rate", 88, 17),
    "Low Income - High Spending": ("Eager Spenders", "#9333EA", "Reward without over-discounting",
        "Test affordable offers and targeted promotions.", "Promo response rate", 26, 79),
    "Low Income - Low Spending": ("Budget Conscious", "#64748B", "Serve efficiently",
        "Value-oriented campaigns.", "Cost per conversion", 26, 21),
    "Average Income - Average Spending": ("Core Shoppers", "#0D9488", "Keep engaged",
        "Broad engagement and retention campaigns.", "Retention rate", 55, 50),
}
FALLBACK = ["#7C3AED", "#EA580C", "#0891B2", "#65A30D", "#BE123C"]

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="st-"], .stApp, button, input, select, textarea {
  font-family:'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif; }
h1, h2, h3, .display { font-family:'Bricolage Grotesque','Manrope',sans-serif; }

.stApp, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1000px 500px at 0% -10%, rgba(79,70,229,.10), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(236,72,153,.08), transparent 60%),
    #F4F6FB;
  background-attachment: fixed; }
[data-testid="stMain"], [data-testid="stMainBlockContainer"], .main { background:transparent; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer,
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none !important; }
[data-testid="stHeaderActionElements"], a.anchor-link, .stMarkdown a[href^="#"] { display:none !important; }
.block-container { max-width:1240px; padding-top:1rem; padding-bottom:3rem; }

/* ---------- top bar ---------- */
.topbar { display:flex; justify-content:space-between; align-items:center; padding:.3rem 0 1rem 0; }
.brand { display:flex; align-items:center; gap:.65rem; font-weight:800; font-size:1.05rem; color:#0B1020; letter-spacing:-.01em; }
.status { display:flex; align-items:center; gap:.45rem; font-size:.82rem; font-weight:700; color:#047857;
  background:#ECFDF5; border:1px solid #A7F3D0; padding:.32rem .85rem; border-radius:999px; }
.status i { width:8px; height:8px; border-radius:50%; background:#10B981; display:inline-block;
  box-shadow:0 0 0 0 rgba(16,185,129,.6); animation:pulse 2.2s infinite; }
@keyframes pulse { 70% { box-shadow:0 0 0 8px rgba(16,185,129,0); } 100% { box-shadow:0 0 0 0 rgba(16,185,129,0); } }
@media (prefers-reduced-motion:reduce) { .status i { animation:none; } }

/* ---------- hero ---------- */
.hero { position:relative; overflow:hidden; border-radius:28px; padding:2.6rem 2.8rem 2.2rem 2.8rem; margin-bottom:1.4rem;
  color:#E8EAF6; background:
    radial-gradient(520px 260px at 92% 0%, rgba(236,72,153,.38), transparent 65%),
    radial-gradient(520px 300px at 70% 120%, rgba(79,70,229,.55), transparent 65%),
    #0B1020;
  box-shadow:0 24px 48px -24px rgba(11,16,32,.55); }
.hero-grid { display:grid; grid-template-columns:1.25fr 1fr; gap:2rem; align-items:center; }
.hero h1 { font-size:3rem; font-weight:800; letter-spacing:-.04em; line-height:1.04; margin:0 0 1rem 0; padding:0; color:#fff; }
.hero p { color:#B6BCD6; font-size:1.05rem; line-height:1.6; max-width:520px; margin:0; }
.chips { display:flex; flex-wrap:wrap; gap:.55rem; justify-content:flex-end; }
.chip { display:inline-flex; align-items:center; gap:.5rem; padding:.5rem .9rem; border-radius:999px; font-weight:700; font-size:.9rem;
  color:#fff; background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.16); backdrop-filter:blur(6px); }
.chip i { width:10px; height:10px; border-radius:50%; background:var(--seg); display:inline-block; }
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:1px; margin-top:2rem; border-radius:18px; overflow:hidden;
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.14); }
.kpi { background:rgba(11,16,32,.55); padding:1rem 1.2rem; }
.kpi span { display:block; font-size:.78rem; color:#9AA3C7; font-weight:600; }
.kpi b { display:block; font-family:'Bricolage Grotesque',sans-serif; font-size:1.8rem; font-weight:700; letter-spacing:-.03em; color:#fff; margin-top:.1rem; }
.model-note { margin-top:1rem; font-size:.8rem; color:#8E97BE; }
@media (max-width:900px) { .hero-grid { grid-template-columns:1fr; } .chips { justify-content:flex-start; }
  .hero h1 { font-size:2.1rem; } .hero { padding:1.6rem; } .kpis { grid-template-columns:repeat(2,1fr); } }

/* ---------- tabs ---------- */
[data-baseweb="tab-list"] { gap:.3rem; background:rgba(255,255,255,.75); border:1px solid #E6E9F2; padding:6px; border-radius:16px;
  width:fit-content; backdrop-filter:blur(8px); margin-bottom:.4rem; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none !important; }
button[role="tab"] { border-radius:11px; padding:.55rem 1.25rem; height:auto; background:transparent; transition:background .15s; }
button[role="tab"] p { font-weight:700; font-size:.95rem; color:#475569; }
button[role="tab"]:hover { background:#EEF0F8; }
button[role="tab"][aria-selected="true"] { background:#0B1020; }
button[role="tab"][aria-selected="true"] p { color:#fff; }
button[role="tab"]:focus { outline:none; box-shadow:none; }
button[role="tab"]:focus-visible { outline:2px solid #4F46E5; outline-offset:2px; }

/* ---------- cards & controls ---------- */
[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border:1px solid #E6E9F2 !important;
  border-radius:22px !important; box-shadow:0 1px 2px rgba(11,16,32,.04), 0 12px 32px -12px rgba(11,16,32,.10); }
[data-testid="stSlider"] label p, .stSelectbox label p, .stMultiSelect label p { font-weight:700; color:#0B1020; font-size:.92rem; }
span[data-baseweb="tag"] { background:#EEF0FF !important; color:#3730A3 !important; border-radius:8px !important; }
.stButton button { border-radius:12px; border:1px solid #E6E9F2; background:#fff; color:#0B1020; font-weight:700;
  font-size:.82rem; padding:.35rem .4rem; transition:all .15s; }
.stButton button:hover { border-color:#4F46E5; color:#4F46E5; background:#F5F5FF; }
.stDownloadButton button { background:#0B1020; color:#fff; border:0; border-radius:12px; font-weight:700; padding:.6rem 1rem; }
.stDownloadButton button:hover { background:#4F46E5; color:#fff; }

.panel-title { font-family:'Bricolage Grotesque',sans-serif; font-size:1.2rem; font-weight:700; color:#0B1020; letter-spacing:-.02em; margin:0 0 .15rem 0; }
.panel-sub { font-size:.88rem; color:#64748B; margin:0 0 1rem 0; }
.mini-label { font-size:.82rem; font-weight:700; color:#0B1020; margin:.9rem 0 .4rem 0; }

/* ---------- prediction result ---------- */
.result { margin-top:1.1rem; padding:1.3rem 1.3rem 1.1rem 1.3rem; border-radius:20px; color:#fff;
  background:linear-gradient(140deg, var(--seg), color-mix(in srgb, var(--seg) 62%, #0B1020)); }
.result small { font-size:.8rem; font-weight:600; opacity:.85; }
.result h3 { font-family:'Bricolage Grotesque',sans-serif; color:#fff; font-size:2rem; font-weight:800; letter-spacing:-.035em; margin:.1rem 0 0 0; padding:0; line-height:1.1; }
.result .seg { font-size:.88rem; opacity:.9; margin:.35rem 0 .9rem 0; }
.result .goal { display:inline-block; font-size:.78rem; font-weight:700; background:rgba(255,255,255,.2); padding:.25rem .65rem; border-radius:999px; margin-bottom:.8rem; }
.result .row { display:flex; justify-content:space-between; font-size:.88rem; padding:.45rem 0; border-top:1px solid rgba(255,255,255,.22); }
.result .act { margin-top:.7rem; font-size:.9rem; line-height:1.5; background:rgba(255,255,255,.16); padding:.75rem .9rem; border-radius:14px; }

/* ---------- segment cards ---------- */
.segs { display:grid; gap:1rem; grid-template-columns:repeat(5,1fr); align-items:stretch; }
@media (max-width:1100px) { .segs { grid-template-columns:repeat(3,1fr); } }
@media (max-width:720px) { .segs { grid-template-columns:1fr; } }
.seg-card { display:flex; flex-direction:column; background:#fff; border:1px solid #E6E9F2; border-radius:20px;
  padding:1.2rem 1.15rem 1.2rem 1.15rem; position:relative; overflow:hidden; box-shadow:0 1px 2px rgba(11,16,32,.04); }
.seg-card:before { content:""; position:absolute; inset:0 0 auto 0; height:5px; background:var(--seg); }
.seg-card h4 { font-family:'Bricolage Grotesque',sans-serif; font-size:1.15rem; font-weight:700; color:#0B1020; margin:.3rem 0 0 0; padding:0; letter-spacing:-.02em; }
.seg-card .orig { font-size:.8rem; color:#64748B; line-height:1.3; min-height:2.6em; margin-top:.2rem; }
.seg-card .share { font-family:'Bricolage Grotesque',sans-serif; font-size:2.4rem; font-weight:800; color:var(--seg); letter-spacing:-.04em; line-height:1.1; margin-top:.5rem; }
.seg-card .cnt { font-size:.84rem; color:#64748B; margin-bottom:.8rem; }
.seg-card .r { display:flex; justify-content:space-between; font-size:.87rem; color:#475569; padding:.42rem 0; border-top:1px solid #EEF0F5; }
.seg-card .r b { color:#0B1020; }
.seg-card .a { font-size:.85rem; color:#334155; line-height:1.5; margin-top:auto; padding-top:.8rem; border-top:1px solid #EEF0F5; }

/* ---------- customers ---------- */
.stats { display:flex; flex-wrap:wrap; gap:.6rem; margin:.9rem 0 .8rem 0; }
.stat { background:#F4F6FB; border:1px solid #E6E9F2; border-radius:12px; padding:.5rem .9rem; font-size:.85rem; color:#475569; }
.stat b { color:#0B1020; margin-left:.35rem; }

/* ---------- playbook ---------- */
table.play { width:100%; border-collapse:collapse; font-size:.92rem; }
table.play th { text-align:left; color:#64748B; font-weight:700; font-size:.82rem; padding:.7rem .8rem; border-bottom:1px solid #E6E9F2; }
table.play td { padding:1rem .8rem; border-bottom:1px solid #F0F1F6; color:#334155; vertical-align:top; line-height:1.45; }
table.play tr:last-child td { border-bottom:0; }
table.play tr:hover td { background:#FAFBFF; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:.55rem; background:var(--seg); }
.pname { font-weight:800; color:#0B1020; white-space:nowrap; }
.kpi-tag { display:inline-block; padding:.22rem .65rem; border-radius:999px; font-size:.8rem; font-weight:700;
  color:var(--seg); background:color-mix(in srgb, var(--seg) 12%, white); }
.note { margin-top:1.2rem; padding:1rem 1.2rem; border-radius:16px; background:#FFFBEB; border:1px solid #FDE68A;
  color:#78350F; font-size:.9rem; line-height:1.55; }
.foot { margin-top:2.2rem; text-align:center; color:#94A3B8; font-size:.82rem; line-height:1.7; }
</style>
""",
    unsafe_allow_html=True,
)


def block(markup: str) -> str:
    """Flatten markup so Streamlit never treats indented HTML as a code block."""
    return " ".join(line.strip() for line in markup.splitlines())


@st.cache_resource
def load_artifact(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()
try:
    artifact = load_artifact(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

pipeline, features, cluster_names = (
    artifact["pipeline"], artifact["features"], artifact["cluster_names"],
)

if not DATA_PATH.exists():
    st.error("Dataset not found. Place Mall_Customers.csv inside the project's data folder.")
    st.stop()
try:
    data = pd.read_csv(DATA_PATH)
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

required = ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
missing = [c for c in required if c not in data.columns]
if missing:
    st.error(f"Dataset is missing columns: {missing}")
    st.stop()

try:
    data["Cluster"] = pipeline.predict(data[features].copy()).astype(int)
    data["Segment"] = data["Cluster"].map(cluster_names)
except Exception as exc:
    st.error(f"Could not assign customer segments: {exc}")
    st.stop()

# -------------------- Derived values --------------------

fallback_iter = iter(FALLBACK)


def meta(name):
    if name in SEGMENTS:
        return SEGMENTS[name]
    return (name, next(fallback_iter, MUTED), "Review", "Validate with campaign data.",
            "Conversion rate", 60, 50)


META = {n: meta(n) for n in data["Segment"].unique()}
data["Persona"] = data["Segment"].map(lambda n: META[n][0])
COLORS = {META[n][0]: META[n][1] for n in META}

total = len(data)
profiles = (
    data.groupby(["Cluster", "Segment", "Persona"], as_index=False)
    .agg(
        Customers=("CustomerID", "count"),
        Income=("Annual Income (k$)", "mean"),
        Spending=("Spending Score (1-100)", "mean"),
        Age=("Age", "mean"),
    )
    .sort_values("Cluster")
)
profiles["Share"] = profiles["Customers"] / total * 100
order = profiles["Persona"].tolist()

avg_income = data["Annual Income (k$)"].mean()
avg_spend = data["Spending Score (1-100)"].mean()

# -------------------- Top bar + hero --------------------

chips_html = "".join(
    f'<span class="chip" style="--seg:{META[n][1]}"><i></i>{escape(META[n][0])}</span>'
    for n in profiles.sort_values("Customers", ascending=False)["Segment"]
)

st.markdown(
    block(
        f"""
<div class="topbar">
  <div class="brand">
    <svg width="32" height="32" viewBox="0 0 30 30"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#4F46E5"/><stop offset="1" stop-color="#EC4899"/></linearGradient></defs>
    <rect width="30" height="30" rx="9" fill="url(#g)"/><circle cx="10" cy="11" r="3" fill="#fff"/>
    <circle cx="20" cy="10" r="2.2" fill="#fff" fill-opacity=".8"/><circle cx="15" cy="20" r="3.4" fill="#fff" fill-opacity=".9"/></svg>
    Customer Segmentation AI
  </div>
  <div class="status"><i></i>Model ready</div>
</div>
<div class="hero">
  <div class="hero-grid">
    <div>
      <h1>Know which customers to reward, win back, or keep warm.</h1>
      <p>Describe any customer by income and spending. The model places them in a segment and suggests the marketing play that fits.</p>
    </div>
    <div class="chips">{chips_html}</div>
  </div>
  <div class="kpis">
    <div class="kpi"><span>Customers analysed</span><b>{total:,}</b></div>
    <div class="kpi"><span>Segments found</span><b>{data['Cluster'].nunique()}</b></div>
    <div class="kpi"><span>Average annual income</span><b>${avg_income:.1f}k</b></div>
    <div class="kpi"><span>Average spending score</span><b>{avg_spend:.1f}</b></div>
  </div>
  <div class="model-note">K-Means model with silhouette score {artifact['silhouette_score']:.2f}</div>
</div>
"""
    ),
    unsafe_allow_html=True,
)


def style_fig(fig, height):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=INK, size=13),
        legend=dict(title_text="", orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", font_size=13, bordercolor=LINE),
    )
    return fig


tab_predict, tab_segments, tab_customers, tab_playbook = st.tabs(
    ["Predictor", "Segments", "Customers", "Playbook"]
)

# -------------------- Predictor --------------------

st.session_state.setdefault("income", 60)
st.session_state.setdefault("score", 50)


def set_profile(inc, sc):
    st.session_state["income"] = int(inc)
    st.session_state["score"] = int(sc)


with tab_predict:
    left, right = st.columns([1, 1.75], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Customer profile</div>'
                '<div class="panel-sub">Pick a sample or move the sliders.</div>'
                '<div class="mini-label">Quick samples</div>',
                unsafe_allow_html=True,
            )
            samples = [(META[n][0], META[n][5], META[n][6]) for n in profiles["Segment"]]
            for i in range(0, len(samples), 3):
                cols = st.columns(3)
                for col, (label, inc, sc) in zip(cols, samples[i:i + 3]):
                    col.button(label, key=f"s_{label}", on_click=set_profile,
                               args=(inc, sc), use_container_width=True)

            income = st.slider("Annual income (k$)", 15, 137, key="income")
            score = st.slider("Spending score (1-100)", 1, 99, key="score")

            cust = pd.DataFrame([[income, score]], columns=features)
            cid = int(pipeline.predict(cust)[0])
            seg = cluster_names[cid]
            persona, color, goal, action = META.get(seg, meta(seg))[:4]
            row = profiles[profiles["Cluster"] == cid].iloc[0]

            st.markdown(
                block(
                    f"""
<div class="result" style="--seg:{color}">
  <small>Predicted segment</small>
  <h3>{escape(persona)}</h3>
  <div class="seg">{escape(str(seg))}, {row['Share']:.1f}% of customers</div>
  <span class="goal">Goal: {escape(goal)}</span>
  <div class="row"><span>Segment avg. income</span><b>${row['Income']:.1f}k</b></div>
  <div class="row"><span>Segment avg. spending</span><b>{row['Spending']:.1f}</b></div>
  <div class="act"><b>Suggested approach:</b> {escape(action)}</div>
</div>
"""
                ),
                unsafe_allow_html=True,
            )

    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Segment map</div>'
                '<div class="panel-sub">Each dot is a customer. The star is the profile you selected.</div>',
                unsafe_allow_html=True,
            )
            fig = go.Figure()
            for p in order:
                d = data[data["Persona"] == p]
                fig.add_trace(
                    go.Scatter(
                        x=d["Annual Income (k$)"], y=d["Spending Score (1-100)"],
                        mode="markers", name=p,
                        marker=dict(size=10, color=COLORS[p], opacity=0.85, line=dict(width=1.2, color="white")),
                        customdata=d[["CustomerID", "Gender", "Age"]],
                        hovertemplate=p + "<br>Customer %{customdata[0]}, %{customdata[1]}, age %{customdata[2]}"
                        "<br>Income %{x}k, spending %{y}<extra></extra>",
                    )
                )
                fig.add_annotation(
                    x=d["Annual Income (k$)"].mean(), y=d["Spending Score (1-100)"].mean(),
                    text=f"<b>{p}</b>", showarrow=False, font=dict(size=12, color=COLORS[p]),
                    bgcolor="rgba(255,255,255,.85)", bordercolor=COLORS[p], borderwidth=1, borderpad=4,
                    yshift=0,
                )
            fig.add_trace(
                go.Scatter(
                    x=[income], y=[score], mode="markers", name="Selected profile",
                    marker=dict(symbol="star", size=24, color=INK, line=dict(width=2, color="white")),
                    hovertemplate="Selected profile<br>Income %{x}k, spending %{y}<extra></extra>",
                )
            )
            style_fig(fig, 540)
            fig.update_xaxes(title="Annual income (k$)", gridcolor="#EEF0F5", zeroline=False, linecolor=LINE)
            fig.update_yaxes(title="Spending score", gridcolor="#EEF0F5", zeroline=False, linecolor=LINE)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# -------------------- Segments --------------------

with tab_segments:
    cards = []
    for _, r in profiles.iterrows():
        col, act = META[r["Segment"]][1], META[r["Segment"]][3]
        cards.append(
            f"""
<div class="seg-card" style="--seg:{col}">
  <h4>{escape(r['Persona'])}</h4>
  <div class="orig">{escape(str(r['Segment']))}</div>
  <div class="share">{r['Share']:.1f}%</div>
  <div class="cnt">{int(r['Customers'])} customers</div>
  <div class="r"><span>Avg. income</span><b>${r['Income']:.1f}k</b></div>
  <div class="r"><span>Avg. spending</span><b>{r['Spending']:.1f}</b></div>
  <div class="r"><span>Avg. age</span><b>{r['Age']:.0f}</b></div>
  <div class="a">{escape(act)}</div>
</div>"""
        )
    st.markdown(block(f'<div class="segs">{"".join(cards)}</div>'), unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Customer share</div>'
                '<div class="panel-sub">How the customer base splits across segments.</div>',
                unsafe_allow_html=True,
            )
            donut = go.Figure(
                go.Pie(
                    labels=profiles["Persona"], values=profiles["Customers"], hole=0.66, sort=False,
                    marker=dict(colors=[COLORS[p] for p in profiles["Persona"]], line=dict(color="white", width=3)),
                    textinfo="percent", textfont=dict(color="white", size=13),
                )
            )
            donut.add_annotation(text=f"<b>{total}</b><br>customers", showarrow=False, font=dict(size=16, color=INK))
            style_fig(donut, 400)
            st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})
    with c2:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-title">Income and spending by segment</div>'
                '<div class="panel-sub">Average annual income (k$) next to average spending score.</div>',
                unsafe_allow_html=True,
            )
            bars = go.Figure()
            bars.add_trace(go.Bar(x=profiles["Persona"], y=profiles["Income"].round(1), name="Avg. income (k$)",
                                  marker_color="#4F46E5", text=profiles["Income"].round(0).astype(int),
                                  textposition="outside", cliponaxis=False))
            bars.add_trace(go.Bar(x=profiles["Persona"], y=profiles["Spending"].round(1), name="Avg. spending score",
                                  marker_color="#EC4899", text=profiles["Spending"].round(0).astype(int),
                                  textposition="outside", cliponaxis=False))
            style_fig(bars, 400)
            bars.update_layout(barmode="group", bargap=0.35, margin=dict(l=10, r=10, t=24, b=10))
            bars.update_xaxes(tickfont=dict(size=11))
            bars.update_yaxes(gridcolor="#EEF0F5", zeroline=False)
            st.plotly_chart(bars, use_container_width=True, config={"displayModeBar": False})

# -------------------- Customers --------------------

with tab_customers:
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([2, 2, 2, 1.2], vertical_alignment="bottom")
        with f1:
            pick = st.multiselect("Segments", order, default=[], placeholder="All segments")
        with f2:
            genders = st.multiselect("Gender", sorted(data["Gender"].unique()), default=[], placeholder="All")
        with f3:
            lo, hi = int(data["Age"].min()), int(data["Age"].max())
            ages = st.slider("Age range", lo, hi, (lo, hi))

        view = data.copy()
        if pick:
            view = view[view["Persona"].isin(pick)]
        if genders:
            view = view[view["Gender"].isin(genders)]
        view = view[view["Age"].between(*ages)]

        with f4:
            st.download_button(
                "Download CSV",
                data=view.to_csv(index=False).encode("utf-8"),
                file_name="customer_segments.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if view.empty:
            st.info("No customers match these filters. Widen the age range or clear a filter.")
        else:
            st.markdown(
                block(
                    f"""
<div class="stats">
  <div class="stat">Showing<b>{len(view)} of {total}</b></div>
  <div class="stat">Avg. income<b>${view['Annual Income (k$)'].mean():.1f}k</b></div>
  <div class="stat">Avg. spending<b>{view['Spending Score (1-100)'].mean():.1f}</b></div>
  <div class="stat">Avg. age<b>{view['Age'].mean():.0f}</b></div>
</div>
"""
                ),
                unsafe_allow_html=True,
            )
            st.dataframe(
                view[["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)", "Persona", "Segment"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Annual Income (k$)": st.column_config.NumberColumn("Income", format="$%dk"),
                    "Spending Score (1-100)": st.column_config.ProgressColumn(
                        "Spending score", min_value=0, max_value=100, format="%d"
                    ),
                },
            )

# -------------------- Playbook --------------------

with tab_playbook:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title">Marketing playbook</div>'
            '<div class="panel-sub">A starting point for each segment and the metric to watch.</div>',
            unsafe_allow_html=True,
        )
        rows = ""
        for _, r in profiles.iterrows():
            persona, col, goal, act, kpi = META[r["Segment"]][:5]
            rows += (
                f'<tr style="--seg:{col}"><td class="pname"><span class="dot"></span>{escape(persona)}</td>'
                f'<td>{int(r["Customers"])} ({r["Share"]:.1f}%)</td><td>{escape(goal)}</td>'
                f'<td>{escape(act)}</td><td><span class="kpi-tag">{escape(kpi)}</span></td></tr>'
            )
        st.markdown(
            block(
                f'<table class="play"><tr><th>Segment</th><th>Customers</th><th>Goal</th>'
                f"<th>Suggested approach</th><th>Metric to track</th></tr>{rows}</table>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="note">These are suggested marketing hypotheses, not proven outcomes. '
        "Validate them with your own campaign and sales data before acting on them.</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="foot">Customer Segmentation AI. K-Means with five clusters, trained on annual income and spending score.<br>'
    "Built with Python, scikit-learn, Streamlit and Plotly.</div>",
    unsafe_allow_html=True,
)
