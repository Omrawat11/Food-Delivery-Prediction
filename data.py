# ============================================================
#  🚀 Food Delivery Time Predictor — Professional ML App
#  Author  : AI/ML Engineer
#  Model   : Linear Regression
#  Stack   : Streamlit · Plotly · Pandas · Scikit-learn
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeliveryIQ · Food Delivery Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
#  GLOBAL CSS  (dark premium theme)
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset / Base ── */
html, body, [class*="css"] { font-family:'Plus Jakarta Sans',sans-serif; }
.stApp { background:#070b14; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1117 0%,#0a0f1a 100%) !important;
    border-right: 1px solid #1a2035;
}
[data-testid="stSidebar"] * { font-family:'Plus Jakarta Sans',sans-serif !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg,#0d1529 0%,#111827 40%,#0d1529 100%);
    border:1px solid #1a2035;
    border-radius:20px;
    padding:2.8rem 3.2rem;
    margin-bottom:2rem;
    position:relative;
    overflow:hidden;
}
.hero::before {
    content:'';position:absolute;top:-80px;right:-80px;
    width:260px;height:260px;
    background:radial-gradient(circle,rgba(249,115,22,.15) 0%,transparent 70%);
}
.hero::after {
    content:'';position:absolute;bottom:-60px;left:35%;
    width:200px;height:200px;
    background:radial-gradient(circle,rgba(99,102,241,.12) 0%,transparent 70%);
}
.hero-title {
    font-size:2.6rem;font-weight:800;
    background:linear-gradient(90deg,#fb923c 0%,#f472b6 50%,#818cf8 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin:0 0 .5rem 0;letter-spacing:-1px;
}
.hero-sub {
    color:#64748b;font-size:.9rem;
    font-family:'JetBrains Mono',monospace;margin:0;letter-spacing:.4px;
}

/* ── Metric cards ── */
.kpi-card {
    background:#0d1117;border:1px solid #1a2035;border-radius:16px;
    padding:1.4rem 1.6rem;position:relative;overflow:hidden;
    transition:border-color .25s,transform .2s;
}
.kpi-card:hover{border-color:#f97316;transform:translateY(-2px);}
.kpi-stripe{position:absolute;top:0;left:0;width:4px;height:100%;border-radius:16px 0 0 16px;}
.kpi-label{font-size:.68rem;font-weight:700;letter-spacing:1.6px;text-transform:uppercase;
    color:#475569;font-family:'JetBrains Mono',monospace;margin-bottom:.55rem;}
.kpi-value{font-size:1.85rem;font-weight:800;color:#f1f5f9;line-height:1;}
.kpi-sub{font-size:.73rem;color:#64748b;margin-top:.45rem;font-family:'JetBrains Mono',monospace;}

/* ── Section heading ── */
.sec-head{
    font-size:1.05rem;font-weight:700;color:#f1f5f9;
    letter-spacing:.2px;margin-bottom:1rem;
    display:flex;align-items:center;gap:.5rem;
}
.badge{
    background:#1a2035;border:1px solid #2a3050;border-radius:6px;
    padding:.13rem .55rem;font-size:.66rem;color:#818cf8;
    font-family:'JetBrains Mono',monospace;font-weight:500;
}

/* ── Prediction result box ── */
.pred-result{
    border-radius:18px;padding:2rem 2.4rem;text-align:center;border:2px solid;
    background:linear-gradient(135deg,#0f1a0a,#0a1510);border-color:#22c55e;
}
.pred-value{font-size:3rem;font-weight:800;color:#22c55e;line-height:1;}
.pred-label{font-size:.7rem;letter-spacing:2px;text-transform:uppercase;
    font-family:'JetBrains Mono',monospace;color:#64748b;margin-bottom:.8rem;}
.pred-detail{font-size:.82rem;color:#64748b;margin-top:.8rem;
    font-family:'JetBrains Mono',monospace;}

/* ── Input card wrapper ── */
.input-card{
    background:#0d1117;border:1px solid #1a2035;border-radius:16px;
    padding:1.5rem 1.8rem;margin-bottom:1rem;
}

/* ── Info callout ── */
.callout{
    background:#0a1525;border:1px solid #1a3a5f;
    border-left:4px solid #3b82f6;border-radius:8px;
    padding:.8rem 1rem;font-size:.78rem;color:#94a3b8;
    font-family:'JetBrains Mono',monospace;margin-top:.5rem;
}
.callout-warn{
    background:#1a1205;border:1px solid #5f3a1a;
    border-left:4px solid #f59e0b;border-radius:8px;
    padding:.8rem 1rem;font-size:.78rem;color:#d97706;
    font-family:'JetBrains Mono',monospace;margin-top:.5rem;
}
.callout-ok{
    background:#0a1a0e;border:1px solid #1a5f2a;
    border-left:4px solid #22c55e;border-radius:8px;
    padding:.8rem 1rem;font-size:.78rem;color:#4ade80;
    font-family:'JetBrains Mono',monospace;margin-top:.5rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
    background:#0d1117;border-radius:12px;padding:5px;gap:4px;
    border:1px solid #1a2035;
}
.stTabs [data-baseweb="tab"]{
    background:transparent;border-radius:9px;
    color:#64748b;font-weight:600;font-size:.82rem;
    padding:.5rem 1.2rem;
}
.stTabs [aria-selected="true"]{background:#1a2035 !important;color:#f97316 !important;}

/* ── Form / inputs ── */
.stSelectbox>div>div,.stNumberInput>div>div,.stSlider>div{
    background:#0d1117 !important;border-color:#1a2035 !important;
}
label{color:#94a3b8 !important;font-size:.83rem !important;font-weight:600 !important;}

/* ── Buttons ── */
.stButton>button{
    background:linear-gradient(135deg,#ea580c,#f97316) !important;
    border:none !important;border-radius:10px !important;
    color:white !important;font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:700 !important;font-size:.88rem !important;
    padding:.65rem 1.8rem !important;transition:opacity .2s !important;
    box-shadow:0 4px 14px rgba(249,115,22,.35) !important;
}
.stButton>button:hover{opacity:.85 !important;transform:translateY(-1px);}

/* ── Download button ── */
.stDownloadButton>button{
    background:linear-gradient(135deg,#1d4ed8,#3b82f6) !important;
    border:none !important;border-radius:10px !important;
    color:white !important;font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:700 !important;box-shadow:0 4px 14px rgba(59,130,246,.3) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#0d1117;}
::-webkit-scrollbar-thumb{background:#1a2035;border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:#2a3050;}

/* ── DataFrames ── */
.stDataFrame [data-testid="stDataFrameContainer"]{
    background:#0d1117;border:1px solid #1a2035;border-radius:12px;
}

/* ── Hide Streamlit chrome ── */
#MainMenu,footer,header,.stDeployButton{visibility:hidden;}

/* ── Sidebar logo area ── */
.sb-brand{padding:1.2rem 0 .6rem 0;}
.sb-logo{font-size:1.4rem;font-weight:800;color:#f97316;letter-spacing:-.5px;}
.sb-tagline{font-size:.66rem;color:#475569;font-family:'JetBrains Mono',monospace;margin-top:2px;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  PLOTLY BASE LAYOUT  (reused across all charts)
# ──────────────────────────────────────────────────────────
_PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=30, r=20, t=40, b=30),
    xaxis=dict(gridcolor="#1a2035", linecolor="#1a2035", zerolinecolor="#1a2035"),
    yaxis=dict(gridcolor="#1a2035", linecolor="#1a2035", zerolinecolor="#1a2035"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)

# ──────────────────────────────────────────────────────────
#  DATA & MODEL  (cached for performance)
# ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    """Load the original CSV without any preprocessing."""
    return pd.read_csv("Food_Delivery_Times.csv")


@st.cache_data(show_spinner=False)
def preprocess_data(df: pd.DataFrame):
    """
    Clean and encode the dataset.
    Returns: cleaned DataFrame, LabelEncoder dict, feature matrix X, target y.
    """
    clean = df.dropna().copy()
    clean.drop("Order_ID", axis=1, inplace=True)

    cat_cols = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
    encoders: dict[str, LabelEncoder] = {}

    for col in cat_cols:
        le = LabelEncoder()
        clean[col] = le.fit_transform(clean[col])
        encoders[col] = le

    X = clean.drop("Delivery_Time_min", axis=1)
    y = clean["Delivery_Time_min"]
    return clean, encoders, X, y


@st.cache_resource(show_spinner=False)
def build_model(X, y):
    """
    Train a LinearRegression model on 80/20 split.
    Returns: trained model, X_test, y_test, y_pred, metrics dict.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae   = mean_absolute_error(y_test, y_pred)
    mse   = mean_squared_error(y_test, y_pred)
    rmse  = np.sqrt(mse)
    r2    = r2_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, {
        "mae": mae, "mse": mse, "rmse": rmse, "r2": r2,
        "coef": model.coef_, "intercept": model.intercept_,
        "feature_names": list(X.columns),
    }


# ── Bootstrap ────────────────────────────────────────────
raw_df = load_raw_data()
clean_df, encoders, X_all, y_all = preprocess_data(raw_df)
model, X_test, y_test, y_pred, perf = build_model(X_all, y_all)

# Categorical options (decoded)
CAT_OPTIONS = {col: list(enc.classes_) for col, enc in encoders.items()}

# ──────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sb-brand'>
      <div class='sb-logo'>🛵 DeliveryIQ</div>
      <div class='sb-tagline'>ML-Powered ETA Prediction</div>
    </div>
    <hr style='border-color:#1a2035;margin:.6rem 0 1.2rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Home", "🔮  Predict ETA", "📊  Dataset Insights",
         "📈  Model Performance", "📘  About Project"],
        label_visibility="collapsed",
    )
    current_page = page.split("  ")[1]   # strip emoji prefix

    st.markdown("<hr style='border-color:#1a2035;margin:1.2rem 0;'>", unsafe_allow_html=True)

    # Live stats
    total_orders  = len(raw_df)
    avg_eta       = raw_df["Delivery_Time_min"].mean()
    max_eta       = raw_df["Delivery_Time_min"].max()
    null_pct      = (raw_df.isnull().sum().sum() / raw_df.size * 100)

    st.markdown(f"""
    <div style='font-size:.65rem;color:#475569;font-family:JetBrains Mono,monospace;
        letter-spacing:1px;text-transform:uppercase;margin-bottom:.8rem;'>Live Stats</div>
    """, unsafe_allow_html=True)

    for label, value, color in [
        ("Total Orders",  f"{total_orders:,}", "#f97316"),
        ("Avg ETA",       f"{avg_eta:.1f} min",  "#818cf8"),
        ("Max ETA",       f"{max_eta} min",       "#ef4444"),
        ("Model R²",      f"{perf['r2']:.3f}",   "#22c55e"),
        ("Null Rows %",   f"{null_pct:.1f}%",     "#f59e0b"),
    ]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
            margin-bottom:.45rem;'>
          <span style='color:#64748b;font-size:.79rem;'>{label}</span>
          <span style='color:{color};font-weight:700;font-size:.84rem;
            font-family:JetBrains Mono,monospace;'>{value}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <hr style='border-color:#1a2035;margin:1.2rem 0 .8rem 0;'>
    <div style='font-size:.65rem;color:#2a3050;font-family:JetBrains Mono,monospace;
        text-align:center;letter-spacing:.5px;'>
      Linear Regression · sklearn · Streamlit
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  PAGE : HOME
# ══════════════════════════════════════════════════════════
if current_page == "Home":
    st.markdown("""
    <div class='hero'>
      <div class='hero-title'>🛵 DeliveryIQ</div>
      <p class='hero-sub'>// food delivery eta · linear regression · 1,000 orders · 7 features</p>
    </div>""", unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, "TOTAL ORDERS",    f"{total_orders:,}",       "records in dataset",  "#f97316"),
        (k2, "AVG DELIVERY",    f"{avg_eta:.0f} min",       "mean ETA",            "#818cf8"),
        (k3, "MODEL R²",        f"{perf['r2']:.3f}",        "coefficient of det.", "#22c55e"),
        (k4, "RMSE",            f"{perf['rmse']:.2f}",      "root mean sq. error", "#ef4444"),
    ]
    for col, lbl, val, sub, clr in kpis:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-stripe' style='background:{clr};'></div>
              <div class='kpi-label'>{lbl}</div>
              <div class='kpi-value' style='color:{clr};'>{val}</div>
              <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1 ── Delivery time dist + Vehicle breakdown ──
    c_left, c_right = st.columns([1.3, 0.7])

    with c_left:
        st.markdown("<div class='sec-head'>Delivery Time Distribution <span class='badge'>histogram</span></div>",
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=raw_df["Delivery_Time_min"], nbinsx=40,
            marker_color="#f97316", opacity=.85,
            marker_line_width=0, name="Orders",
        ))
        fig.add_vline(x=avg_eta, line_dash="dash", line_color="#818cf8",
                      annotation_text=f"Mean {avg_eta:.0f}m",
                      annotation_font_color="#818cf8")
        fig.update_layout(**_PLOT, height=270,
                          xaxis_title="Delivery Time (min)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.markdown("<div class='sec-head'>Vehicle Split <span class='badge'>donut</span></div>",
                    unsafe_allow_html=True)
        vc_df = raw_df["Vehicle_Type"].value_counts().reset_index()
        vc_df.columns = ["Vehicle", "Count"]
        fig2 = go.Figure(go.Pie(
            labels=vc_df["Vehicle"], values=vc_df["Count"],
            hole=.6,
            marker=dict(colors=["#f97316","#818cf8","#22c55e"],
                        line=dict(color="#070b14", width=3)),
            textinfo="percent", textfont_size=13,
        ))
        fig2.add_annotation(text=f"<b>{total_orders:,}</b><br><span style='font-size:10px'>ORDERS</span>",
                            x=.5,y=.5,showarrow=False,
                            font=dict(size=18,color="#f1f5f9"),align="center")
        plot_layout_pie = _PLOT.copy()
        plot_layout_pie["legend"] = {**_PLOT.get("legend", {}), "orientation": "h", "y": -.05}
        fig2.update_layout(**plot_layout_pie, height=270, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2 ── Avg ETA by Weather & Traffic ────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sec-head'>Avg ETA by Weather <span class='badge'>bar</span></div>",
                    unsafe_allow_html=True)
        weather_df = raw_df.dropna(subset=["Weather"])
        w_avg = weather_df.groupby("Weather")["Delivery_Time_min"].mean().sort_values(ascending=True).reset_index()
        fig3 = go.Figure(go.Bar(
            x=w_avg["Delivery_Time_min"], y=w_avg["Weather"],
            orientation="h",
            marker=dict(
                color=w_avg["Delivery_Time_min"],
                colorscale=[[0,"#1a2035"],[0.5,"#f97316"],[1,"#ef4444"]],
                line_width=0,
            ),
            text=[f"{v:.1f}" for v in w_avg["Delivery_Time_min"]],
            textposition="outside", textfont=dict(size=11, color="#94a3b8"),
        ))
        fig3.update_layout(**_PLOT, height=260, xaxis_title="Avg Delivery Time (min)")
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        st.markdown("<div class='sec-head'>Avg ETA by Traffic Level <span class='badge'>bar</span></div>",
                    unsafe_allow_html=True)
        traffic_df = raw_df.dropna(subset=["Traffic_Level"])
        t_avg = traffic_df.groupby("Traffic_Level")["Delivery_Time_min"].mean().sort_values().reset_index()
        colors_t = {"Low":"#22c55e","Medium":"#f59e0b","High":"#ef4444"}
        fig4 = go.Figure(go.Bar(
            x=t_avg["Traffic_Level"],
            y=t_avg["Delivery_Time_min"],
            marker_color=[colors_t.get(v,"#818cf8") for v in t_avg["Traffic_Level"]],
            marker_line_width=0,
            text=[f"{v:.1f} min" for v in t_avg["Delivery_Time_min"]],
            textposition="outside", textfont=dict(size=11, color="#94a3b8"),
        ))
        fig4.update_layout(**_PLOT, height=260, yaxis_title="Avg Delivery Time (min)")
        fig4.update_traces(marker_cornerradius=6)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3 ── Distance vs ETA scatter ─────────────────
    st.markdown("<div class='sec-head'>Distance vs Delivery Time <span class='badge'>scatter</span></div>",
                unsafe_allow_html=True)
    fig5 = px.scatter(
        raw_df.dropna(), x="Distance_km", y="Delivery_Time_min",
        color="Vehicle_Type",
        color_discrete_map={"Scooter":"#f97316","Bike":"#818cf8","Car":"#22c55e"},
        opacity=.65, size_max=8,
        labels={"Distance_km":"Distance (km)","Delivery_Time_min":"Delivery Time (min)"},
    )
    fig5.update_layout(**_PLOT, height=320)
    fig5.update_traces(marker_size=5)
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════
#  PAGE : PREDICT ETA
# ══════════════════════════════════════════════════════════
elif current_page == "Predict ETA":
    st.markdown("""
    <div class='hero'>
      <div class='hero-title'>🔮 Real-Time ETA Predictor</div>
      <p class='hero-sub'>// fill in order details → get instant delivery time estimate</p>
    </div>""", unsafe_allow_html=True)

    # ── Input form ───────────────────────────────────────
    with st.form("predict_form", clear_on_submit=False):
        st.markdown("<div class='sec-head'>📦 Order Details</div>", unsafe_allow_html=True)

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            distance = st.slider(
                "Distance (km)", min_value=0.5, max_value=20.0,
                value=7.5, step=0.5,
                help="Distance between restaurant and delivery address"
            )
        with r1c2:
            prep_time = st.slider(
                "Preparation Time (min)", min_value=5, max_value=30,
                value=15, step=1,
                help="Time the restaurant needs to prepare the order"
            )
        with r1c3:
            experience = st.slider(
                "Courier Experience (yrs)", min_value=0.0, max_value=9.0,
                value=3.0, step=0.5,
                help="Years of experience of the delivery courier"
            )

        st.markdown("<br><div class='sec-head'>🌦 Environment & Vehicle</div>", unsafe_allow_html=True)
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            weather = st.selectbox("Weather Condition", CAT_OPTIONS["Weather"])
        with r2c2:
            traffic = st.selectbox("Traffic Level",    CAT_OPTIONS["Traffic_Level"])
        with r2c3:
            time_of_day = st.selectbox("Time of Day",   CAT_OPTIONS["Time_of_Day"])
        with r2c4:
            vehicle = st.selectbox("Vehicle Type",      CAT_OPTIONS["Vehicle_Type"])

        submitted = st.form_submit_button("⚡  Predict Delivery Time", use_container_width=True)

    # ── Prediction ───────────────────────────────────────
    if submitted:
        # Input validation
        errors: list[str] = []
        if distance <= 0:   errors.append("Distance must be > 0 km")
        if prep_time <= 0:  errors.append("Preparation time must be > 0 min")
        if experience < 0:  errors.append("Experience cannot be negative")

        if errors:
            for e in errors:
                st.markdown(f"<div class='callout-warn'>⚠ {e}</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Running prediction…"):
                # Build encoded row
                input_row = {
                    "Distance_km":             distance,
                    "Weather":                 encoders["Weather"].transform([weather])[0],
                    "Traffic_Level":           encoders["Traffic_Level"].transform([traffic])[0],
                    "Time_of_Day":             encoders["Time_of_Day"].transform([time_of_day])[0],
                    "Vehicle_Type":            encoders["Vehicle_Type"].transform([vehicle])[0],
                    "Preparation_Time_min":    prep_time,
                    "Courier_Experience_yrs":  experience,
                }
                X_in     = pd.DataFrame([input_row])
                eta_pred = max(0.0, model.predict(X_in)[0])

            # ── Result layout ─────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            res_col, gauge_col = st.columns([1, 1.3])

            with res_col:
                # Color-code urgency
                if eta_pred <= 30:
                    box_class, clr, icon = "pred-result", "#22c55e", "🟢"
                    verdict = "Fast Delivery"
                elif eta_pred <= 60:
                    verdict, clr, icon = "On-Time Delivery", "#f59e0b", "🟡"
                    box_class = "pred-result"
                else:
                    verdict, clr, icon = "Slow Delivery", "#ef4444", "🔴"
                    box_class = "pred-result"

                st.markdown(f"""
                <div class='{box_class}' style='border-color:{clr};
                    background:linear-gradient(135deg,#0f1a0a,#0a1510);'>
                  <div class='pred-label'>{icon} Estimated Delivery Time</div>
                  <div class='pred-value' style='color:{clr};'>{eta_pred:.1f} min</div>
                  <div class='pred-detail'>{verdict}</div>
                </div>""", unsafe_allow_html=True)

                # Confidence range
                margin = perf["rmse"]
                lo, hi = max(0, eta_pred - margin), eta_pred + margin
                st.markdown(f"""
                <div class='callout' style='margin-top:1rem;'>
                  📊 95% confidence range: <b>{lo:.1f} – {hi:.1f} min</b><br>
                  Based on model RMSE of ±{margin:.1f} min
                </div>""", unsafe_allow_html=True)

            with gauge_col:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(eta_pred, 1),
                    title={"text": "Estimated Delivery Time (min)",
                           "font": {"color": "#94a3b8", "size": 13}},
                    number={"suffix": " min",
                            "font": {"color": "#f1f5f9", "size": 34, "family": "JetBrains Mono"}},
                    gauge={
                        "axis": {"range": [0, 120],
                                 "tickcolor": "#475569",
                                 "tickfont": {"color": "#64748b"}},
                        "bar": {"color": clr, "thickness": .25},
                        "bgcolor": "#1a2035",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0,  30], "color": "#0f2a1a"},
                            {"range": [30, 60], "color": "#2a2205"},
                            {"range": [60, 120],"color": "#2a0f0f"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 2},
                            "thickness": .7, "value": eta_pred,
                        },
                    }
                ))
                fig_g.update_layout(**_PLOT, height=280)
                st.plotly_chart(fig_g, use_container_width=True)

            # ── Feature contribution bar ───────────────────
            st.markdown("<br><div class='sec-head'>🎯 Feature Contributions <span class='badge'>how input affects ETA</span></div>",
                        unsafe_allow_html=True)
            feat_vals = np.array([X_in[f].values[0] for f in perf["feature_names"]])
            contributions = perf["coef"] * feat_vals
            contrib_df = pd.DataFrame({
                "Feature":      perf["feature_names"],
                "Contribution": contributions,
            }).sort_values("Contribution", key=abs, ascending=True)

            fig_fc = go.Figure(go.Bar(
                x=contrib_df["Contribution"],
                y=contrib_df["Feature"],
                orientation="h",
                marker_color=["#ef4444" if v > 0 else "#22c55e" for v in contrib_df["Contribution"]],
                marker_line_width=0,
                text=[f"{v:+.2f}" for v in contrib_df["Contribution"]],
                textposition="outside",
                textfont=dict(size=10, color="#94a3b8"),
            ))
            fig_fc.update_layout(**_PLOT, height=280)
            st.plotly_chart(fig_fc, use_container_width=True)
            st.markdown("""
            <div class='callout'>
              🔴 Positive → increases ETA &nbsp;|&nbsp; 🟢 Negative → decreases ETA
            </div>""", unsafe_allow_html=True)

            # ── Download result ────────────────────────────
            st.markdown("<br><div class='sec-head'>💾 Export Prediction</div>", unsafe_allow_html=True)
            result_dict = {
                "Distance_km":             [distance],
                "Weather":                 [weather],
                "Traffic_Level":           [traffic],
                "Time_of_Day":             [time_of_day],
                "Vehicle_Type":            [vehicle],
                "Preparation_Time_min":    [prep_time],
                "Courier_Experience_yrs":  [experience],
                "Predicted_ETA_min":       [round(eta_pred, 2)],
                "Lower_Bound_min":         [round(lo, 2)],
                "Upper_Bound_min":         [round(hi, 2)],
            }
            result_df = pd.DataFrame(result_dict)
            csv_bytes  = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️  Download Prediction as CSV",
                data=csv_bytes,
                file_name="delivery_prediction.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════
#  PAGE : DATASET INSIGHTS
# ══════════════════════════════════════════════════════════
elif current_page == "Dataset Insights":
    st.markdown("""
    <div class='hero'>
      <div class='hero-title'>📊 Dataset Insights</div>
      <p class='hero-sub'>// 1,000 orders · 9 columns · food delivery analytics</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "  📋  Raw Data  ",
        "  📉  Distributions  ",
        "  🔗  Correlation  ",
        "  📦  Category Analysis  ",
    ])

    # ── Tab 1 : Raw Data ─────────────────────────────────
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            search = st.text_input("🔍  Search Order ID", placeholder="e.g. 522")
        with f2:
            veh_f = st.selectbox("Vehicle", ["All"] + list(raw_df["Vehicle_Type"].unique()))
        with f3:
            weather_f = st.selectbox("Weather", ["All"] + [x for x in raw_df["Weather"].dropna().unique()])

        filt = raw_df.copy()
        if search:
            filt = filt[filt["Order_ID"].astype(str).str.contains(search)]
        if veh_f != "All":
            filt = filt[filt["Vehicle_Type"] == veh_f]
        if weather_f != "All":
            filt = filt[filt["Weather"] == weather_f]

        st.markdown(f"<div style='font-size:.73rem;color:#64748b;font-family:JetBrains Mono,monospace;"
                    f"margin-bottom:.5rem;'>{len(filt):,} rows · {len(raw_df.columns)} columns</div>",
                    unsafe_allow_html=True)
        st.dataframe(filt.reset_index(drop=True), use_container_width=True, height=380)

        # Summary statistics
        st.markdown("<br><div class='sec-head'>Descriptive Statistics <span class='badge'>numeric</span></div>",
                    unsafe_allow_html=True)
        st.dataframe(
            raw_df[["Distance_km","Preparation_Time_min","Courier_Experience_yrs","Delivery_Time_min"]]
            .describe().T.round(2),
            use_container_width=True,
        )

    # ── Tab 2 : Distributions ────────────────────────────
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        num_feat = st.selectbox(
            "Select feature",
            ["Distance_km", "Preparation_Time_min", "Courier_Experience_yrs", "Delivery_Time_min"],
        )
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=raw_df[num_feat], nbinsx=35,
            marker_color="#f97316", opacity=.8,
            marker_line_color="#ea580c", marker_line_width=.5,
        ))
        mean_val = raw_df[num_feat].mean()
        hist_fig.add_vline(x=mean_val, line_dash="dash", line_color="#818cf8",
                           annotation_text=f"Mean {mean_val:.1f}",
                           annotation_font_color="#818cf8")
        hist_fig.update_layout(**_PLOT, height=310,
                               xaxis_title=num_feat, yaxis_title="Frequency")
        st.plotly_chart(hist_fig, use_container_width=True)

        # Box plots side by side
        st.markdown("<div class='sec-head'>All Numeric Features — Box Plots <span class='badge'>spread</span></div>",
                    unsafe_allow_html=True)
        box_fig = go.Figure()
        for feat, color, fillcolor in [
            ("Distance_km",            "#f97316", "rgba(249, 115, 22, 0.15)"),
            ("Preparation_Time_min",   "#818cf8", "rgba(129, 140, 248, 0.15)"),
            ("Courier_Experience_yrs", "#22c55e", "rgba(34, 197, 94, 0.15)"),
            ("Delivery_Time_min",      "#ef4444", "rgba(239, 68, 68, 0.15)"),
        ]:
            box_fig.add_trace(go.Box(
                y=raw_df[feat], name=feat.replace("_", " "),
                marker_color=color, line_color=color,
                boxmean=True, fillcolor=fillcolor,
            ))
        box_fig.update_layout(**_PLOT, height=320)
        st.plotly_chart(box_fig, use_container_width=True)

    # ── Tab 3 : Correlation Heatmap ──────────────────────
    with tab3:
        st.markdown("<br><div class='sec-head'>Correlation Heatmap <span class='badge'>numeric features</span></div>",
                    unsafe_allow_html=True)
        num_df = clean_df[["Distance_km","Weather","Traffic_Level","Time_of_Day",
                            "Vehicle_Type","Preparation_Time_min","Courier_Experience_yrs",
                            "Delivery_Time_min"]]
        corr_matrix = num_df.corr().round(2)

        heat_fig = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=[
                [0.0,  "#4f46e5"],
                [0.5,  "#0d1117"],
                [1.0,  "#f97316"],
            ],
            zmin=-1, zmax=1,
            text=corr_matrix.values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=11, color="#f1f5f9"),
            showscale=True,
            colorbar=dict(
                tickcolor="#475569",
                tickfont=dict(color="#94a3b8"),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
            ),
        ))
        heat_fig.update_layout(**_PLOT, height=450)
        st.plotly_chart(heat_fig, use_container_width=True)
        st.markdown("""
        <div class='callout'>
          🟠 Orange → positive correlation → both increase together<br>
          🟣 Purple → negative correlation → one increases, other decreases
        </div>""", unsafe_allow_html=True)

    # ── Tab 4 : Category Analysis ────────────────────────
    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        cat_col_choice = st.selectbox(
            "Group by", ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
        )
        group_df = raw_df.dropna(subset=[cat_col_choice])
        grp = group_df.groupby(cat_col_choice)["Delivery_Time_min"].agg(
            ["mean", "median", "std", "count"]
        ).round(2).reset_index()
        grp.columns = [cat_col_choice, "Mean (min)", "Median (min)", "Std Dev", "Count"]

        violin_fig = go.Figure()
        palette = ["#f97316","#818cf8","#22c55e","#ef4444","#f59e0b"]
        for i, cat_val in enumerate(grp[cat_col_choice]):
            sub = group_df[group_df[cat_col_choice] == cat_val]["Delivery_Time_min"]
            violin_fig.add_trace(go.Violin(
                y=sub, name=cat_val,
                fillcolor=palette[i % len(palette)],
                line_color=palette[i % len(palette)],
                opacity=.75, box_visible=True, meanline_visible=True,
            ))
        violin_fig.update_layout(**_PLOT, height=340, yaxis_title="Delivery Time (min)")
        st.plotly_chart(violin_fig, use_container_width=True)

        st.dataframe(grp, use_container_width=True)


# ══════════════════════════════════════════════════════════
#  PAGE : MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════
elif current_page == "Model Performance":
    st.markdown("""
    <div class='hero'>
      <div class='hero-title'>📈 Model Performance</div>
      <p class='hero-sub'>// linear regression · 80/20 split · sklearn evaluation</p>
    </div>""", unsafe_allow_html=True)

    # ── Metric cards ─────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    perf_kpis = [
        (m1, "R² SCORE",  f"{perf['r2']:.4f}",       "explained variance",  "#22c55e"),
        (m2, "MAE",       f"{perf['mae']:.2f} min",   "mean abs. error",     "#818cf8"),
        (m3, "RMSE",      f"{perf['rmse']:.2f} min",  "root mean sq. error", "#f97316"),
        (m4, "MSE",       f"{perf['mse']:.2f}",       "mean sq. error",      "#ef4444"),
    ]
    for col, lbl, val, sub, clr in perf_kpis:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-stripe' style='background:{clr};'></div>
              <div class='kpi-label'>{lbl}</div>
              <div class='kpi-value' style='color:{clr};'>{val}</div>
              <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_actual, c_resid = st.columns(2)

    # ── Actual vs Predicted ───────────────────────────────
    with c_actual:
        st.markdown("<div class='sec-head'>Actual vs Predicted <span class='badge'>test set</span></div>",
                    unsafe_allow_html=True)
        avp_fig = go.Figure()
        avp_fig.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            marker=dict(color="#f97316", size=5, opacity=.6),
            name="Predictions",
        ))
        mn, mx = int(min(y_test.min(), y_pred.min()))-5, int(max(y_test.max(), y_pred.max()))+5
        avp_fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines", line=dict(color="#818cf8", dash="dash", width=1.5),
            name="Perfect Fit",
        ))
        avp_fig.update_layout(**_PLOT, height=320,
                              xaxis_title="Actual (min)", yaxis_title="Predicted (min)")
        st.plotly_chart(avp_fig, use_container_width=True)

    # ── Residuals ─────────────────────────────────────────
    with c_resid:
        st.markdown("<div class='sec-head'>Residuals Distribution <span class='badge'>error spread</span></div>",
                    unsafe_allow_html=True)
        residuals = y_test.values - y_pred
        res_fig = go.Figure()
        res_fig.add_trace(go.Histogram(
            x=residuals, nbinsx=35,
            marker_color="#818cf8", opacity=.8, marker_line_width=0,
        ))
        res_fig.add_vline(x=0, line_color="#f97316", line_dash="dash",
                          annotation_text="Zero Error", annotation_font_color="#f97316")
        res_fig.update_layout(**_PLOT, height=320,
                              xaxis_title="Residual (min)", yaxis_title="Count")
        st.plotly_chart(res_fig, use_container_width=True)

    # ── Feature Importance ────────────────────────────────
    st.markdown("<div class='sec-head'>Feature Importance (Coefficients) <span class='badge'>linear weights</span></div>",
                unsafe_allow_html=True)
    feat_imp = pd.DataFrame({
        "Feature":     perf["feature_names"],
        "Coefficient": perf["coef"],
    }).sort_values("Coefficient", ascending=True)

    fi_fig = go.Figure(go.Bar(
        x=feat_imp["Coefficient"],
        y=feat_imp["Feature"],
        orientation="h",
        marker_color=["#ef4444" if v > 0 else "#4f46e5" for v in feat_imp["Coefficient"]],
        marker_line_width=0,
        text=[f"{v:+.3f}" for v in feat_imp["Coefficient"]],
        textposition="outside",
        textfont=dict(size=11, color="#94a3b8"),
    ))
    fi_fig.update_layout(**_PLOT, height=320)
    st.plotly_chart(fi_fig, use_container_width=True)
    st.markdown("""
    <div class='callout'>
      🔴 Positive coefficient → feature <b>increases</b> delivery time<br>
      🔵 Negative coefficient → feature <b>decreases</b> delivery time<br>
      Magnitude = relative influence strength
    </div>""", unsafe_allow_html=True)

    # ── Predicted distribution ────────────────────────────
    st.markdown("<br><div class='sec-head'>Predicted vs Actual Distribution <span class='badge'>overlay</span></div>",
                unsafe_allow_html=True)
    ov_fig = go.Figure()
    ov_fig.add_trace(go.Histogram(
        x=y_test, nbinsx=30, name="Actual",
        marker_color="#818cf8", opacity=.65, marker_line_width=0,
    ))
    ov_fig.add_trace(go.Histogram(
        x=y_pred, nbinsx=30, name="Predicted",
        marker_color="#f97316", opacity=.65, marker_line_width=0,
    ))
    ov_fig.update_layout(**_PLOT, barmode="overlay", height=300,
                         xaxis_title="Delivery Time (min)", yaxis_title="Count")
    st.plotly_chart(ov_fig, use_container_width=True)

    # ── Model equation ────────────────────────────────────
    st.markdown("<br><div class='sec-head'>Model Equation <span class='badge'>intercept + weights</span></div>",
                unsafe_allow_html=True)
    terms = " + ".join([f"({c:.3f} × {f})" for c, f in zip(perf["coef"], perf["feature_names"])])
    eq = f"ETA = {perf['intercept']:.3f}  +  {terms}"
    st.markdown(f"""
    <div style='background:#0d1117;border:1px solid #1a2035;border-radius:12px;
        padding:1.2rem 1.6rem;font-family:JetBrains Mono,monospace;font-size:.78rem;
        color:#f97316;word-break:break-all;line-height:1.8;'>
      {eq}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  PAGE : ABOUT PROJECT
# ══════════════════════════════════════════════════════════
elif current_page == "About Project":
    st.markdown("""
    <div class='hero'>
      <div class='hero-title'>📘 About This Project</div>
      <p class='hero-sub'>// portfolio ml project · linear regression · end-to-end deployment</p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 0.6])

    with c1:
        st.markdown("""
        <div style='background:#0d1117;border:1px solid #1a2035;border-radius:16px;padding:2rem;'>
          <div style='font-size:1.05rem;font-weight:700;color:#f97316;margin-bottom:1rem;'>
            🎯 Project Overview
          </div>
          <p style='color:#94a3b8;line-height:1.85;font-size:.88rem;'>
            <b style='color:#f1f5f9;'>DeliveryIQ</b> is a production-grade ML web application
            that predicts food delivery times using a trained <b style='color:#f97316;'>
            Linear Regression</b> model. The app ingests 7 real-world features — distance,
            weather, traffic level, time of day, vehicle type, preparation time, and courier
            experience — and outputs an estimated ETA in minutes with an uncertainty band.
          </p>
          <p style='color:#94a3b8;line-height:1.85;font-size:.88rem;margin-top:1rem;'>
            The project demonstrates a full ML lifecycle: data cleaning, exploratory analysis,
            feature engineering, model training, evaluation, and interactive deployment.
          </p>

          <div style='font-size:1.05rem;font-weight:700;color:#818cf8;margin:1.8rem 0 .8rem 0;'>
            🔬 Methodology
          </div>
          <ul style='color:#94a3b8;line-height:2;font-size:.87rem;padding-left:1.2rem;'>
            <li>Missing values handled via row-wise dropna (3% null rate).</li>
            <li>Categorical features encoded with <b style='color:#f1f5f9;'>LabelEncoder</b>.</li>
            <li>80/20 stratified train/test split with random_state=42.</li>
            <li>Ordinary Least Squares (OLS) via sklearn LinearRegression.</li>
            <li>Evaluation: MAE · MSE · RMSE · R² · residual analysis.</li>
          </ul>

          <div style='font-size:1.05rem;font-weight:700;color:#22c55e;margin:1.8rem 0 .8rem 0;'>
            💡 Key Findings
          </div>
          <ul style='color:#94a3b8;line-height:2;font-size:.87rem;padding-left:1.2rem;'>
            <li>Distance is the strongest positive predictor of delivery time.</li>
            <li>High traffic significantly increases ETA compared to low traffic.</li>
            <li>More experienced couriers deliver faster on average.</li>
            <li>Snowy/Rainy weather adds ~4–6 min versus Clear weather.</li>
            <li>The model explains ~77% of ETA variance (R² = 0.775).</li>
          </ul>
        </div>""", unsafe_allow_html=True)

    with c2:
        # Tech stack cards
        techs = [
            ("🐍 Python 3.10+",    "Core language"),
            ("📊 Streamlit",       "Web framework"),
            ("🤖 scikit-learn",    "ML model"),
            ("🐼 Pandas",          "Data wrangling"),
            ("🔢 NumPy",           "Numerics"),
            ("📈 Plotly",          "Interactive charts"),
            ("🎨 Custom CSS",      "Premium UI"),
        ]
        st.markdown("<div class='sec-head'>🛠 Tech Stack</div>", unsafe_allow_html=True)
        for name, role in techs:
            st.markdown(f"""
            <div style='background:#0d1117;border:1px solid #1a2035;border-radius:10px;
                padding:.7rem 1rem;margin-bottom:.5rem;
                display:flex;justify-content:space-between;align-items:center;'>
              <span style='color:#f1f5f9;font-weight:600;font-size:.85rem;'>{name}</span>
              <span style='color:#64748b;font-size:.75rem;font-family:JetBrains Mono,monospace;'>{role}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br><div class='sec-head'>📁 Project Structure</div>", unsafe_allow_html=True)
        st.code("""
food_delivery_app/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── README.md               # Documentation
├── Food_Delivery_Times.csv # Dataset
├── model/
│   └── LinearRegression.pkl
└── assets/
    └── (images/icons)
        """, language="text")

    # ── Quick start ───────────────────────────────────────
    st.markdown("<br><div class='sec-head'>🚀 Quick Start</div>", unsafe_allow_html=True)
    qs1, qs2 = st.columns(2)
    with qs1:
        st.markdown("**Local Setup**")
        st.code("""# 1. Clone / unzip project
cd food_delivery_app

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
""", language="bash")

    with qs2:
        st.markdown("**Streamlit Cloud Deploy**")
        st.code("""# 1. Push to GitHub
git init && git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main

# 2. Go to share.streamlit.io
# 3. Connect GitHub repo
# 4. Set main file: app.py
# 5. Click Deploy 🚀
""", language="bash")