"""
Ericsson Patent Domain Predictor — Streamlit App
=================================================
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ericsson Patent Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
.metric-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 18px 20px; text-align: center; }
.metric-label { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 32px; font-weight: 700; line-height: 1; }
.metric-sub { font-size: 11px; color: #8b949e; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
.pred-box { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 32px; text-align: center; }
.pred-number { font-family: 'JetBrains Mono', monospace; font-size: 72px; font-weight: 800; line-height: 1; }
.pred-label { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px; }
.pred-range { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #8b949e; margin-top: 10px; }
.section-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px; }
.info-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #21262d; font-size: 13px; }
.info-row:last-child { border-bottom: none; }
.info-key { color: #8b949e; font-family: 'JetBrains Mono', monospace; }
.info-val { color: #e6edf3; font-weight: 500; }
.pill-good { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid rgba(63,185,80,0.3); border-radius: 20px; padding: 2px 10px; font-size: 12px; font-family: 'JetBrains Mono', monospace; }
.pill-warn { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid rgba(210,153,34,0.3); border-radius: 20px; padding: 2px 10px; font-size: 12px; font-family: 'JetBrains Mono', monospace; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DOMAIN_FEATURES = ['year', 'quarter', 'lag1', 'lag2', 'lag4',
                   'roll4_mean', 'roll4_std', 'qoq', 'yoy']

DOMAINS = {
    'kw_5g'        : {'label': '5G',          'color': '#2f81f7'},
    'kw_ai_ml'     : {'label': 'AI / ML',      'color': '#a371f7'},
    'kw_cloud_edge': {'label': 'Cloud & Edge', 'color': '#3fb950'},
    'kw_security'  : {'label': 'Security',     'color': '#f85149'},
    'kw_iot'       : {'label': 'IoT',          'color': '#d29922'},
    'kw_network'   : {'label': 'Network',      'color': '#58a6ff'},
    'kw_energy'    : {'label': 'Energy',       'color': '#56d364'},
    'kw_antenna'   : {'label': 'Antenna',      'color': '#ff9a6c'},
    'kw_data'      : {'label': 'Data',         'color': '#bc8cff'},
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(BASE_DIR, "ericsson_patent_rich_dataset.csv")
    df = pd.read_csv(path)
    df['patent_date'] = pd.to_datetime(df['patent_date'])
    return df

def build_domain_ts(df, col):
    ts = (df[df[col] == 1]
          .groupby(['year', 'quarter']).size()
          .reset_index(name='patent_count')
          .sort_values(['year', 'quarter']).reset_index(drop=True))
    ts['lag1']       = ts['patent_count'].shift(1)
    ts['lag2']       = ts['patent_count'].shift(2)
    ts['lag4']       = ts['patent_count'].shift(4)
    ts['roll4_mean'] = ts['patent_count'].shift(1).rolling(4).mean()
    ts['roll4_std']  = ts['patent_count'].shift(1).rolling(4).std()
    ts['qoq']        = ts['patent_count'].pct_change(1)
    ts['yoy']        = ts['patent_count'].pct_change(4)
    ts['target']     = ts['patent_count'].shift(-1)
    return ts.dropna().reset_index(drop=True)

def train_all_domains(df):
    results = {}
    for col, info in DOMAINS.items():
        ts    = build_domain_ts(df, col)
        split = int(len(ts) * 0.8)
        tr    = ts.iloc[:split]
        te    = ts.iloc[split:]
        sc    = StandardScaler()
        Xtr   = sc.fit_transform(tr[DOMAIN_FEATURES])
        Xte   = sc.transform(te[DOMAIN_FEATURES])

        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(Xtr, tr['target'])
        rf_pred = rf.predict(Xte)
        rf_mae  = mean_absolute_error(te['target'], rf_pred)

        rg = Ridge(alpha=10.0)
        rg.fit(Xtr, tr['target'])
        rg_pred = rg.predict(Xte)
        rg_mae  = mean_absolute_error(te['target'], rg_pred)

        if rf_mae <= rg_mae:
            model, pred, mae, mname = rf, rf_pred, rf_mae, 'Random Forest'
        else:
            model, pred, mae, mname = rg, rg_pred, rg_mae, 'Ridge'

        results[col] = {
            'label'     : info['label'],
            'model'     : model,
            'model_name': mname,
            'scaler'    : sc,
            'ts'        : ts,
            'train'     : tr,
            'test'      : te,
            'y_pred'    : pred,
            'mae'       : round(mae, 2),
            'rmse'      : round(float(np.sqrt(mean_squared_error(te['target'], pred))), 2),
            'r2'        : round(r2_score(te['target'], pred), 3),
            'naive_mae' : round(mean_absolute_error(te['target'], te['lag1']), 2),
        }
    return results

@st.cache_resource
def load_models(_df):
    pkl_path = os.path.join(BASE_DIR, "domain_models.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return train_all_domains(_df)

def predict_domain(col, models):
    r      = models[col]
    ts     = r['ts']
    last   = ts.iloc[-1]
    X_row  = {f: last[f] for f in DOMAIN_FEATURES}
    X      = pd.DataFrame([X_row])
    pred   = float(r['model'].predict(r['scaler'].transform(X))[0])
    pred   = max(0.0, round(pred, 1))
    pred_q = int(last['quarter']) % 4 + 1
    pred_y = int(last['year']) + (1 if pred_q == 1 else 0)
    return pred, pred_y, pred_q

# ── Plot helpers ──────────────────────────────────────────────────────────────
def dark_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', linewidth=0.8)
    return fig, ax

def dark_fig_multi(nrows, ncols, w, h):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor('#0d1117')
    flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for ax in flat:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363d')
        ax.grid(True, color='#21262d', linewidth=0.6)
    return fig, axes

def metric_card(col_widget, label, value, sub, color):
    col_widget.markdown(
        "<div class='metric-card'>"
        "<div class='metric-label'>" + label + "</div>"
        "<div class='metric-value' style='color:" + color + "'>" + value + "</div>"
        "<div class='metric-sub'>" + sub + "</div>"
        "</div>",
        unsafe_allow_html=True
    )

def info_row(key, val):
    st.markdown(
        "<div class='info-row'>"
        "<span class='info-key'>" + key + "</span>"
        "<span class='info-val'>" + val + "</span>"
        "</div>",
        unsafe_allow_html=True
    )

# ── Load data & models ────────────────────────────────────────────────────────
df     = load_data()
models = load_models(df)

label_to_color = {DOMAINS[c]['label']: DOMAINS[c]['color'] for c in DOMAINS}
label_to_col   = {DOMAINS[c]['label']: c for c in DOMAINS}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Patent Predictor")
    st.markdown(
        "<div style='color:#8b949e;font-size:12px;margin-bottom:20px;'>"
        "Ericsson Innovation Dataset<br>Domain-Specific Forecasting</div>",
        unsafe_allow_html=True
    )
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔮 Predict", "📈 Trends", "📊 Model Report"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    yr_min = int(df['year'].min())
    yr_max = int(df['year'].max())
    st.markdown(
        "<div style='color:#8b949e;font-size:11px;font-family:monospace;'>"
        "Dataset: " + f"{df.shape[0]:,}" + " patents<br>"
        "Domains: 9<br>"
        "Years: " + str(yr_min) + "–" + str(yr_max) + "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        "<div style='color:#8b949e;font-size:11px;'>Built with Streamlit · scikit-learn</div>",
        unsafe_allow_html=True
    )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("## 🏠 Patent Landscape Overview")
    st.markdown(
        "<div style='color:#8b949e;margin-bottom:24px;'>"
        "Ericsson patent activity across 9 technology domains — "
        "total counts, trends, and next-quarter forecasts at a glance.</div>",
        unsafe_allow_html=True
    )

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    total = df.shape[0]
    years = yr_max - yr_min + 1
    kw_cols = list(DOMAINS.keys())
    multi = int((df[kw_cols].sum(axis=1) > 1).sum())

    metric_card(c1, "Total Patents",  str(total) + " patents",   str(yr_min) + "–" + str(yr_max), "#2f81f7")
    metric_card(c2, "Years Covered",  str(years),                "years of data",                "#3fb950")
    metric_card(c3, "Tech Domains",   "9",                       "tracked separately",           "#a371f7")
    metric_card(c4, "Multi-domain",   str(multi) + " patents",   "span 2+ domains",              "#d29922")

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("<div class='section-eyebrow'>TOTAL PATENTS BY DOMAIN</div>",
                    unsafe_allow_html=True)
        counts   = {DOMAINS[c]['label']: int(df[c].sum()) for c in DOMAINS}
        df_c     = (pd.DataFrame(list(counts.items()), columns=['Domain', 'Count'])
                      .sort_values('Count', ascending=True))
        bar_clrs = [label_to_color[l] for l in df_c['Domain']]

        fig, ax = dark_fig(7, 4)
        bars = ax.barh(df_c['Domain'], df_c['Count'], color=bar_clrs,
                       edgecolor='none', alpha=0.9)
        ax.bar_label(bars, fmt='{:,.0f}', padding=5,
                     color='#8b949e', fontsize=9, fontfamily='monospace')
        ax.set_xlabel('Total Patents')
        ax.set_title('Patent Volume per Domain', color='#e6edf3',
                     fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("<div class='section-eyebrow'>NEXT QUARTER FORECAST</div>",
                    unsafe_allow_html=True)
        for col_key, info in DOMAINS.items():
            pred, py, pq = predict_domain(col_key, models)
            mae   = models[col_key]['mae']
            color = info['color']
            label = info['label']
            st.markdown(
                "<div style='display:flex;justify-content:space-between;"
                "align-items:center;padding:9px 0;border-bottom:1px solid #21262d;'>"
                "<div style='display:flex;align-items:center;gap:10px;'>"
                "<div style='width:8px;height:8px;border-radius:50%;background:"
                + color + ";flex-shrink:0'></div>"
                "<span style='font-size:13px;'>" + label + "</span></div>"
                "<div style='text-align:right;'>"
                "<span style='font-family:monospace;font-size:14px;font-weight:700;color:"
                + color + "'>" + str(int(pred)) + "</span>"
                "<span style='font-size:11px;color:#8b949e;margin-left:4px;'>"
                "±" + str(mae) + "  " + str(py) + " Q" + str(pq) + "</span>"
                "</div></div>",
                unsafe_allow_html=True
            )

    # Yearly trend
    st.markdown("<br><div class='section-eyebrow'>ANNUAL PATENT ACTIVITY BY DOMAIN</div>",
                unsafe_allow_html=True)
    yearly_df = pd.DataFrame({
        DOMAINS[c]['label']: df[df[c] == 1].groupby('year').size()
        for c in DOMAINS
    }).fillna(0)

    fig, ax = dark_fig(12, 4)
    for col_key, info in DOMAINS.items():
        lbl = info['label']
        if lbl in yearly_df.columns:
            ax.plot(yearly_df.index, yearly_df[lbl],
                    color=info['color'], linewidth=2, marker='o',
                    markersize=3, label=lbl, alpha=0.85)
    ax.set_xlabel('Year')
    ax.set_ylabel('Patents Filed')
    ax.set_title('Yearly Patent Trends by Domain', color='#e6edf3',
                 fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='upper left', facecolor='#161b22', edgecolor='#30363d',
              labelcolor='#e6edf3', fontsize=9, ncol=3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown("## 🔮 Next-Quarter Prediction")
    st.markdown(
        "<div style='color:#8b949e;margin-bottom:24px;'>"
        "Select a technology domain to see the model forecast for the upcoming quarter.</div>",
        unsafe_allow_html=True
    )

    selected_label = st.selectbox(
        "Technology Domain",
        [DOMAINS[c]['label'] for c in DOMAINS],
        index=5
    )
    selected_col = label_to_col[selected_label]
    info         = DOMAINS[selected_col]
    m            = models[selected_col]
    ts           = m['ts']
    last         = ts.iloc[-1]
    pred, py, pq = predict_domain(selected_col, models)
    color        = info['color']
    mae          = m['mae']

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([2, 3])

    with left:
        beat     = mae < m['naive_mae']
        pill_cls = "pill-good" if beat else "pill-warn"
        pill_txt = "✅ Beats naive baseline" if beat else "⚠️ More data needed"
        lo       = str(int(max(0, pred - mae)))
        hi       = str(int(pred + mae))

        st.markdown(
            "<div class='pred-box' style='border-color:" + color + "33;'>"
            "<div class='pred-label'>Predicted patents — "
            + str(py) + " Q" + str(pq) + "</div>"
            "<div class='pred-number' style='color:" + color + "'>"
            + str(int(pred)) + "</div>"
            "<div class='pred-range'>likely range: "
            + lo + " – " + hi + " patents</div>"
            "<br><span class='" + pill_cls + "'>" + pill_txt + "</span>"
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-eyebrow'>MODEL DETAILS</div>",
                    unsafe_allow_html=True)

        last_yr = str(int(last['year']))
        last_q  = str(int(last['quarter']))
        last_ct = str(int(last['patent_count']))
        rows_detail = [
            ("Domain",             info['label']),
            ("Model",              m['model_name']),
            ("MAE",                "± " + str(mae) + " patents/quarter"),
            ("Naive MAE",          "± " + str(m['naive_mae']) + " (repeat-last baseline)"),
            ("RMSE",               str(m['rmse'])),
            ("R²",                 str(m['r2'])),
            ("Training quarters",  str(len(m['train']))),
            ("Test quarters",      str(len(m['test']))),
            ("Last data point",    last_yr + " Q" + last_q + " (" + last_ct + " patents)"),
        ]
        for k, v in rows_detail:
            info_row(k, v)

    with right:
        st.markdown("<div class='section-eyebrow'>RECENT TREND + FORECAST</div>",
                    unsafe_allow_html=True)
        recent  = ts.tail(16).copy()
        recent['t'] = recent['year'] + (recent['quarter'] - 1) / 4
        future_t    = py + (pq - 1) / 4

        fig, ax = dark_fig(8, 4.5)
        ax.plot(recent['t'], recent['patent_count'],
                color=color, linewidth=2.5, marker='o',
                markersize=5, label='Actual (recent)')
        ax.plot([recent['t'].iloc[-1], future_t],
                [recent['patent_count'].iloc[-1], pred],
                color='white', linewidth=1.5, linestyle='--', alpha=0.5)
        ax.scatter([future_t], [pred], color='white', s=100,
                   zorder=5, label='Predicted: ' + str(int(pred)))
        ax.fill_between(
            [future_t - 0.05, future_t + 0.05],
            [max(0, pred - mae)], [pred + mae],
            color='white', alpha=0.1
        )
        ax.axvline(x=recent['t'].iloc[-1] + 0.1, color='#30363d',
                   linestyle=':', linewidth=1)
        ax.set_xlabel('Year')
        ax.set_ylabel('Patents / Quarter')
        ax.set_title(info['label'] + ' — Last 16 Quarters + Next Quarter Forecast',
                     color='#e6edf3', fontsize=11, fontweight='bold', pad=10)
        ax.legend(facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#e6edf3', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if m['model_name'] == 'Random Forest':
            st.markdown(
                "<div class='section-eyebrow' style='margin-top:18px;'>"
                "WHAT DRIVES THIS PREDICTION</div>",
                unsafe_allow_html=True
            )
            imps = pd.Series(
                m['model'].feature_importances_, index=DOMAIN_FEATURES
            ).sort_values(ascending=True)
            fig2, ax2 = dark_fig(8, 3.5)
            ax2.barh(imps.index, imps.values, color=color, alpha=0.8, edgecolor='none')
            ax2.set_xlabel('Importance')
            ax2.set_title('Feature Importance (Random Forest)',
                          color='#e6edf3', fontsize=10, fontweight='bold', pad=8)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    # All domains bar chart
    st.markdown("<br><div class='section-eyebrow'>ALL DOMAINS — NEXT QUARTER</div>",
                unsafe_allow_html=True)

    ap_rows = []
    for c in DOMAINS:
        p, py2, pq2 = predict_domain(c, models)
        ap_rows.append({
            'Domain': DOMAINS[c]['label'],
            'Prediction': p,
            'MAE': models[c]['mae'],
            'Color': DOMAINS[c]['color']
        })
    ap_df = pd.DataFrame(ap_rows).sort_values('Prediction', ascending=True)
    ap_colors = [label_to_color[l] for l in ap_df['Domain']]

    fig, ax = dark_fig(10, 4)
    bars = ax.barh(ap_df['Domain'], ap_df['Prediction'],
                   color=ap_colors, edgecolor='none', alpha=0.9)
    for i, row in enumerate(ap_df.itertuples()):
        ax.barh(row.Domain, row.MAE, left=row.Prediction,
                color=ap_colors[i], alpha=0.15, edgecolor='none')
    ax.bar_label(bars, fmt='{:.0f}', padding=5,
                 color='#8b949e', fontsize=10, fontfamily='monospace')
    ax.set_xlabel('Predicted Patent Count')
    ax.set_title('Next Quarter Predictions — All Domains  (faint = ± MAE)',
                 color='#e6edf3', fontsize=11, fontweight='bold', pad=10)
    # Highlight selected domain
    labels_list = ap_df['Domain'].tolist()
    if selected_label in labels_list:
        bars[labels_list.index(selected_label)].set_edgecolor('white')
        bars[labels_list.index(selected_label)].set_linewidth(1.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRENDS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trends":
    st.markdown("## 📈 Historical Trends")
    st.markdown(
        "<div style='color:#8b949e;margin-bottom:24px;'>"
        "Quarterly patent activity per domain over time.</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='section-eyebrow'>QUARTERLY TIME SERIES — ALL DOMAINS</div>",
                unsafe_allow_html=True)

    fig, axes = dark_fig_multi(3, 3, 16, 11)
    axes_flat = axes.flatten()
    for i, (col_key, info) in enumerate(DOMAINS.items()):
        ts2     = models[col_key]['ts'].copy()
        ts2['t'] = ts2['year'] + (ts2['quarter'] - 1) / 4
        ax      = axes_flat[i]
        color   = info['color']
        ax.plot(ts2['t'], ts2['patent_count'],
                color=color, linewidth=2, marker='o', markersize=3)
        ax.fill_between(ts2['t'], ts2['patent_count'], alpha=0.1, color=color)
        ax.plot(ts2['t'], ts2['roll4_mean'],
                color='white', linewidth=1, linestyle='--',
                alpha=0.4, label='4-qtr avg')
        ax.set_title(info['label'], color='#e6edf3',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Year', fontsize=8)
        ax.set_ylabel('Patents/Qtr', fontsize=8)
        ax.legend(fontsize=7, facecolor='#0d1117', edgecolor='#30363d',
                  labelcolor='#8b949e')

    fig.suptitle('Quarterly Patent Counts by Technology Domain',
                 color='#e6edf3', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Single domain deep dive
    st.markdown("<br><div class='section-eyebrow'>SINGLE DOMAIN DEEP DIVE</div>",
                unsafe_allow_html=True)
    sel = st.selectbox("Select domain",
                       [DOMAINS[c]['label'] for c in DOMAINS],
                       index=5, key='trend_sel')
    sel_col  = label_to_col[sel]
    sel_info = DOMAINS[sel_col]
    ts3      = models[sel_col]['ts'].copy()
    ts3['t'] = ts3['year'] + (ts3['quarter'] - 1) / 4
    color3   = sel_info['color']

    fig, axes = dark_fig_multi(1, 2, 14, 4.5)
    axes[0].plot(ts3['t'], ts3['patent_count'],
                 color=color3, linewidth=2.5, marker='o',
                 markersize=4, label='Quarterly count')
    axes[0].plot(ts3['t'], ts3['roll4_mean'],
                 color='white', linewidth=1.5, linestyle='--',
                 alpha=0.5, label='4-qtr rolling avg')
    axes[0].set_title(sel + ' — Full History', color='#e6edf3',
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Patents / Quarter')
    axes[0].legend(facecolor='#161b22', edgecolor='#30363d',
                   labelcolor='#e6edf3', fontsize=9)

    ts_yoy  = ts3.dropna(subset=['yoy'])
    yoy_clr = [color3 if v >= 0 else '#f85149' for v in ts_yoy['yoy']]
    axes[1].bar(ts_yoy['t'], ts_yoy['yoy'] * 100,
                color=yoy_clr, edgecolor='none', alpha=0.85, width=0.2)
    axes[1].axhline(0, color='white', linewidth=1, linestyle='--', alpha=0.4)
    axes[1].set_title(sel + ' — Year-over-Year Growth (%)',
                      color='#e6edf3', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('YoY Growth (%)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<div class='section-eyebrow' style='margin-top:18px;'>"
                "DESCRIPTIVE STATISTICS</div>", unsafe_allow_html=True)
    stats_df = ts3['patent_count'].describe().round(2).reset_index()
    stats_df.columns = ['Statistic', 'Value']
    st.dataframe(stats_df, use_container_width=False, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL REPORT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Report":
    st.markdown("## 📊 Model Evaluation Report")
    st.markdown(
        "<div style='color:#8b949e;margin-bottom:24px;'>"
        "How well each domain model performs on held-out test data.</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='section-eyebrow'>PERFORMANCE SUMMARY</div>",
                unsafe_allow_html=True)

    rpt_rows = []
    for col_key, info in DOMAINS.items():
        m = models[col_key]
        rpt_rows.append({
            'Domain'       : info['label'],
            'Model'        : m['model_name'],
            'MAE'          : m['mae'],
            'Naive MAE'    : m['naive_mae'],
            'Improvement'  : round(m['naive_mae'] - m['mae'], 2),
            'RMSE'         : m['rmse'],
            'R2'           : m['r2'],
            'Test Quarters': len(m['test']),
            'Beats Naive'  : '✅' if m['mae'] < m['naive_mae'] else '❌',
        })

    rpt_df = pd.DataFrame(rpt_rows).sort_values('Improvement', ascending=False)

    def color_impr(val):
        if isinstance(val, float):
            if val > 0: return 'color: #3fb950; font-weight: bold'
            if val < 0: return 'color: #f85149'
        return ''

    # Avoid using the pandas Styler (which requires jinja2); instead
    # format numeric columns as strings for display in Streamlit.
    rpt_disp = rpt_df.copy()
    rpt_disp['MAE'] = rpt_disp['MAE'].map(lambda v: f"{v:.2f}")
    rpt_disp['Naive MAE'] = rpt_disp['Naive MAE'].map(lambda v: f"{v:.2f}")
    rpt_disp['Improvement'] = rpt_disp['Improvement'].map(lambda v: f"{v:+.2f}")
    rpt_disp['RMSE'] = rpt_disp['RMSE'].map(lambda v: f"{v:.2f}")
    rpt_disp['R2'] = rpt_disp['R2'].map(lambda v: f"{v:.3f}")

    st.dataframe(
        rpt_disp,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "**R² note:** Negative R² on small test sets (< 10 rows) is expected — "
        "one bad prediction skews the score. Always check MAE vs Naive MAE "
        "to judge real-world usefulness."
    )

    # Actual vs Predicted grid
    st.markdown("<br><div class='section-eyebrow'>ACTUAL VS PREDICTED — ALL DOMAINS</div>",
                unsafe_allow_html=True)

    fig, axes = dark_fig_multi(3, 3, 16, 12)
    axes_flat = axes.flatten()
    for i, (col_key, info) in enumerate(DOMAINS.items()):
        m2   = models[col_key]
        ax   = axes_flat[i]
        tr2  = m2['train'].copy()
        te2  = m2['test'].copy()
        tr2['t'] = tr2['year'] + (tr2['quarter'] - 1) / 4
        te2['t'] = te2['year'] + (te2['quarter'] - 1) / 4
        color2 = info['color']

        ax.plot(tr2['t'], tr2['target'],
                color=color2, alpha=0.2, linewidth=1.5, label='Train')
        ax.plot(te2['t'], te2['target'],
                color=color2, linewidth=2, marker='o', markersize=4, label='Actual')
        ax.plot(te2['t'], m2['y_pred'],
                color='white', linewidth=2, linestyle='--',
                marker='x', markersize=5, label='Predicted')
        ax.axvspan(te2['t'].min(), te2['t'].max(), alpha=0.06, color=color2)
        title_str = (info['label'] + "\nMAE=" + str(m2['mae'])
                     + " | Naive=" + str(m2['naive_mae'])
                     + " | R²=" + str(m2['r2']))
        ax.set_title(title_str, fontsize=9, fontweight='bold', color='#e6edf3')
        ax.set_xlabel('Year', fontsize=8)
        ax.set_ylabel('Patents', fontsize=8)
        ax.legend(fontsize=7, facecolor='#0d1117', edgecolor='#30363d',
                  labelcolor='#e6edf3')

    fig.suptitle('Actual vs Predicted — Test Period (shaded)',
                 color='#e6edf3', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Residual analysis
    st.markdown("<br><div class='section-eyebrow'>RESIDUAL ANALYSIS</div>",
                unsafe_allow_html=True)
    sel4 = st.selectbox("Domain", [DOMAINS[c]['label'] for c in DOMAINS],
                        index=5, key='report_sel')
    sel4_col   = label_to_col[sel4]
    sel4_color = DOMAINS[sel4_col]['color']
    m4         = models[sel4_col]
    residuals  = m4['test']['target'].values - m4['y_pred']

    fig, axes = dark_fig_multi(1, 2, 12, 4)
    lo4 = min(float(m4['test']['target'].min()), float(m4['y_pred'].min()))
    hi4 = max(float(m4['test']['target'].max()), float(m4['y_pred'].max()))
    axes[0].scatter(m4['test']['target'], m4['y_pred'],
                    color=sel4_color, alpha=0.8, s=60, zorder=3)
    axes[0].plot([lo4, hi4], [lo4, hi4], 'w--', linewidth=1.5,
                 alpha=0.6, label='Perfect prediction')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(sel4 + ' — Actual vs Predicted',
                      color='#e6edf3', fontsize=10, fontweight='bold')
    axes[0].legend(facecolor='#161b22', edgecolor='#30363d',
                   labelcolor='#e6edf3', fontsize=8)

    axes[1].hist(residuals, bins=min(15, len(residuals)),
                 color=sel4_color, edgecolor='none', alpha=0.8)
    axes[1].axvline(0, color='white', linewidth=1.5, linestyle='--', alpha=0.7)
    mean_err = residuals.mean()
    axes[1].axvline(mean_err, color='#d29922', linewidth=1.5,
                    label='Mean error: ' + str(round(mean_err, 1)))
    axes[1].set_xlabel('Error (Actual − Predicted)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(sel4 + ' — Residual Distribution',
                      color='#e6edf3', fontsize=10, fontweight='bold')
    axes[1].legend(facecolor='#161b22', edgecolor='#30363d',
                   labelcolor='#e6edf3', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Error",   str(round(float(residuals.mean()), 2)))
    c2.metric("Std of Error", str(round(float(residuals.std()), 2)))
    c3.metric("Max Error",    str(round(float(np.abs(residuals).max()), 2)))
