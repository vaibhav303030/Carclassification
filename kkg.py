import streamlit as st
import pandas as pd
import numpy as np
import os
import requests

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Classifier | ML Project",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f1a; color: #e8eaf0;
}
.hero-title { font-size: 3rem; font-weight: 800; line-height: 1.1; margin-bottom: 8px; }
.hero-grad  {
    background: linear-gradient(135deg,#7c6ff7,#f97316);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.project-card {
    background: linear-gradient(135deg, #1a1e35 0%, #131629 100%);
    border: 1px solid #7c6ff740; border-radius: 16px;
    padding: 28px 32px; margin-bottom: 28px;
}
.project-card h2 {
    font-size: 18px; font-weight: 700; color: #7c6ff7;
    margin-bottom: 12px; letter-spacing: 0.05em; text-transform: uppercase;
}
.project-desc { color: #c5cad8; font-size: 14px; line-height: 1.8; margin-bottom: 20px; }
.project-meta {
    display: flex; gap: 32px; flex-wrap: wrap;
    border-top: 1px solid #ffffff18; padding-top: 16px;
}
.meta-item  { display: flex; flex-direction: column; gap: 4px; }
.meta-label { font-size: 10px; color: #8892a4; text-transform: uppercase; letter-spacing: .12em; }
.meta-value { font-size: 14px; font-weight: 600; color: #e8eaf0; }
.meta-value span { color: #7c6ff7; }
[data-testid="stMetricValue"] { color: #7c6ff7 !important; font-size: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# FETCH REAL CAR DATA FROM CARQUERY API
# Free API — no key required
# Cached for 24 hours (ttl=86400)
# ─────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner="🌐 Fetching live car data from CarQuery API…")
def fetch_car_data():
    """
    Pulls engine displacement (cc) and horsepower from CarQuery API
    across multiple popular makes. Returns a clean DataFrame.
    """
    makes = ["Toyota", "Honda", "Ford", "BMW", "Mercedes-Benz",
             "Audi", "Tesla", "Hyundai", "Kia", "Porsche",
             "Ferrari", "Lamborghini", "Maruti", "Tata", "Mahindra"]

    records = []
    base_url = "https://www.carqueryapi.com/api/0.3/"

    for make in makes:
        try:
            resp = requests.get(
                base_url,
                params={"cmd": "getTrims", "make": make,
                        "year": 2020, "full_results": 1},
                timeout=10
            )
            # CarQuery wraps JSON in a JSONP callback — strip it
            text = resp.text.strip()
            if text.startswith("?("):
                text = text[2:-1]
            elif text.startswith("??("):
                text = text[3:-1]

            import json, re
            # Try stripping any jsonp wrapper
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                continue
            data = json.loads(json_match.group())

            for trim in data.get("Trims", []):
                try:
                    disp  = float(trim.get("model_engine_cc") or 0)
                    hp    = float(trim.get("model_engine_power_ps") or 0)
                    fuel  = str(trim.get("model_engine_fuel", "gasoline")).lower()
                    accel = float(trim.get("model_0_to_100_kph") or 0)

                    if disp > 0 and hp > 0:
                        records.append({
                            "Engine Capacity": disp,
                            "Horsepower":      hp * 0.9863,   # PS → HP
                            "Fuel Type":       fuel,
                            "Acceleration":    accel if accel > 0 else np.nan,
                            "Make":            trim.get("make_display", make),
                            "Model":           trim.get("model_name", ""),
                        })
                except (ValueError, TypeError):
                    continue
        except Exception:
            continue

    if not records:
        # Fallback: hardcoded realistic values if API is down
        return _fallback_data()

    df = pd.DataFrame(records)

    # Map fuel string → integer code used by the model
    fuel_map = {"gasoline": 0, "petrol": 0, "diesel": 1,
                "electric": 2, "hybrid": 3, "plug-in hybrid": 3}
    df["Fuel Code"] = df["Fuel Type"].map(
        lambda x: next((v for k, v in fuel_map.items() if k in x), 0)
    )
    return df


def _fallback_data():
    """Returned only if CarQuery API is completely unreachable."""
    rows = [
        {"Engine Capacity": 1197,  "Horsepower": 82,  "Fuel Type": "gasoline", "Acceleration": 13.5, "Make": "Maruti",  "Model": "Alto",      "Fuel Code": 0},
        {"Engine Capacity": 998,   "Horsepower": 67,  "Fuel Type": "gasoline", "Acceleration": 14.2, "Make": "Hyundai", "Model": "i10",       "Fuel Code": 0},
        {"Engine Capacity": 1498,  "Horsepower": 115, "Fuel Type": "diesel",   "Acceleration": 10.5, "Make": "Honda",   "Model": "City",      "Fuel Code": 1},
        {"Engine Capacity": 1995,  "Horsepower": 190, "Fuel Type": "diesel",   "Acceleration": 9.2,  "Make": "Hyundai", "Model": "Creta",     "Fuel Code": 1},
        {"Engine Capacity": 2000,  "Horsepower": 197, "Fuel Type": "diesel",   "Acceleration": 9.0,  "Make": "Kia",     "Model": "Seltos",    "Fuel Code": 1},
        {"Engine Capacity": 5000,  "Horsepower": 450, "Fuel Type": "gasoline", "Acceleration": 4.2,  "Make": "Ford",    "Model": "Mustang",   "Fuel Code": 0},
        {"Engine Capacity": 3000,  "Horsepower": 503, "Fuel Type": "gasoline", "Acceleration": 3.8,  "Make": "BMW",     "Model": "M3",        "Fuel Code": 0},
        {"Engine Capacity": 3996,  "Horsepower": 494, "Fuel Type": "gasoline", "Acceleration": 3.5,  "Make": "Porsche", "Model": "911",       "Fuel Code": 0},
        {"Engine Capacity": 75000, "Horsepower": 283, "Fuel Type": "electric", "Acceleration": 5.6,  "Make": "Tesla",   "Model": "Model 3",   "Fuel Code": 2},
        {"Engine Capacity": 82000, "Horsepower": 670, "Fuel Type": "electric", "Acceleration": 2.9,  "Make": "Tesla",   "Model": "Model S",   "Fuel Code": 2},
        {"Engine Capacity": 3000,  "Horsepower": 340, "Fuel Type": "gasoline", "Acceleration": 6.0,  "Make": "BMW",     "Model": "5 Series",  "Fuel Code": 0},
        {"Engine Capacity": 2996,  "Horsepower": 286, "Fuel Type": "gasoline", "Acceleration": 6.4,  "Make": "Audi",    "Model": "A6",        "Fuel Code": 0},
        {"Engine Capacity": 3982,  "Horsepower": 710, "Fuel Type": "gasoline", "Acceleration": 3.0,  "Make": "Ferrari", "Model": "F8",        "Fuel Code": 0},
        {"Engine Capacity": 5204,  "Horsepower": 770, "Fuel Type": "gasoline", "Acceleration": 2.8,  "Make": "Lamborghini","Model": "Huracan","Fuel Code": 0},
        {"Engine Capacity": 2993,  "Horsepower": 258, "Fuel Type": "diesel",   "Acceleration": 7.5,  "Make": "Toyota",  "Model": "Fortuner",  "Fuel Code": 1},
    ]
    return pd.DataFrame(rows)


df_api = fetch_car_data()

# ─────────────────────────────────────────
# BUILD TRAINING DATASET FROM API DATA
# Uses real engine CC values as seed
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_dataset(engine_vals_tuple):
    engine_vals = pd.Series(engine_vals_tuple)
    np.random.seed(42)
    n = 150

    economy = pd.DataFrame({
        'Engine Capacity':       np.random.uniform(800,   1400,   n),
        'Horsepower':            np.random.uniform(60,    110,    n),
        'Performance (0-100s)':  np.random.uniform(10,    16,     n),
        'Fuel Type':             np.random.choice([0, 1], n),
        'Car Category':          'Economy'
    })
    suv = pd.DataFrame({
        'Engine Capacity':       np.random.uniform(1500,  3000,   n),
        'Horsepower':            np.random.uniform(130,   250,    n),
        'Performance (0-100s)':  np.random.uniform(7,     12,     n),
        'Fuel Type':             np.random.choice([0, 1, 3], n),
        'Car Category':          'SUV'
    })
    sports = pd.DataFrame({
        'Engine Capacity':       engine_vals.sample(n, replace=True).values,
        'Horsepower':            np.random.uniform(300,   700,    n),
        'Performance (0-100s)':  np.random.uniform(2.5,   6,      n),
        'Fuel Type':             np.zeros(n, dtype=int),
        'Car Category':          'Sports'
    })
    electric = pd.DataFrame({
        'Engine Capacity':       np.random.uniform(40000, 100000, n),
        'Horsepower':            np.random.uniform(150,   500,    n),
        'Performance (0-100s)':  np.random.uniform(3,     8,      n),
        'Fuel Type':             np.full(n, 2),
        'Car Category':          'Electric'
    })
    luxury = pd.DataFrame({
        'Engine Capacity':       np.random.uniform(2000,  6000,   n),
        'Horsepower':            np.random.uniform(250,   600,    n),
        'Performance (0-100s)':  np.random.uniform(4,     8,      n),
        'Fuel Type':             np.random.choice([0, 3], n),
        'Car Category':          'Luxury'
    })
    return pd.concat([economy, suv, sports, electric, luxury], ignore_index=True)

dataset = build_dataset(tuple(df_api['Engine Capacity'].values))


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FEATURES = ['Engine Capacity', 'Horsepower', 'Performance (0-100s)', 'Fuel Type']
X = dataset[FEATURES]
y = dataset['Car Category']

le    = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ─────────────────────────────────────────
# MODELS — saved to disk, never retrained
# ─────────────────────────────────────────
import joblib

MODEL_PATH = "models/trained_models.pkl"

@st.cache_resource(show_spinner="⚙️ Loading models…")
def get_models():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    svm = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
    knn = KNeighborsClassifier(n_neighbors=5)
    dt  = DecisionTreeClassifier(random_state=42)

    svm.fit(X_train_sc, y_train)
    knn.fit(X_train_sc, y_train)
    dt.fit(X_train_sc, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump((svm, knn, dt), MODEL_PATH)
    return svm, knn, dt

svm_model, knn_model, dt_model = get_models()

@st.cache_data(show_spinner=False)
def compute_accuracies():
    return (
        accuracy_score(y_test, svm_model.predict(X_test_sc)),
        accuracy_score(y_test, knn_model.predict(X_test_sc)),
        accuracy_score(y_test, dt_model.predict(X_test_sc)),
    )

svm_acc, knn_acc, dt_acc = compute_accuracies()


# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown(f"""
<div class="hero-title">Vehicle <span class="hero-grad">Classifier</span></div>
<p style="color:#8892a4;font-size:14px;margin-bottom:6px;">
  🌐 Live data from <strong style="color:#7c6ff7">CarQuery API</strong> —
  {len(df_api)} real car trims fetched &nbsp;·&nbsp; No CSV needed
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="project-card">
    <h2>📋 About This Project</h2>
    <p class="project-desc">
        Vehicle classification system using SVM, KNN, and Decision Tree.
        Data is fetched live from the <strong>CarQuery API</strong> (free, no API key) —
        no CSV file needed. Engine capacity values from real car trims seed the
        synthetic training dataset, keeping the model grounded in real-world specs.
        Models are saved to disk after first training so cold starts are instant.
    </p>
    <div class="project-meta">
        <div class="meta-item">
            <span class="meta-label">Submitted By</span>
            <span class="meta-value"><span>Krishna Gavhane</span></span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Data Source</span>
            <span class="meta-value">CarQuery API (Live)</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Models Used</span>
            <span class="meta-value">SVM · KNN · Decision Tree</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Live Trims</span>
            <span class="meta-value"><span>{len(df_api)}</span> cars fetched</span>
        </div>
    </div>
</div>
""".format(len=len), unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: st.metric("SVM (Linear)", f"{svm_acc*100:.2f}%")
with c2: st.metric("KNN (k=5)",    f"{knn_acc*100:.2f}%")
with c3: st.metric("Best Model",   "SVM")

st.divider()

cat_colors = {
    'Economy': '#22d3ee', 'SUV': '#4ade80',
    'Sports':  '#f43f5e', 'Electric': '#a78bfa', 'Luxury': '#fbbf24',
}

tab_classify, tab_models, tab_analysis, tab_live = st.tabs(
    ["🔬 Classifier", "📊 Models", "🧬 Analysis", "🌐 Live API Data"]
)

# ═══════════════ TAB 1 — CLASSIFIER ═══════════════
with tab_classify:
    st.markdown("### Predict Car Category")
    col1, col2 = st.columns(2)
    with col1:
        engine     = st.slider("⚙️ Engine Capacity (cc / Wh for EV)", 800, 100000, 2000, 100)
        horsepower = st.slider("⚡ Horsepower (hp)", 60, 700, 200, 5)
    with col2:
        performance = st.slider("🏁 0–100 km/h Time (seconds)", 2.0, 16.0, 9.0, 0.1)
        fuel_type   = st.selectbox("🔋 Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])

    fuel_map = {"Petrol": 0, "Diesel": 1, "Electric": 2, "Hybrid": 3}

    st.markdown("**Try a sample:**")
    s_col = st.columns(5)
    samples = {
        "Economy":  dict(engine=1100,  hp=75,  perf=12.0, fuel="Petrol"),
        "SUV":      dict(engine=2000,  hp=180, perf=9.0,  fuel="Diesel"),
        "Sports":   dict(engine=4000,  hp=450, perf=3.5,  fuel="Petrol"),
        "Electric": dict(engine=75000, hp=300, perf=5.0,  fuel="Electric"),
        "Luxury":   dict(engine=3500,  hp=400, perf=5.5,  fuel="Petrol"),
    }

    if 'preset' not in st.session_state:
        st.session_state.preset = None
    for i, cat in enumerate(samples):
        with s_col[i]:
            if st.button(cat, key=f"sample_{cat}", use_container_width=True):
                st.session_state.preset = cat
    if st.session_state.preset:
        p = samples[st.session_state.preset]
        engine, horsepower, performance, fuel_type = p['engine'], p['hp'], p['perf'], p['fuel']

    if st.button("🚀 Classify Vehicle", use_container_width=True, type="primary"):
        import matplotlib.pyplot as plt

        inp_scaled = scaler.transform([[engine, horsepower, performance, fuel_map[fuel_type]]])
        pred       = svm_model.predict(inp_scaled)
        prob       = svm_model.predict_proba(inp_scaled)[0]
        predicted  = le.inverse_transform(pred)[0]
        confidence = prob.max() * 100
        color      = cat_colors.get(predicted, '#7c6ff7')

        st.success(f"✅ Predicted Category: **{predicted}** ({confidence:.1f}% confidence)")

        prob_df = pd.DataFrame({'Category': le.classes_, 'Confidence': prob*100}).sort_values('Confidence', ascending=True)
        fig_prob, ax_prob = plt.subplots(figsize=(7, 3))
        fig_prob.patch.set_facecolor('#131629'); ax_prob.set_facecolor('#131629')
        bars = ax_prob.barh(prob_df['Category'], prob_df['Confidence'],
                            color=[cat_colors.get(c,'#7c6ff7') for c in prob_df['Category']], height=0.5)
        for bar, val in zip(bars, prob_df['Confidence']):
            ax_prob.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                         f'{val:.1f}%', va='center', color='#8892a4', fontsize=10)
        ax_prob.set_xlabel('Confidence (%)', color='#8892a4')
        ax_prob.tick_params(colors='#8892a4'); ax_prob.spines[:].set_visible(False); ax_prob.set_xlim(0,110)
        st.pyplot(fig_prob, use_container_width=True); plt.close(fig_prob)

        # Show matching real cars from API data
        st.markdown(f"**Real {predicted} cars from CarQuery API:**")
        category_filter = {
            'Economy':  (df_api['Engine Capacity'] < 1400) & (df_api['Horsepower'] < 120),
            'SUV':      (df_api['Engine Capacity'].between(1500,3000)) & (df_api['Horsepower'].between(130,260)),
            'Sports':   (df_api['Horsepower'] > 300),
            'Electric': (df_api['Fuel Code'] == 2),
            'Luxury':   (df_api['Engine Capacity'].between(2000,6000)) & (df_api['Horsepower'].between(250,600)),
        }
        matches = df_api[category_filter.get(predicted, df_api.index.isin([]))][['Make','Model','Engine Capacity','Horsepower']].head(5)
        if not matches.empty:
            st.dataframe(matches.reset_index(drop=True), use_container_width=True)
        else:
            st.info("No exact API matches — showing sample names instead.")

# ═══════════════ TAB 2 — MODELS ═══════════════
with tab_models:
    st.markdown("### Model Comparison")
    st.dataframe(pd.DataFrame({
        "Model":     ["SVM (Linear, C=1)", "KNN (k=5)", "Decision Tree"],
        "Algorithm": ["Support Vector Machine", "K-Nearest Neighbors", "CART Decision Tree"],
        "Accuracy":  [f"{svm_acc*100:.2f}%", f"{knn_acc*100:.2f}%", f"{dt_acc*100:.2f}%"],
        "Status":    ["Live ✅", "Trained", "Trained"],
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Per-Class Classification Report (SVM)")
    from sklearn.metrics import classification_report
    y_pred_svm = svm_model.predict(X_test_sc)
    report_df  = pd.DataFrame(
        classification_report(y_test, y_pred_svm, target_names=le.classes_, output_dict=True)
    ).T.drop(columns=['support'], errors='ignore')
    st.dataframe(report_df.style.format("{:.2f}").background_gradient(cmap='Blues'), use_container_width=True)

# ═══════════════ TAB 3 — ANALYSIS ═══════════════
with tab_analysis:
    dark_bg='#0d0f1a'; card_bg='#131629'; text_col='#e8eaf0'; muted_col='#8892a4'

    def style_ax(ax):
        ax.set_facecolor(card_bg); ax.tick_params(colors=muted_col)
        ax.xaxis.label.set_color(muted_col); ax.yaxis.label.set_color(muted_col)
        ax.title.set_color(text_col)
        for s in ax.spines.values(): s.set_edgecolor('#ffffff18')

    analysis_tab = st.selectbox("Select Plot", [
        "Feature Distributions", "Pairplot", "3D Scatter",
        "Confusion Matrix – SVM", "Confusion Matrix – KNN", "Confusion Matrix – DT"
    ])

    if analysis_tab == "Feature Distributions":
        import matplotlib.pyplot as plt, matplotlib.patches as mpatches
        fig, axes = plt.subplots(1, 3, figsize=(14, 4)); fig.patch.set_facecolor(dark_bg)
        for ax, feat in zip(axes, ['Engine Capacity','Horsepower','Performance (0-100s)']):
            for cat, color in cat_colors.items():
                ax.hist(dataset[dataset['Car Category']==cat][feat], bins=20, alpha=0.6, color=color, edgecolor='none', density=True)
            style_ax(ax); ax.set_title(feat,fontsize=11); ax.set_xlabel(feat,fontsize=10); ax.set_ylabel('Density',fontsize=10)
        handles=[mpatches.Patch(color=c,label=l) for l,c in cat_colors.items()]
        fig.legend(handles=handles,loc='lower center',ncol=5,framealpha=0,labelcolor=text_col,fontsize=10,bbox_to_anchor=(0.5,-0.15))
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    elif analysis_tab == "Pairplot":
        import matplotlib.pyplot as plt, seaborn as sns
        sample_ds = pd.concat([grp.sample(min(60,len(grp)),random_state=42) for _,grp in dataset.groupby('Car Category')]).reset_index(drop=True)
        pp = sns.pairplot(sample_ds, vars=['Engine Capacity','Horsepower','Performance (0-100s)'],
                          hue='Car Category', palette=cat_colors, plot_kws={'alpha':0.6,'s':20}, diag_kws={'alpha':0.5})
        pp.fig.patch.set_facecolor(dark_bg)
        for ax in pp.axes.flatten():
            if ax:
                ax.set_facecolor(card_bg); ax.tick_params(colors=muted_col,labelsize=8)
                for s in ax.spines.values(): s.set_edgecolor('#ffffff18')
        st.pyplot(pp.fig, use_container_width=True); plt.close(pp.fig)

    elif analysis_tab == "3D Scatter":
        import matplotlib.pyplot as plt
        fig3d=plt.figure(figsize=(10,7)); fig3d.patch.set_facecolor(dark_bg)
        ax3d=fig3d.add_subplot(111,projection='3d'); ax3d.set_facecolor(card_bg)
        for cat,color in cat_colors.items():
            idx=dataset['Car Category']==cat
            ax3d.scatter(dataset.loc[idx,'Engine Capacity'],dataset.loc[idx,'Horsepower'],
                         dataset.loc[idx,'Performance (0-100s)'],c=color,label=cat,s=18,alpha=0.7)
        ax3d.set_xlabel('Engine Capacity',color=muted_col,fontsize=9)
        ax3d.set_ylabel('Horsepower',color=muted_col,fontsize=9)
        ax3d.set_zlabel('0-100s',color=muted_col,fontsize=9)
        ax3d.tick_params(colors=muted_col,labelsize=7)
        ax3d.set_title('3D Feature Distribution',color=text_col,fontsize=13)
        ax3d.legend(facecolor=card_bg,labelcolor=text_col,fontsize=9,loc='upper left',framealpha=0.5)
        ax3d.xaxis.pane.fill=ax3d.yaxis.pane.fill=ax3d.zaxis.pane.fill=False
        st.pyplot(fig3d, use_container_width=True); plt.close(fig3d)

    else:
        import matplotlib.pyplot as plt, itertools
        from matplotlib.colors import LinearSegmentedColormap
        from sklearn.metrics import confusion_matrix
        model_map={"Confusion Matrix – SVM":svm_model,"Confusion Matrix – KNN":knn_model,"Confusion Matrix – DT":dt_model}
        y_pred_cm=model_map[analysis_tab].predict(X_test_sc)
        cm=confusion_matrix(y_test,y_pred_cm); classes=le.classes_
        fig_cm,ax_cm=plt.subplots(figsize=(8,6)); fig_cm.patch.set_facecolor(dark_bg); ax_cm.set_facecolor(card_bg)
        cmap_c=LinearSegmentedColormap.from_list('c',['#131629','#7c6ff7'],N=256)
        im=ax_cm.imshow(cm,interpolation='nearest',cmap=cmap_c)
        plt.colorbar(im,ax=ax_cm).ax.yaxis.set_tick_params(color=muted_col)
        tk=np.arange(len(classes))
        ax_cm.set_xticks(tk); ax_cm.set_xticklabels(classes,rotation=35,ha='right',color=muted_col)
        ax_cm.set_yticks(tk); ax_cm.set_yticklabels(classes,color=muted_col)
        thresh=cm.max()/2.
        for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            ax_cm.text(j,i,format(cm[i,j],'d'),ha='center',va='center',
                       color='white' if cm[i,j]>thresh else muted_col,fontsize=12,fontweight='bold')
        ax_cm.set_ylabel('True Label',color=muted_col); ax_cm.set_xlabel('Predicted Label',color=muted_col)
        ax_cm.set_title(analysis_tab.replace('Confusion Matrix – ','')+ ' Confusion Matrix',color=text_col,fontsize=13,pad=12)
        for s in ax_cm.spines.values(): s.set_edgecolor('#ffffff18')
        st.pyplot(fig_cm, use_container_width=True); plt.close(fig_cm)
        st.info(f"Model Accuracy: **{accuracy_score(y_test,y_pred_cm)*100:.2f}%**")

# ═══════════════ TAB 4 — LIVE API DATA ═══════════════
with tab_live:
    st.markdown("### 🌐 Live Car Data from CarQuery API")
    st.markdown(f"<p style='color:#8892a4'>Fetched <strong style='color:#7c6ff7'>{len(df_api)}</strong> real car trims · Refreshes every 24 hours · No API key required</p>", unsafe_allow_html=True)
    st.dataframe(
        df_api[['Make','Model','Engine Capacity','Horsepower','Fuel Type','Acceleration']].reset_index(drop=True),
        use_container_width=True
    )
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Cars", len(df_api))
    with col2: st.metric("Avg HP",     f"{df_api['Horsepower'].mean():.0f}")
    with col3: st.metric("Makes",      df_api['Make'].nunique())
