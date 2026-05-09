import streamlit as st
import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────
# PAGE CONFIG  (must be first st call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Classifier | ML Project",
    page_icon="🚗",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f1a;
    color: #e8eaf0;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: #ffffff0a; border: 1px solid #ffffff18;
    border-radius: 20px; padding: 6px 18px;
    font-size: 12px; color: #8892a4; letter-spacing: .08em;
    margin-bottom: 12px;
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
.meta-item { display: flex; flex-direction: column; gap: 4px; }
.meta-label { font-size: 10px; color: #8892a4; text-transform: uppercase; letter-spacing: .12em; }
.meta-value { font-size: 14px; font-weight: 600; color: #e8eaf0; }
.meta-value span { color: #7c6ff7; }
[data-testid="stMetricValue"] { color: #7c6ff7 !important; font-size: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD & CLEAN DATASET
# ─────────────────────────────────────────
@st.cache_data(show_spinner="📂 Loading dataset…")
def load_data():
    df = pd.read_csv('Cars Datasets 2025.csv', encoding='latin1')

    def extract_cc(x):
        if pd.isna(x): return np.nan
        s = str(x).lower().replace(',','').replace('cc','').strip()
        try: return float(s)
        except: return np.nan

    df['Engine Capacity'] = df['CC/Battery Capacity'].apply(extract_cc)
    return df.dropna(subset=['Engine Capacity'])

df_cars = load_data()


# ─────────────────────────────────────────
# BUILD SYNTHETIC DATASET
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

# Pass tuple — required for cache hashing
dataset = build_dataset(tuple(df_cars['Engine Capacity'].values))


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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
# MODELS — saved to disk with joblib
#
# KEY OPTIMIZATION:
#   First run  → trains and saves to models/trained_models.pkl
#   Every cold start after → loads from file in ~0.01s instead of ~8s
# ─────────────────────────────────────────
import joblib

MODEL_PATH = "models/trained_models.pkl"

@st.cache_resource(show_spinner="⚙️ Loading models…")
def get_models():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)   # ← instant on cold start

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


# ─────────────────────────────────────────
# ACCURACIES
# ─────────────────────────────────────────
from sklearn.metrics import accuracy_score

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
st.markdown("""
<div class="hero-badge">🚗 SVM POWERED &bull; CARS DATASET 2025</div>
<div class="hero-title">Vehicle <span class="hero-grad">Classifier</span></div>
<p style="color:#8892a4;font-size:15px;max-width:560px;margin-bottom:24px;line-height:1.6;">
Enter your car's specs and our ML model will instantly identify which of the
<strong style="color:#e8eaf0">5 categories</strong> it belongs to.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="project-card">
    <h2>📋 About This Project</h2>
    <p class="project-desc">
        This project presents an intelligent vehicle classification system built using supervised machine learning
        algorithms — Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Decision Tree — trained on a
        synthetic dataset derived from the Cars Dataset 2025. The system classifies any vehicle into one of five
        categories (Economy, SUV, Sports, Electric, or Luxury) based on engine capacity, horsepower, fuel type,
        and acceleration performance.
    </p>
    <div class="project-meta">
        <div class="meta-item">
            <span class="meta-label">Submitted By</span>
            <span class="meta-value"><span>Krishna Gavhane</span></span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Dataset</span>
            <span class="meta-value">Cars Dataset 2025</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Models Used</span>
            <span class="meta-value">SVM · KNN · Decision Tree</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Categories</span>
            <span class="meta-value">5 Vehicle Classes</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: st.metric("SVM (Linear)", f"{svm_acc*100:.2f}%")
with c2: st.metric("KNN (k=5)",    f"{knn_acc*100:.2f}%")
with c3: st.metric("Best Model",   "SVM")

st.divider()

cat_colors = {
    'Economy':  '#22d3ee',
    'SUV':      '#4ade80',
    'Sports':   '#f43f5e',
    'Electric': '#a78bfa',
    'Luxury':   '#fbbf24',
}

tab_classify, tab_models, tab_analysis = st.tabs(
    ["🔬 Classifier", "📊 Models", "🧬 Analysis"]
)

# ═══════════════ TAB 1 — CLASSIFIER ═══════════════
with tab_classify:
    st.markdown("### Predict Car Category")
    st.markdown("<p style='color:#8892a4;font-size:13px;margin-bottom:20px'>Adjust the sliders or type values directly</p>", unsafe_allow_html=True)

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

    st.markdown("")

    if st.button("🚀 Classify Vehicle", use_container_width=True, type="primary"):
        import matplotlib.pyplot as plt

        inp_scaled = scaler.transform([[engine, horsepower, performance, fuel_map[fuel_type]]])
        pred       = svm_model.predict(inp_scaled)
        prob       = svm_model.predict_proba(inp_scaled)[0]
        predicted  = le.inverse_transform(pred)[0]
        confidence = prob.max() * 100
        color      = cat_colors.get(predicted, '#7c6ff7')

        st.success(f"✅ Predicted Category: **{predicted}** ({confidence:.1f}% confidence)")

        prob_df = pd.DataFrame({
            'Category':   le.classes_,
            'Confidence': prob * 100
        }).sort_values('Confidence', ascending=True)

        fig_prob, ax_prob = plt.subplots(figsize=(7, 3))
        fig_prob.patch.set_facecolor('#131629')
        ax_prob.set_facecolor('#131629')
        bars = ax_prob.barh(prob_df['Category'], prob_df['Confidence'],
                            color=[cat_colors.get(c,'#7c6ff7') for c in prob_df['Category']], height=0.5)
        for bar, val in zip(bars, prob_df['Confidence']):
            ax_prob.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                         f'{val:.1f}%', va='center', color='#8892a4', fontsize=10)
        ax_prob.set_xlabel('Confidence (%)', color='#8892a4')
        ax_prob.tick_params(colors='#8892a4')
        ax_prob.spines[:].set_visible(False)
        ax_prob.set_xlim(0, 110)
        st.pyplot(fig_prob, use_container_width=True)
        plt.close(fig_prob)

        cat_examples = {
            'Economy':  ['Maruti Alto', 'Hyundai i10', 'Tata Nano', 'Renault Kwid', 'Chevrolet Beat'],
            'SUV':      ['Hyundai Creta', 'Mahindra Scorpio', 'Toyota Fortuner', 'Kia Seltos', 'MG Hector'],
            'Sports':   ['Ford Mustang', 'BMW M3', 'Porsche 911', 'Audi TT', 'Ferrari SF90'],
            'Electric': ['Tata Nexon EV', 'MG ZS EV', 'Tesla Model 3', 'Hyundai Ioniq 5', 'Nissan Leaf'],
            'Luxury':   ['Mercedes E-Class', 'BMW 7 Series', 'Audi A8', 'Jaguar XF', 'Volvo S90'],
        }
        st.markdown(f"**Example {predicted} cars:**")
        ex_cols = st.columns(5)
        for i, car in enumerate(cat_examples.get(predicted, [])):
            with ex_cols[i]:
                st.markdown(
                    f"<div style='background:#1a1e35;border:1px solid #ffffff18;"
                    f"border-top:3px solid {color};border-radius:8px;padding:10px;"
                    f"text-align:center;font-size:12px;color:#e8eaf0'>🚗<br>{car}</div>",
                    unsafe_allow_html=True
                )

# ═══════════════ TAB 2 — MODELS ═══════════════
with tab_models:
    st.markdown("### Model Comparison")
    st.markdown("<p style='color:#8892a4;font-size:13px'>All classifiers trained on 80% of the synthetic Cars Dataset 2025</p>", unsafe_allow_html=True)

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
    dark_bg   = '#0d0f1a'
    card_bg   = '#131629'
    text_col  = '#e8eaf0'
    muted_col = '#8892a4'

    def style_ax(ax):
        ax.set_facecolor(card_bg)
        ax.tick_params(colors=muted_col)
        ax.xaxis.label.set_color(muted_col)
        ax.yaxis.label.set_color(muted_col)
        ax.title.set_color(text_col)
        for spine in ax.spines.values():
            spine.set_edgecolor('#ffffff18')

    analysis_tab = st.selectbox("Select Plot", [
        "Feature Distributions", "Pairplot", "3D Scatter",
        "Confusion Matrix – SVM", "Confusion Matrix – KNN", "Confusion Matrix – DT"
    ])

    if analysis_tab == "Feature Distributions":
        import matplotlib.pyplot as plt, matplotlib.patches as mpatches
        st.markdown("#### Feature Distributions by Category")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor(dark_bg)
        for ax, feat in zip(axes, ['Engine Capacity', 'Horsepower', 'Performance (0-100s)']):
            for cat, color in cat_colors.items():
                ax.hist(dataset[dataset['Car Category']==cat][feat],
                        bins=20, alpha=0.6, color=color, edgecolor='none', density=True)
            style_ax(ax); ax.set_title(feat, fontsize=11)
            ax.set_xlabel(feat, fontsize=10); ax.set_ylabel('Density', fontsize=10)
        handles = [mpatches.Patch(color=c, label=l) for l, c in cat_colors.items()]
        fig.legend(handles=handles, loc='lower center', ncol=5, framealpha=0,
                   labelcolor=text_col, fontsize=10, bbox_to_anchor=(0.5,-0.15))
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    elif analysis_tab == "Pairplot":
        import matplotlib.pyplot as plt, seaborn as sns
        st.markdown("#### Pairwise Feature Relationships")
        sample_ds = pd.concat([
            grp.sample(min(60,len(grp)), random_state=42)
            for _, grp in dataset.groupby('Car Category')
        ]).reset_index(drop=True)
        pp = sns.pairplot(sample_ds,
                          vars=['Engine Capacity','Horsepower','Performance (0-100s)'],
                          hue='Car Category', palette=cat_colors,
                          plot_kws={'alpha':0.6,'s':20}, diag_kws={'alpha':0.5})
        pp.fig.patch.set_facecolor(dark_bg)
        for ax in pp.axes.flatten():
            if ax:
                ax.set_facecolor(card_bg); ax.tick_params(colors=muted_col, labelsize=8)
                for spine in ax.spines.values(): spine.set_edgecolor('#ffffff18')
        st.pyplot(pp.fig, use_container_width=True); plt.close(pp.fig)

    elif analysis_tab == "3D Scatter":
        import matplotlib.pyplot as plt
        st.markdown("#### 3D Feature Distribution")
        fig3d = plt.figure(figsize=(10,7)); fig3d.patch.set_facecolor(dark_bg)
        ax3d  = fig3d.add_subplot(111, projection='3d'); ax3d.set_facecolor(card_bg)
        for cat, color in cat_colors.items():
            idx = dataset['Car Category']==cat
            ax3d.scatter(dataset.loc[idx,'Engine Capacity'],
                         dataset.loc[idx,'Horsepower'],
                         dataset.loc[idx,'Performance (0-100s)'],
                         c=color, label=cat, s=18, alpha=0.7)
        ax3d.set_xlabel('Engine Capacity',color=muted_col,fontsize=9)
        ax3d.set_ylabel('Horsepower',     color=muted_col,fontsize=9)
        ax3d.set_zlabel('0-100s',         color=muted_col,fontsize=9)
        ax3d.tick_params(colors=muted_col,labelsize=7)
        ax3d.set_title('3D Feature Distribution',color=text_col,fontsize=13)
        ax3d.legend(facecolor=card_bg,labelcolor=text_col,fontsize=9,loc='upper left',framealpha=0.5)
        ax3d.xaxis.pane.fill=ax3d.yaxis.pane.fill=ax3d.zaxis.pane.fill=False
        st.pyplot(fig3d, use_container_width=True); plt.close(fig3d)

    else:
        import matplotlib.pyplot as plt, itertools
        from matplotlib.colors import LinearSegmentedColormap
        from sklearn.metrics import confusion_matrix
        model_map = {
            "Confusion Matrix – SVM": svm_model,
            "Confusion Matrix – KNN": knn_model,
            "Confusion Matrix – DT":  dt_model,
        }
        chosen_model = model_map[analysis_tab]
        y_pred_cm = chosen_model.predict(X_test_sc)
        cm        = confusion_matrix(y_test, y_pred_cm)
        classes   = le.classes_

        fig_cm, ax_cm = plt.subplots(figsize=(8,6))
        fig_cm.patch.set_facecolor(dark_bg); ax_cm.set_facecolor(card_bg)
        cmap_custom = LinearSegmentedColormap.from_list('custom',['#131629','#7c6ff7'],N=256)
        im = ax_cm.imshow(cm, interpolation='nearest', cmap=cmap_custom)
        plt.colorbar(im, ax=ax_cm).ax.yaxis.set_tick_params(color=muted_col)
        tick_marks = np.arange(len(classes))
        ax_cm.set_xticks(tick_marks); ax_cm.set_xticklabels(classes, rotation=35, ha='right', color=muted_col)
        ax_cm.set_yticks(tick_marks); ax_cm.set_yticklabels(classes, color=muted_col)
        thresh = cm.max()/2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax_cm.text(j,i,format(cm[i,j],'d'),ha='center',va='center',
                       color='white' if cm[i,j]>thresh else muted_col,fontsize=12,fontweight='bold')
        ax_cm.set_ylabel('True Label',color=muted_col)
        ax_cm.set_xlabel('Predicted Label',color=muted_col)
        ax_cm.set_title(analysis_tab.replace('Confusion Matrix – ','')+ ' Confusion Matrix',
                        color=text_col,fontsize=13,pad=12)
        for spine in ax_cm.spines.values(): spine.set_edgecolor('#ffffff18')
        st.pyplot(fig_cm, use_container_width=True); plt.close(fig_cm)
        st.info(f"Model Accuracy: **{accuracy_score(y_test, y_pred_cm)*100:.2f}%**")
