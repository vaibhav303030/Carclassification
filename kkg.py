import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Vehicle Classifier | ML Project",
    page_icon="🚗",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSS  (dark theme matching screenshots)
# ─────────────────────────────────────────

st.markdown("""
<style>
/* ── global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f1a;
    color: #e8eaf0;
}
/* ── header ── */
.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: #ffffff0a; border: 1px solid #ffffff18;
    border-radius: 20px; padding: 6px 18px;
    font-size: 12px; color: #8892a4; letter-spacing: .08em;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 3rem; font-weight: 800; line-height: 1.1;
    margin-bottom: 8px;
}
.hero-grad {
    background: linear-gradient(135deg,#7c6ff7,#f97316);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
/* ── stat cards ── */
.stat-row { display: flex; gap: 16px; margin: 20px 0 32px; }
.stat-card {
    background: #1a1e35; border: 1px solid #ffffff18;
    border-radius: 12px; padding: 16px 22px; flex: 1;
}
.stat-label { font-size: 11px; color: #8892a4; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 6px; }
.stat-val   { font-size: 22px; font-weight: 700; }
.accent     { color: #7c6ff7; }
/* ── section cards ── */
.section-card {
    background: #131629; border: 1px solid #ffffff18;
    border-radius: 16px; padding: 28px 32px; margin-bottom: 28px;
}
.section-title { font-size: 20px; font-weight: 700; margin-bottom: 4px; }
.section-sub   { color: #8892a4; font-size: 13px; margin-bottom: 20px; }
/* ── result box ── */
.result-success {
    background: #131629; border: 1px solid #7c6ff740;
    border-radius: 12px; padding: 20px 24px; margin-top: 20px;
}
.result-cat { font-size: 24px; font-weight: 700; margin-bottom: 8px; }
/* ── model table ── */
.model-table { width: 100%; border-collapse: collapse; }
.model-table th { font-size: 11px; color: #8892a4; text-transform: uppercase; letter-spacing: .1em; padding: 0 0 12px; text-align: left; }
.model-table td { padding: 14px 0; border-top: 1px solid #ffffff18; font-size: 14px; }
.model-table td:first-child { font-weight: 600; }
.badge-live    { background: #22d3ee18; color: #22d3ee; border: 1px solid #22d3ee40; border-radius: 20px; padding: 4px 12px; font-size: 12px; }
.badge-trained { background: #ffffff0a; color: #8892a4; border: 1px solid #ffffff18; border-radius: 20px; padding: 4px 12px; font-size: 12px; }
/* ── stMetric override ── */
[data-testid="stMetricValue"] { color: #7c6ff7 !important; font-size: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD & CLEAN DATASET
# ─────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv('Cars Datasets 2025.csv', encoding='latin1')

    def extract_cc(x):
        if pd.isna(x):
            return np.nan
        s = str(x).lower().replace(',', '').replace('cc', '').strip()
        try:
            return float(s)
        except Exception:
            return np.nan

    df['Engine Capacity'] = df['CC/Battery Capacity'].apply(extract_cc)
    df = df.dropna(subset=['Engine Capacity'])
    return df

df_cars = load_data()

# ─────────────────────────────────────────
# BUILD SYNTHETIC TRAINING DATASET
# ─────────────────────────────────────────

@st.cache_data
def build_dataset(engine_vals):
    np.random.seed(42)
    n = 150  # samples per category

    economy = pd.DataFrame({
        'Engine Capacity':       np.random.uniform(800,  1400,   n),
        'Horsepower':            np.random.uniform(60,   110,    n),
        'Performance (0-100s)':  np.random.uniform(10,   16,     n),
        'Fuel Type':             np.random.choice([0, 1], n),
        'Car Category':          'Economy'
    })
    suv = pd.DataFrame({
        'Engine Capacity':       np.random.uniform(1500, 3000,   n),
        'Horsepower':            np.random.uniform(130,  250,    n),
        'Performance (0-100s)':  np.random.uniform(7,    12,     n),
        'Fuel Type':             np.random.choice([0, 1, 3], n),
        'Car Category':          'SUV'
    })
    sports = pd.DataFrame({
        'Engine Capacity':       engine_vals.sample(n, replace=True).values,
        'Horsepower':            np.random.uniform(300,  700,    n),
        'Performance (0-100s)':  np.random.uniform(2.5,  6,      n),
        'Fuel Type':             np.zeros(n, dtype=int),
        'Car Category':          'Sports'
    })
    electric = pd.DataFrame({
        'Engine Capacity':       np.random.uniform(40000, 100000, n),
        'Horsepower':            np.random.uniform(150,  500,    n),
        'Performance (0-100s)':  np.random.uniform(3,    8,      n),
        'Fuel Type':             np.full(n, 2),
        'Car Category':          'Electric'
    })
    luxury = pd.DataFrame({
        'Engine Capacity':       np.random.uniform(2000, 6000,   n),
        'Horsepower':            np.random.uniform(250,  600,    n),
        'Performance (0-100s)':  np.random.uniform(4,    8,      n),
        'Fuel Type':             np.random.choice([0, 3], n),
        'Car Category':          'Luxury'
    })

    return pd.concat([economy, suv, sports, electric, luxury], ignore_index=True)

dataset = build_dataset(df_cars['Engine Capacity'])

# ─────────────────────────────────────────
# PREPROCESSING & TRAINING
# ─────────────────────────────────────────

FEATURES = ['Engine Capacity', 'Horsepower', 'Performance (0-100s)', 'Fuel Type']

X = dataset[FEATURES]
y = dataset['Car Category']

le      = LabelEncoder()
y_enc   = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

@st.cache_resource
def train_models(Xtr, ytr):
    svm = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
    knn = KNeighborsClassifier(n_neighbors=5)
    dt  = DecisionTreeClassifier(random_state=42)
    svm.fit(Xtr, ytr)
    knn.fit(Xtr, ytr)
    dt.fit(Xtr, ytr)
    return svm, knn, dt

svm_model, knn_model, dt_model = train_models(X_train_sc, y_train)

# Accuracies
svm_acc = accuracy_score(y_test, svm_model.predict(X_test_sc))
knn_acc = accuracy_score(y_test, knn_model.predict(X_test_sc))
dt_acc  = accuracy_score(y_test, dt_model.predict(X_test_sc))

# ─────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────

st.markdown("""
<div class="hero-badge">🚗 SVM POWERED &bull; CARS DATASET 2025</div>
<div class="hero-title">Vehicle <span class="hero-grad">Classifier</span></div>
<p style="color:#8892a4;font-size:15px;max-width:560px;margin-bottom:24px;line-height:1.6;">
Enter your car's specs and our ML model will instantly identify which of the
<strong style="color:#e8eaf0">5 categories</strong> it belongs to.
</p>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("SVM (Linear)", f"{svm_acc*100:.2f}%")
with c2:
    st.metric("KNN (k=5)",    f"{knn_acc*100:.2f}%")
with c3:
    st.metric("Best Model",   "SVM")

st.divider()

# ─────────────────────────────────────────
# NAVIGATION TABS
# ─────────────────────────────────────────

tab_classify, tab_models, tab_analysis = st.tabs(
    ["🔬 Classifier", "📊 Models", "🧬 Analysis"]
)

# ═══════════════════════════════════════════
# TAB 1 — CLASSIFIER
# ═══════════════════════════════════════════
with tab_classify:

    st.markdown("### Predict Car Category")
    st.markdown("<p style='color:#8892a4;font-size:13px;margin-bottom:20px'>Adjust the sliders or type values directly</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        engine = st.slider(
            "⚙️ Engine Capacity (cc for petrol/diesel · Wh for electric)",
            min_value=800, max_value=100000, value=2000, step=100
        )
        horsepower = st.slider(
            "⚡ Horsepower (hp)",
            min_value=60, max_value=700, value=200, step=5
        )

    with col2:
        performance = st.slider(
            "🏁 0–100 km/h Time (seconds)",
            min_value=2.0, max_value=16.0, value=9.0, step=0.1
        )
        fuel_type = st.selectbox(
            "🔋 Fuel Type",
            options=["Petrol", "Diesel", "Electric", "Hybrid"]
        )

    fuel_map = {"Petrol": 0, "Diesel": 1, "Electric": 2, "Hybrid": 3}

    # ── Sample presets ──
    st.markdown("**Try a sample:**")
    s_col = st.columns(5)
    samples = {
        "Economy":  dict(engine=1100, hp=75,  perf=12.0, fuel="Petrol"),
        "SUV":      dict(engine=2000, hp=180, perf=9.0,  fuel="Diesel"),
        "Sports":   dict(engine=4000, hp=450, perf=3.5,  fuel="Petrol"),
        "Electric": dict(engine=75000,hp=300, perf=5.0,  fuel="Electric"),
        "Luxury":   dict(engine=3500, hp=400, perf=5.5,  fuel="Petrol"),
    }
    sample_colors = {"Economy":"#22d3ee","SUV":"#4ade80","Sports":"#f43f5e","Electric":"#a78bfa","Luxury":"#fbbf24"}

    if 'preset' not in st.session_state:
        st.session_state.preset = None

    for i, (cat, sc) in enumerate(samples.items()):
        with s_col[i]:
            if st.button(cat, key=f"sample_{cat}", use_container_width=True):
                st.session_state.preset = cat

    # Apply preset values
    if st.session_state.preset:
        p = samples[st.session_state.preset]
        engine      = p['engine']
        horsepower  = p['hp']
        performance = p['perf']
        fuel_type   = p['fuel']

    st.markdown("")

    if st.button("🚀 Classify Vehicle", use_container_width=True, type="primary"):

        inp_scaled = scaler.transform([[engine, horsepower, performance, fuel_map[fuel_type]]])
        pred       = svm_model.predict(inp_scaled)
        prob       = svm_model.predict_proba(inp_scaled)[0]
        predicted  = le.inverse_transform(pred)[0]
        confidence = prob.max() * 100

        cat_colors = {
            'Economy':  '#22d3ee',
            'SUV':      '#4ade80',
            'Sports':   '#f43f5e',
            'Electric': '#a78bfa',
            'Luxury':   '#fbbf24',
        }
        color = cat_colors.get(predicted, '#7c6ff7')

        st.success(f"✅ Predicted Category: **{predicted}** ({confidence:.1f}% confidence)")

        # Confidence breakdown
        prob_df = pd.DataFrame({
            'Category':   le.classes_,
            'Confidence': prob * 100
        }).sort_values('Confidence', ascending=True)

        fig_prob, ax_prob = plt.subplots(figsize=(7, 3))
        fig_prob.patch.set_facecolor('#131629')
        ax_prob.set_facecolor('#131629')
        bar_colors = [cat_colors.get(c, '#7c6ff7') for c in prob_df['Category']]
        bars = ax_prob.barh(prob_df['Category'], prob_df['Confidence'], color=bar_colors, height=0.5)
        for bar, val in zip(bars, prob_df['Confidence']):
            ax_prob.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                         f'{val:.1f}%', va='center', color='#8892a4', fontsize=10)
        ax_prob.set_xlabel('Confidence (%)', color='#8892a4')
        ax_prob.tick_params(colors='#8892a4')
        ax_prob.spines[:].set_visible(False)
        ax_prob.set_xlim(0, 110)
        st.pyplot(fig_prob, use_container_width=True)
        plt.close(fig_prob)

        # Example cars
        cat_examples = {
            'Economy':  ['Maruti Alto', 'Hyundai i10', 'Tata Nano', 'Renault Kwid', 'Chevrolet Beat'],
            'SUV':      ['Hyundai Creta', 'Mahindra Scorpio', 'Toyota Fortuner', 'Kia Seltos', 'MG Hector'],
            'Sports':   ['Ford Mustang', 'BMW M3', 'Porsche 911', 'Audi TT', 'Ferrari SF90'],
            'Electric': ['Tata Nexon EV', 'MG ZS EV', 'Tesla Model 3', 'Hyundai Ioniq 5', 'Nissan Leaf'],
            'Luxury':   ['Mercedes E-Class', 'BMW 7 Series', 'Audi A8', 'Jaguar XF', 'Volvo S90'],
        }

        st.markdown(f"**Example {predicted} cars from the dataset:**")
        ex_cols = st.columns(5)
        for i, car in enumerate(cat_examples.get(predicted, [])):
            with ex_cols[i]:
                st.markdown(f"<div style='background:#1a1e35;border:1px solid #ffffff18;border-top:3px solid {color};border-radius:8px;padding:10px;text-align:center;font-size:12px;color:#e8eaf0'>🚗<br>{car}</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 2 — MODELS
# ═══════════════════════════════════════════
with tab_models:

    st.markdown("### Model Comparison")
    st.markdown("<p style='color:#8892a4;font-size:13px'>All classifiers trained on 80% of the synthetic Cars Dataset 2025</p>", unsafe_allow_html=True)

    model_data = {
        "Model":     ["SVM (Linear, C=1)", "KNN (k=5)", "Decision Tree"],
        "Algorithm": ["Support Vector Machine", "K-Nearest Neighbors", "CART Decision Tree"],
        "Accuracy":  [f"{svm_acc*100:.2f}%", f"{knn_acc*100:.2f}%", f"{dt_acc*100:.2f}%"],
        "Status":    ["Live ✅", "Trained", "Trained"],
    }
    model_df = pd.DataFrame(model_data)
    st.dataframe(model_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Per-Class Classification Report (SVM)")

    y_pred_svm = svm_model.predict(X_test_sc)
    report = classification_report(y_test, y_pred_svm, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).T.drop(columns=['support'], errors='ignore')
    st.dataframe(report_df.style.format("{:.2f}").background_gradient(cmap='Blues'), use_container_width=True)

# ═══════════════════════════════════════════
# TAB 3 — ANALYSIS
# ═══════════════════════════════════════════
with tab_analysis:

    cat_palette = {
        'Economy':  '#22d3ee',
        'SUV':      '#4ade80',
        'Sports':   '#f43f5e',
        'Electric': '#a78bfa',
        'Luxury':   '#fbbf24',
    }

    analysis_tab = st.selectbox(
        "Select Plot",
        ["Feature Distributions", "Pairplot", "3D Scatter", "Confusion Matrix – SVM",
         "Confusion Matrix – KNN", "Confusion Matrix – DT"]
    )

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

    # ── Feature Distributions ──
    if analysis_tab == "Feature Distributions":
        st.markdown("#### Feature Distributions by Category")
        features_to_plot = ['Engine Capacity', 'Horsepower', 'Performance (0-100s)']
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor(dark_bg)
        for ax, feat in zip(axes, features_to_plot):
            for cat, color in cat_palette.items():
                subset = dataset[dataset['Car Category'] == cat][feat]
                ax.hist(subset, bins=20, alpha=0.6, color=color, label=cat,
                        edgecolor='none', density=True)
            style_ax(ax)
            ax.set_title(feat, fontsize=11)
            ax.set_xlabel(feat, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
        handles = [mpatches.Patch(color=c, label=l) for l, c in cat_palette.items()]
        fig.legend(handles=handles, loc='lower center', ncol=5, framealpha=0,
                   labelcolor=text_col, fontsize=10, bbox_to_anchor=(0.5, -0.15))
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Pairplot ──
    elif analysis_tab == "Pairplot":
        st.markdown("#### Pairwise Feature Relationships")
        sample_ds = dataset.groupby('Car Category').apply(
            lambda x: x.sample(min(60, len(x)), random_state=42)
        ).reset_index(drop=True)
        pp = sns.pairplot(
            sample_ds,
            vars=['Engine Capacity', 'Horsepower', 'Performance (0-100s)'],
            hue='Car Category',
            palette=cat_palette,
            plot_kws={'alpha': 0.6, 's': 20},
            diag_kws={'alpha': 0.5}
        )
        pp.fig.patch.set_facecolor(dark_bg)
        for ax in pp.axes.flatten():
            if ax:
                ax.set_facecolor(card_bg)
                ax.tick_params(colors=muted_col, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#ffffff18')
        st.pyplot(pp.fig, use_container_width=True)
        plt.close(pp.fig)

    # ── 3D Scatter ──
    elif analysis_tab == "3D Scatter":
        st.markdown("#### 3D Feature Distribution")
        fig3d = plt.figure(figsize=(10, 7))
        fig3d.patch.set_facecolor(dark_bg)
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.set_facecolor(card_bg)
        for cat, color in cat_palette.items():
            idx = dataset['Car Category'] == cat
            ax3d.scatter(
                dataset.loc[idx, 'Engine Capacity'],
                dataset.loc[idx, 'Horsepower'],
                dataset.loc[idx, 'Performance (0-100s)'],
                c=color, label=cat, s=18, alpha=0.7
            )
        ax3d.set_xlabel('Engine Capacity', color=muted_col, fontsize=9)
        ax3d.set_ylabel('Horsepower',      color=muted_col, fontsize=9)
        ax3d.set_zlabel('0-100s',          color=muted_col, fontsize=9)
        ax3d.tick_params(colors=muted_col, labelsize=7)
        ax3d.set_title('3D Feature Distribution', color=text_col, fontsize=13)
        ax3d.legend(facecolor=card_bg, labelcolor=text_col, fontsize=9,
                    loc='upper left', framealpha=0.5)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        st.pyplot(fig3d, use_container_width=True)
        plt.close(fig3d)

    # ── Confusion Matrices ──
    else:
        model_map = {
            "Confusion Matrix – SVM": svm_model,
            "Confusion Matrix – KNN": knn_model,
            "Confusion Matrix – DT":  dt_model,
        }
        chosen_model = model_map[analysis_tab]
        y_pred_cm    = chosen_model.predict(X_test_sc)
        cm           = confusion_matrix(y_test, y_pred_cm)
        classes      = le.classes_

        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        fig_cm.patch.set_facecolor(dark_bg)
        ax_cm.set_facecolor(card_bg)

        from matplotlib.colors import LinearSegmentedColormap
        cmap_custom = LinearSegmentedColormap.from_list(
            'custom', ['#131629', '#7c6ff7'], N=256
        )
        im = ax_cm.imshow(cm, interpolation='nearest', cmap=cmap_custom)
        plt.colorbar(im, ax=ax_cm).ax.yaxis.set_tick_params(color=muted_col)

        tick_marks = np.arange(len(classes))
        ax_cm.set_xticks(tick_marks); ax_cm.set_xticklabels(classes, rotation=35, ha='right', color=muted_col)
        ax_cm.set_yticks(tick_marks); ax_cm.set_yticklabels(classes, color=muted_col)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax_cm.text(j, i, format(cm[i, j], 'd'),
                       ha='center', va='center',
                       color='white' if cm[i, j] > thresh else muted_col,
                       fontsize=12, fontweight='bold')

        ax_cm.set_ylabel('True Label',      color=muted_col)
        ax_cm.set_xlabel('Predicted Label', color=muted_col)
        ax_cm.set_title(analysis_tab.replace('Confusion Matrix – ', '') + ' Confusion Matrix',
                        color=text_col, fontsize=13, pad=12)
        for spine in ax_cm.spines.values():
            spine.set_edgecolor('#ffffff18')

        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)
        acc_shown = accuracy_score(y_test, y_pred_cm)
        st.info(f"Model Accuracy: **{acc_shown*100:.2f}%**")
