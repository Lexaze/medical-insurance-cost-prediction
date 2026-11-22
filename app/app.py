"""
Medical Insurance Cost Prediction — Streamlit Application
An interactive tool to predict insurance charges using linear regression models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import io, os

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.5rem 0 0; }

    .prediction-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0ea5e9;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-card h2 { color: #0369a1; font-size: 2rem; margin: 0; }
    .prediction-card p  { color: #0c4a6e; margin: 0; }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-card .label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.5rem; font-weight: bold; color: #1e293b; }

    .model-desc {
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0 1rem;
        font-size: 0.9rem;
        color: #475569;
    }
    .stSelectbox label, .stSlider label { font-weight: 600 !important; }
    .insight-box {
        background: #fefce8;
        border: 1px solid #fde047;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load & train models ──────────────────────────────────────────────────────
@st.cache_resource
def load_data_and_models():
    # Locate data file relative to this script
    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, '..', 'data', 'medical_insurance.csv')
    df = pd.read_csv(data_path)

    df['smoker_enc'] = (df['smoker'] == 'yes').astype(int)
    df['sex_enc']    = (df['sex'] == 'male').astype(int)
    y = df['charges'].values

    SEED = 42
    models, metrics = {}, {}

    configs = {
        'model1': {
            'features': ['smoker_enc'],
            'display': ['Smoker Status'],
            'name': 'Univariate (Smoker only)',
            'color': '#2563eb',
        },
        'model2': {
            'features': ['smoker_enc', 'age'],
            'display': ['Smoker Status', 'Age'],
            'name': 'Bivariate (Smoker + Age)',
            'color': '#16a34a',
        },
        'model3': {
            'features': ['smoker_enc', 'age', 'bmi', 'children'],
            'display': ['Smoker Status', 'Age', 'BMI', 'Children'],
            'name': 'Multivariate (Smoker + Age + BMI + Children)',
            'color': '#dc2626',
        },
    }

    for key, cfg in configs.items():
        X = df[cfg['features']].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
        m = LinearRegression().fit(Xtr, ytr)
        m_full = LinearRegression().fit(X, y)
        yp = m.predict(Xte)
        metrics[key] = {
            'r2':   r2_score(yte, yp),
            'mae':  mean_absolute_error(yte, yp),
            'rmse': np.sqrt(mean_squared_error(yte, yp)),
            'ytest': yte,
            'ypred': yp,
        }
        models[key] = m_full

    return df, models, metrics, configs


df, models, metrics, configs = load_data_and_models()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏥 Medical Insurance Cost Predictor</h1>
  <p>Compare Univariate, Bivariate & Multivariate Linear Regression · Powered by Real-World Data</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar: user inputs ─────────────────────────────────────────────────────
st.sidebar.header("👤 Enter Your Information")
st.sidebar.markdown("---")

age      = st.sidebar.slider("Age (years)", 18, 64, 35)
sex      = st.sidebar.selectbox("Sex", ["Female", "Male"])
bmi      = st.sidebar.slider("BMI", 10.0, 55.0, 27.5, step=0.1,
                               help="Body Mass Index. Normal range: 18.5–24.9")
children = st.sidebar.selectbox("Number of Children / Dependants", [0, 1, 2, 3, 4, 5])
smoker   = st.sidebar.radio("Smoking Status", ["Non-Smoker", "Smoker"], horizontal=True)
region   = st.sidebar.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

st.sidebar.markdown("---")
model_choice = st.sidebar.radio(
    "🔧 Model to Use for Prediction",
    ["model1", "model2", "model3"],
    format_func=lambda x: configs[x]['name'],
)

st.sidebar.markdown("---")
st.sidebar.caption("**Data:** Kaggle Medical Insurance Dataset (n=2,772)  \n"
                   "**Method:** Ordinary Least Squares Linear Regression")


# ─── Encode inputs ────────────────────────────────────────────────────────────
smoker_enc = 1 if smoker == "Smoker" else 0
sex_enc    = 1 if sex == "Male" else 0

input_map = {
    'smoker_enc': smoker_enc,
    'age': age,
    'bmi': bmi,
    'children': children,
    'sex_enc': sex_enc,
}


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Prediction", "📊 Model Comparison", "🔍 Exploratory Analysis", "📖 About"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.subheader("💡 Predicted Insurance Cost")

        # Predict with selected model
        feats = configs[model_choice]['features']
        X_input = np.array([[input_map[f] for f in feats]])
        prediction = models[model_choice].predict(X_input)[0]
        prediction = max(prediction, 0)

        # Also compute all 3 for comparison strip
        preds_all = {}
        for k in ['model1', 'model2', 'model3']:
            feat_k = configs[k]['features']
            X_k = np.array([[input_map[f] for f in feat_k]])
            preds_all[k] = max(models[k].predict(X_k)[0], 0)

        # Main prediction card
        st.markdown(f"""
        <div class="prediction-card">
          <p>Estimated Annual Insurance Charges<br>
             <small>({configs[model_choice]['name']})</small></p>
          <h2>${prediction:,.2f}</h2>
          <p>≈ ${prediction/12:,.2f} per month</p>
        </div>
        """, unsafe_allow_html=True)

        # Model description
        descriptions = {
            'model1': "Uses <strong>smoker status alone</strong> — the single strongest predictor (r=0.789). Simple but powerful: smoking vs non-smoking is the dominant cost driver.",
            'model2': "Adds <strong>age</strong> to smoker status. Captures both the categorical risk factor (smoking) and the most informative continuous gradient (age).",
            'model3': "Uses all four key features: <strong>smoker, age, BMI, and children</strong>. Best overall performance (R²≈0.74), incorporating multiple health risk dimensions.",
        }
        st.markdown(f'<div class="model-desc"> {descriptions[model_choice]}</div>',
                    unsafe_allow_html=True)

        # All-model comparison strip
        st.markdown("**Estimates across all three models:**")
        c1, c2, c3 = st.columns(3)
        for col, k, label in [
            (c1, 'model1', '🔵 Univariate'),
            (c2, 'model2', '🟢 Bivariate'),
            (c3, 'model3', '🔴 Multivariate'),
        ]:
            with col:
                highlighted = k == model_choice
                st.markdown(f"""
                <div class="metric-card" style="{'border: 2px solid #2563eb;' if highlighted else ''}">
                  <div class="label">{label}</div>
                  <div class="value">${preds_all[k]:,.0f}</div>
                  <div style="font-size:0.75rem; color:#64748b;">R²={metrics[k]['r2']:.3f}</div>
                </div>""", unsafe_allow_html=True)

        # Insight box
        st.markdown("")
        if smoker_enc == 1:
            st.markdown("""<div class="insight-box">
            🚬 <strong>Smoker detected:</strong> Smoking status increases predicted charges by approximately
            <strong>$23,000–$24,000/year</strong> compared to a non-smoker with the same profile.
            This is by far the largest cost driver in the model.
            </div>""", unsafe_allow_html=True)

        bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
        if bmi >= 30:
            st.markdown(f"""<div class="insight-box">
            ⚖️ <strong>BMI {bmi:.1f} — {bmi_category}:</strong> An obese BMI is associated with higher insurance
            costs, particularly for smokers where the interaction is amplified.
            </div>""", unsafe_allow_html=True)

    with col_side:
        st.subheader("📋 Your Profile")
        profile = {
            "Age": f"{age} years",
            "Sex": sex,
            "BMI": f"{bmi:.1f}",
            "Children": str(children),
            "Smoker": smoker,
            "Region": region,
        }
        for k, v in profile.items():
            st.markdown(f"**{k}:** {v}")

        st.markdown("---")
        st.subheader(" Model Equation")
        m = models[model_choice]
        eq_lines = [f"**intercept:** ${m.intercept_:,.2f}"]
        for feat, coef in zip(configs[model_choice]['features'], m.coef_):
            sign = "+" if coef >= 0 else "−"
            eq_lines.append(f"**{sign}** ${abs(coef):,.2f} × {feat}")
        for line in eq_lines:
            st.markdown(line)

        st.markdown("---")
        # Gauge-style visual
        st.subheader(" Cost Context")
        dataset_min  = df['charges'].min()
        dataset_max  = df['charges'].max()
        dataset_mean = df['charges'].mean()
        pct = (prediction - dataset_min) / (dataset_max - dataset_min) * 100

        fig, ax = plt.subplots(figsize=(3.5, 1.2))
        ax.barh([0], [dataset_max], color='#e2e8f0', height=0.4)
        ax.barh([0], [prediction],  color=configs[model_choice]['color'], height=0.4, alpha=0.85)
        ax.axvline(dataset_mean, color='#f59e0b', linestyle='--', linewidth=1.2)
        ax.set_xlim(0, dataset_max)
        ax.set_yticks([]); ax.set_xticks([0, dataset_mean, dataset_max])
        ax.set_xticklabels(['$0', f'Avg\n${dataset_mean:,.0f}', f'${dataset_max:,.0f}'], fontsize=7)
        ax.set_title(f"Percentile: {pct:.0f}th", fontsize=9)
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Model Performance Comparison (Test Set)")

    # Metrics table
    rows = []
    for k in ['model1', 'model2', 'model3']:
        rows.append({
            'Model': configs[k]['name'],
            'Features Used': len(configs[k]['features']),
            'Test R²': f"{metrics[k]['r2']:.4f}",
            'Test MAE': f"${metrics[k]['mae']:,.2f}",
            'Test RMSE': f"${metrics[k]['rmse']:,.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Charts
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Model Performance Metrics (Test Set)', fontsize=12, fontweight='bold')

    model_labels = ['M1\nUnivariate', 'M2\nBivariate', 'M3\nMultivariate']
    colors = [configs[k]['color'] for k in ['model1', 'model2', 'model3']]

    for ax, metric_key, title, fmt in [
        (axes[0], 'r2',   'R² Score (↑ better)',    '.3f'),
        (axes[1], 'mae',  'MAE in $ (↓ better)',    ',.0f'),
        (axes[2], 'rmse', 'RMSE in $ (↓ better)',   ',.0f'),
    ]:
        vals = [metrics[k][metric_key] for k in ['model1', 'model2', 'model3']]
        bars = ax.bar(model_labels, vals, color=colors, alpha=0.85, edgecolor='white', width=0.5)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, vals):
            label = f'{val:{fmt}}' if metric_key == 'r2' else f'${val:{fmt}}'
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    label, ha='center', fontsize=8, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.subheader("🎯 Actual vs Predicted (Test Set)")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, k in zip(axes, ['model1', 'model2', 'model3']):
        yt = metrics[k]['ytest']
        yp = metrics[k]['ypred']
        ax.scatter(yt, yp, alpha=0.35, s=12, color=configs[k]['color'])
        lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lim, lim, 'k--', linewidth=1.2)
        ax.set_xlabel('Actual ($)'); ax.set_ylabel('Predicted ($)')
        ax.set_title(f"{configs[k]['name'].split('(')[0]}\nR²={metrics[k]['r2']:.3f}", fontsize=9, fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.subheader("🔢 Regression Coefficients — Model 3 (Full Model)")
    feats = configs['model3']['features']
    coefs = models['model3'].coef_
    intercept = models['model3'].intercept_

    coef_df = pd.DataFrame({'Feature': feats, 'Coefficient ($)': coefs})
    coef_df['Interpretation'] = [
        f"Smokers pay ${coefs[0]:,.0f} more per year",
        f"Each year of age adds ${coefs[1]:,.0f} to annual cost",
        f"Each BMI unit adds ${coefs[2]:,.0f} to annual cost",
        f"Each additional child adds ${coefs[3]:,.0f} to annual cost",
    ]
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    st.caption(f"**Intercept:** ${intercept:,.2f} (baseline cost when all features = 0)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    for col, label, val in [
        (col1, "Total Records", f"{len(df):,}"),
        (col2, "Mean Charges", f"${df['charges'].mean():,.0f}"),
        (col3, "Smoker %", f"{(df['smoker']=='yes').mean()*100:.1f}%"),
        (col4, "Avg Age", f"{df['age'].mean():.0f} yrs"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
              <div class="label">{label}</div>
              <div class="value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Feature Distributions & Relationships with Charges', fontsize=12, fontweight='bold')

    smoke_colors = df['smoker'].map({'yes': '#dc2626', 'no': '#2563eb'})

    # Age vs Charges
    axes[0,0].scatter(df['age'], df['charges'], c=smoke_colors, alpha=0.3, s=10)
    axes[0,0].set_xlabel('Age'); axes[0,0].set_ylabel('Charges ($)')
    axes[0,0].set_title(f'Age vs Charges (r={df["age"].corr(df["charges"]):.3f})', fontweight='bold')

    # BMI vs Charges
    axes[0,1].scatter(df['bmi'], df['charges'], c=smoke_colors, alpha=0.3, s=10)
    axes[0,1].set_xlabel('BMI'); axes[0,1].set_ylabel('Charges ($)')
    axes[0,1].set_title(f'BMI vs Charges (r={df["bmi"].corr(df["charges"]):.3f})', fontweight='bold')

    # Charges distribution
    axes[0,2].hist(df['charges'], bins=40, color='#2563eb', alpha=0.75, edgecolor='white')
    axes[0,2].axvline(df['charges'].mean(), color='red', linestyle='--', linewidth=1.5)
    axes[0,2].set_xlabel('Charges ($)'); axes[0,2].set_ylabel('Count')
    axes[0,2].set_title('Distribution of Charges', fontweight='bold')

    # Smoker vs Charges
    ns = df[df['smoker']=='no']['charges']
    sk = df[df['smoker']=='yes']['charges']
    bp = axes[1,0].boxplot([ns, sk], tick_labels=['Non-Smoker', 'Smoker'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2563eb'); bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('#dc2626'); bp['boxes'][1].set_alpha(0.6)
    axes[1,0].set_ylabel('Charges ($)')
    axes[1,0].set_title('Charges by Smoker Status', fontweight='bold')

    # Age distribution
    axes[1,1].hist(df['age'], bins=20, color='#16a34a', alpha=0.75, edgecolor='white')
    axes[1,1].set_xlabel('Age'); axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Age Distribution', fontweight='bold')

    # Correlation bar chart
    df_num = df.copy()
    df_num['smoker_enc'] = (df['smoker']=='yes').astype(int)
    df_num['sex_enc']    = (df['sex']=='male').astype(int)
    corr_vals = df_num[['age','bmi','children','sex_enc','smoker_enc']].corrwith(df_num['charges']).sort_values(key=abs, ascending=False)
    bar_colors = ['#dc2626' if v > 0 else '#2563eb' for v in corr_vals.values]
    axes[1,2].barh(corr_vals.index, corr_vals.values, color=bar_colors, alpha=0.8, edgecolor='white')
    axes[1,2].axvline(0, color='black', linewidth=0.8)
    axes[1,2].set_xlabel('Pearson r with Charges')
    axes[1,2].set_title('Feature Correlations with Charges', fontweight='bold')

    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend
    legend_handles = [
        mpatches.Patch(color='#dc2626', alpha=0.7, label='Smoker'),
        mpatches.Patch(color='#2563eb', alpha=0.7, label='Non-Smoker'),
    ]
    fig.legend(handles=legend_handles, loc='upper right', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: About
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📖 About This Project")
    st.markdown("""
    ## Medical Insurance Cost Prediction Using Linear Regression

    This application demonstrates an end-to-end machine learning project comparing
    three linear regression models on the **Kaggle Medical Insurance Cost dataset**.

    ---

    ### Models

    | Model | Features | Test R² | Test MAE | Test RMSE |
    |-------|----------|---------|----------|-----------|
    | **M1: Univariate** | Smoker | 0.5991 | $5,928 | $7,844 |
    | **M2: Bivariate** | Smoker + Age | 0.7082 | $4,106 | $6,692 |
    | **M3: Multivariate** | Smoker + Age + BMI + Children | 0.7388 | $4,169 | $6,332 |

    ---

    ### Feature Selection Rationale

    **Why Smoker for Model 1?**
    Smoking status has the highest correlation with charges (r = 0.789). Smokers pay
    an average of **$32,223/year** vs **$8,418/year** for non-smokers — a 3.8× difference.

    **Why Smoker + Age for Model 2?**
    After smoking (r=0.789), age has the second-highest correlation (r=0.299). These two
    features capture independent dimensions: a categorical health risk and a continuous age gradient.

    **Why Smoker + Age + BMI + Children for Model 3?**
    BMI adds an independent health risk (r=0.200). Children adds marginal but consistent
    predictive power (r=0.066). Sex and region were excluded due to negligible correlations
    (r<0.07) and VIF analysis showed no multicollinearity concerns among selected features.

    ---

    ### Key Findings

    - 🚬 **Smoking is the dominant cost driver** — explains ~60% of variance alone
    - 👴 **Age matters** — each year adds approximately $250–260 to annual charges
    - ⚖️ **BMI adds incremental value** — especially for smokers (interaction effect)
    - 📈 **Model improves progressively** — R² improves from 0.60 → 0.71 → 0.74

    ---

    ### Limitations

    - Linear regression assumes additive effects; BMI × smoker interactions are not fully captured
    - ~26% of variance remains unexplained (pre-existing conditions, coverage type, etc.)
    - Residuals show mild heteroscedasticity at high predicted values (common for financial data)
    - Tree-based models (Random Forest, XGBoost) would likely achieve R² > 0.85

    ---

    ### Dataset
    **Source:** Kaggle Medical Cost Personal Datasets
    **Records:** 2,772 individuals
    **Features:** age, sex, bmi, children, smoker, region, charges
    **Train/Test Split:** 80% / 20% (stratified random, seed=42)
    """)

    st.markdown("---")
    st.info("**Built with:** Python · Scikit-learn · Streamlit · Matplotlib · Pandas")
