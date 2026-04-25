import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
import openai

st.sidebar.markdown('---')
dark_mode = st.sidebar.checkbox('Dark Mode', value=False, key='dark_mode_css')
st.sidebar.markdown('---')

# Advanced CSS with dark mode support
if dark_mode:
    css = """
    <style>
        .main {
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #2d2d2d;
            border-right: 2px solid #404040;
            color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #61dafb;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: #2d2d2d;
            padding: 10px;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #404040;
            border-radius: 6px;
            border: 1px solid #555555;
            color: #cccccc;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: #61dafb;
            color: #1e1e1e;
            border-color: #61dafb;
            box-shadow: 0 2px 4px rgba(97,218,251,0.2);
        }
        .metric-card {
            background-color: #2d2d2d;
            border: 1px solid #404040;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            margin: 10px 0;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #61dafb;
        }
        .metric-label {
            font-size: 0.9em;
            color: #cccccc;
            text-transform: uppercase;
        }
        .stButton>button {
            background-color: #61dafb;
            color: #1e1e1e;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: 500;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #21b4d6;
        }
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            background-color: #2d2d2d;
            color: #ffffff;
        }
        .dataframe th {
            background-color: #404040;
            color: #ffffff;
        }
        .dataframe td {
            background-color: #2d2d2d;
            color: #cccccc;
        }
    </style>
    """
else:
    css = """
    <style>
        .main {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            border-right: 2px solid #e9ecef;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: #f1f3f4;
            padding: 10px;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #ffffff;
            border-radius: 6px;
            border: 1px solid #dee2e6;
            color: #495057;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: #007bff;
            color: white;
            border-color: #007bff;
            box-shadow: 0 2px 4px rgba(0,123,255,0.2);
        }
        .metric-card {
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #28a745;
        }
        .metric-label {
            font-size: 0.9em;
            color: #6c757d;
            text-transform: uppercase;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: 500;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """

st.markdown(css, unsafe_allow_html=True)

st.set_page_config(page_title='US Housing x Macro Dashboard (Advanced)', layout='wide')
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / 'data' / 'us_home_price_analysis_2004_2024.csv'

def find_date_col(df: pd.DataFrame):
    cols = [c.lower() for c in df.columns]
    for i, c in enumerate(cols):
        if c in ['date','month'] or 'date' in c or 'time' in c:
            return df.columns[i]
    return None

def safe_parse_date(df: pd.DataFrame, date_col):
    if not date_col:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors='coerce')
    return out.sort_values(date_col)

def safe_numeric(df: pd.DataFrame, date_col):
    out = df.copy()
    for c in out.columns:
        if date_col and c == date_col:
            continue
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def admin_label(dt):
    if pd.isna(dt):
        return 'Unknown'
    if dt < pd.Timestamp('2017-01-20'):
        return 'Pre-Trump'
    if dt <= pd.Timestamp('2021-01-19'):
        return 'Trump (2017-2020)'
    if dt <= pd.Timestamp('2025-01-19'):
        return 'Biden (2021-2024)'
    return 'Post-Biden'

def metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))  # sklearn-old compatible (no squared=)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def corr_matrix(df: pd.DataFrame):
    num = df.select_dtypes(include=['number'])
    if num.empty:
        return pd.DataFrame()
    return num.corr(numeric_only=True)

def build_supervised_with_lags(df: pd.DataFrame, date_col: str, target: str, features, lags=(1,3,6,12), roll_windows=(3,6,12)):
    d = df[[date_col, target] + list(features)].copy()
    d = d.sort_values(date_col).reset_index(drop=True)
    for L in lags:
        d[f'{target}_lag{L}'] = d[target].shift(L)
    for w in roll_windows:
        d[f'{target}_rollmean{w}'] = d[target].rolling(w).mean()
    d[f'{target}_diff1'] = d[target].diff(1)
    for f in features:
        d[f'{f}_lag1'] = d[f].shift(1)
    return d

def radar_compare(df: pd.DataFrame, metrics_cols):
    need = {'Trump (2017-2020)', 'Biden (2021-2024)'}
    if not need.issubset(set(df['administration'].unique())):
        return None
    d = df[list(metrics_cols) + ['administration']].copy().dropna()
    if d.empty:
        return None
    mins = d[list(metrics_cols)].min()
    maxs = d[list(metrics_cols)].max()
    scaled = (d[list(metrics_cols)] - mins) / (maxs - mins).replace(0, np.nan)
    d2 = pd.concat([scaled, d['administration']], axis=1)
    prof = d2.groupby('administration')[list(metrics_cols)].mean().loc[['Trump (2017-2020)','Biden (2021-2024)']]
    cats = list(metrics_cols)
    cats2 = cats + [cats[0]]
    trump = prof.loc['Trump (2017-2020)'].tolist()
    biden = prof.loc['Biden (2021-2024)'].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=trump+[trump[0]], theta=cats2, fill='toself', name='Trump (scaled)'))
    fig.add_trace(go.Scatterpolar(r=biden+[biden[0]], theta=cats2, fill='toself', name='Biden (scaled)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=520, title='Radar (0-1 normalized mean profile)')
    return fig

st.sidebar.title('US Housing x Macro (Advanced)')
st.sidebar.caption('Kaggle CSV + EDA + Admin comparison + ML + Forecast')
st.sidebar.markdown('---')
st.sidebar.subheader('AI Code Assistant (Codex)')
api_key = st.sidebar.text_input('OpenAI API Key', type='password')
if api_key:
    import os
    os.environ['OPENAI_API_KEY'] = api_key
    openai.api_key = api_key
    prompt = st.sidebar.text_area('Enter a code prompt', height=100)
    if st.sidebar.button('Generate Code'):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates Python code."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            generated_code = response.choices[0].message.content.strip()
            st.sidebar.code(generated_code, language='python')
        except Exception as e:
            st.sidebar.error(f'Error: {e}')
st.sidebar.markdown('**Where to put your data**:')
st.sidebar.code(str(DEFAULT_CSV), language='text')
uploaded = st.sidebar.file_uploader('Upload Kaggle CSV', type=['csv'])
use_default = st.sidebar.checkbox('Use default CSV from /data', value=(uploaded is None))

if uploaded is not None:
    df = pd.read_csv(uploaded)
    source_name = f'Uploaded: {uploaded.name}'
elif use_default and DEFAULT_CSV.exists():
    df = pd.read_csv(DEFAULT_CSV)
    source_name = f'Local: {DEFAULT_CSV}'
else:
    st.error('No dataset found. Upload the CSV OR place it at data/us_home_price_analysis_2004_2024.csv')
    st.stop()

df.columns = df.columns.str.strip()
date_col = find_date_col(df)
df = safe_parse_date(df, date_col)
df = safe_numeric(df, date_col)
if date_col and df[date_col].notna().any():
    df['administration'] = df[date_col].apply(admin_label)
else:
    df['administration'] = 'Unknown'

st.sidebar.markdown('---')
dark_mode = st.sidebar.checkbox('Dark Mode', value=False, key='dark_mode_toggle')
st.sidebar.markdown('---')
show_raw = st.sidebar.checkbox('Show raw data preview', value=False)
if date_col and df[date_col].notna().any():
    dmin, dmax = df[date_col].min(), df[date_col].max()
    sel = st.sidebar.slider('Date range', min_value=dmin.to_pydatetime(), max_value=dmax.to_pydatetime(), value=(dmin.to_pydatetime(), dmax.to_pydatetime()))
    df = df[(df[date_col] >= pd.Timestamp(sel[0])) & (df[date_col] <= pd.Timestamp(sel[1]))].copy()
if show_raw:
    st.subheader('Raw preview')
    st.dataframe(df.head(50), width='stretch')

st.title('US Housing & Economic Indicators — Advanced Dashboard')
st.caption(f'Source: {source_name} • Goal: explain + predict housing dynamics using macro indicators and compare patterns under Trump vs Biden.')

# Scroll to top button with JS
st.components.v1.html("""
<button onclick="window.scrollTo({top: 0, behavior: 'smooth'});" style="position: fixed; bottom: 20px; right: 20px; background-color: #007bff; color: white; border: none; border-radius: 50%; width: 50px; height: 50px; font-size: 20px; cursor: pointer; z-index: 1000;">↑</button>
""")

# Custom metric cards
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Rows</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df.shape[1]:,}</div>
        <div class="metric-label">Columns</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{int(df.isna().sum().sum()):,}</div>
        <div class="metric-label">Missing Cells</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{int(df.duplicated().sum()):,}</div>
        <div class="metric-label">Duplicate Rows</div>
    </div>
    """, unsafe_allow_html=True)
with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df['administration'].nunique():,}</div>
        <div class="metric-label">Periods</div>
    </div>
    """, unsafe_allow_html=True)
with col6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{date_col if date_col else '—'}</div>
        <div class="metric-label">Date Col</div>
    </div>
    """, unsafe_allow_html=True)

tabs = st.tabs(['1) Goal & Interpretation','2) Data Quality','3) Time Series','4) Correlations & Scatter','5) Charts','6) Trump vs Biden','7) ML','8) Forecast','9) Export','10) Chatbot','11) OLAP Cube'])

with tabs[0]:
    st.subheader('Goal & Target')
    st.markdown('- Goal: quantify how housing moves with macro indicators, compare Trump (2017-2020) vs Biden (2021-2024).')
    st.markdown('- Target: choose a housing metric (recommend Home_Price_Index if present).')
    st.markdown('- Outputs: EDA + admin comparison + ML prediction + baseline forecast + exports.')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if num_cols:
        target_default = 'Home_Price_Index' if 'Home_Price_Index' in num_cols else num_cols[0]
        target = st.selectbox('Target for interpretation', num_cols, index=num_cols.index(target_default))
        cmat = corr_matrix(df[[target] + [c for c in num_cols if c != target]])
        if not cmat.empty:
            s = cmat[target].drop(target).dropna().sort_values(key=lambda x: x.abs(), ascending=False)
            st.write('Top correlations with target (association only):')
            st.dataframe(s.head(8).reset_index().rename(columns={'index':'Feature', target:'Corr'}), width='stretch')

with tabs[1]:
    st.subheader('Data Quality')
    miss = df.isna().sum().sort_values(ascending=False).reset_index()
    miss.columns = ['column','missing']
    miss['missing_%'] = (miss['missing']/len(df)*100).round(2)
    st.dataframe(miss, width='stretch')

with tabs[2]:
    st.subheader('Time Series')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not date_col or df[date_col].isna().all():
        st.info('No usable date column detected.')
    elif not num_cols:
        st.info('No numeric columns.')
    else:
        y = st.selectbox('Metric', num_cols, index=0)
        fig = px.line(df, x=date_col, y=y, color='administration', markers=True)
        fig.update_layout(height=520)
        st.plotly_chart(fig, width='stretch')
        win = st.slider('Rolling window (months)', 3, 24, 12)
        tmp = df[[date_col,y]].dropna().sort_values(date_col).copy()
        tmp['rolling_mean'] = tmp[y].rolling(win).mean()
        fig2 = px.line(tmp, x=date_col, y=['rolling_mean',y])
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, width='stretch')

with tabs[3]:
    st.subheader('Correlations')
    cmat = corr_matrix(df)
    if cmat.empty:
        st.info('Need numeric columns.')
    else:
        fig = px.imshow(cmat, text_auto='.2f', aspect='auto', zmin=-1, zmax=1, color_continuous_scale='RdBu')
        fig.update_layout(height=650)
        st.plotly_chart(fig, width='stretch')
    st.markdown('---')
    st.subheader('Scatter + trendline')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(num_cols) >= 2:
        x = st.selectbox('X', num_cols, index=0, key='scx')
        y = st.selectbox('Y', num_cols, index=1, key='scy')
        fig = px.scatter(df, x=x, y=y, color='administration', trendline='ols', opacity=0.65)
        fig.update_layout(height=520)
        st.plotly_chart(fig, width='stretch')

with tabs[4]:
    st.subheader('Charts')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if num_cols:
        metric = st.selectbox('Metric for distributions', num_cols, index=0)
        st.write('Box plot by administration')
        st.plotly_chart(px.box(df, x='administration', y=metric, points='outliers').update_layout(height=520), width='stretch')
        st.write('Violin plot by administration')
        st.plotly_chart(px.violin(df, x='administration', y=metric, box=True, points='all').update_layout(height=520), width='stretch')
        st.write('Radar (Spider): Trump vs Biden')
        cols = st.multiselect('Choose 3-8 metrics', num_cols, default=num_cols[:min(5,len(num_cols))])
        if cols:
            rfig = radar_compare(df, cols)
            if rfig is None:
                st.info('Need both Trump and Biden periods with enough data.')
            else:
                st.plotly_chart(rfig, width='stretch')

with tabs[5]:
    st.subheader('Trump vs Biden (descriptive stats)')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols = st.multiselect('Metrics', num_cols, default=num_cols[:min(6,len(num_cols))], key='adm_metrics')
    if cols:
        grp = df.groupby('administration')[cols].agg(['mean','median','std','min','max']).round(3)
        st.dataframe(grp, width='stretch')
        if {'Trump (2017-2020)','Biden (2021-2024)'} <= set(df['administration'].unique()):
            gmean = df.groupby('administration')[cols].mean(numeric_only=True)
            delta = (gmean.loc['Biden (2021-2024)'] - gmean.loc['Trump (2017-2020)']).sort_values()
            st.plotly_chart(px.bar(delta, orientation='h', labels={'value':'Delta (Biden - Trump)','index':'Metric'}).update_layout(height=450), width='stretch')

with tabs[6]:
    st.subheader('ML Prediction')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not num_cols:
        st.stop()
    target_default = 'Home_Price_Index' if 'Home_Price_Index' in num_cols else num_cols[0]
    target = st.selectbox('Target', num_cols, index=num_cols.index(target_default), key='ml_target')
    feature_candidates = [c for c in num_cols if c != target]
    feats = st.multiselect('Features', feature_candidates, default=feature_candidates[:min(10,len(feature_candidates))])
    if not feats:
        st.info('Select at least one feature.')
        st.stop()
    use_scaling = st.checkbox('Normalize features', value=True)
    scaler_type = st.selectbox('Scaling method', ['StandardScaler (z-score)','MinMaxScaler (0-1)'])
    scaler = StandardScaler() if scaler_type.startswith('Standard') else MinMaxScaler()
    task_type = st.radio('Task Type', ['Regression', 'Classification'], index=0)
    if task_type == 'Regression':
        model_choice = st.selectbox('Model', ['Ridge','RandomForest','LinearRegression','SVR','GradientBoosting'], index=1)
    else:
        model_choice = st.selectbox('Model', ['RandomForest','LogisticRegression'], index=0)
    d = df[[target] + feats].copy().dropna(subset=[target])
    split = int(len(d)*0.8)
    X_train, X_test = d[feats].iloc[:split], d[feats].iloc[split:]
    y_train, y_test = d[target].iloc[:split], d[target].iloc[split:]
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if use_scaling:
        steps.append(('scaler', scaler))
    if task_type == 'Regression':
        if model_choice == 'Ridge':
            steps.append(('model', Ridge(alpha=1.0, random_state=0)))
        elif model_choice == 'RandomForest':
            steps.append(('model', RandomForestRegressor(n_estimators=600, random_state=0, n_jobs=-1, min_samples_leaf=2)))
        elif model_choice == 'LinearRegression':
            steps.append(('model', LinearRegression()))
        elif model_choice == 'SVR':
            steps.append(('model', SVR(kernel='rbf', C=1.0, epsilon=0.1)))
        elif model_choice == 'GradientBoosting':
            steps.append(('model', GradientBoostingRegressor(n_estimators=100, random_state=0)))
        model = Pipeline(steps)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae, rmse, r2 = metrics(y_test, pred)
        c1,c2,c3 = st.columns(3)
        c1.metric('MAE', f'{mae:.3f}')
        c2.metric('RMSE', f'{rmse:.3f}')
        c3.metric('R2', f'{r2:.3f}')
        out = pd.DataFrame({'actual': y_test.values, 'pred': pred})
        st.plotly_chart(px.line(out, y=['actual','pred']).update_layout(height=420), width='stretch')
        st.subheader('Cross-Validation Evaluation')
        cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_mae = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_rmse = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        st.write(f'CV R²: {cv_r2.mean():.3f} ± {cv_r2.std():.3f}')
        st.write(f'CV MAE: {-cv_mae.mean():.3f} ± {cv_mae.std():.3f}')
        st.write(f'CV RMSE: {-cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}')
        st.info('Note: Accuracy and recall are for classification tasks. For regression, we evaluate with MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R² (coefficient of determination). Cross-validation provides robust estimates across different data splits.')
        st.subheader('Model Interpretation')
        if model_choice in ['RandomForest', 'GradientBoosting']:
            imp = pd.Series(model.named_steps['model'].feature_importances_, index=feats).sort_values()
            st.write('Feature Importance: Shows how much each feature contributes to the model\'s predictions. Higher values indicate more important features.')
            st.plotly_chart(px.bar(imp.tail(20), orientation='h', labels={'value':'Importance','index':'Feature'}).update_layout(height=520), width='stretch')
        elif model_choice in ['Ridge', 'LinearRegression']:
            coefs = pd.Series(model.named_steps['model'].coef_, index=feats).sort_values()
            st.write('Coefficients: Positive values increase the prediction, negative values decrease it. Magnitude shows strength of influence.')
            st.plotly_chart(px.bar(coefs, orientation='h', labels={'value':'Coefficient','index':'Feature'}).update_layout(height=520), width='stretch')
        elif model_choice == 'SVR':
            st.info('SVR does not provide interpretable feature importance or coefficients. Consider using tree-based or linear models for better interpretability.')
    else:  # Classification
        # Bin the target into 3 classes
        y_train_binned = pd.qcut(y_train, q=3, labels=['Low', 'Medium', 'High'])
        y_test_binned = pd.qcut(y_test, q=3, labels=['Low', 'Medium', 'High'])
        if model_choice == 'RandomForest':
            steps.append(('model', RandomForestClassifier(n_estimators=600, random_state=0, n_jobs=-1)))
        elif model_choice == 'LogisticRegression':
            steps.append(('model', LogisticRegression(random_state=0, max_iter=1000)))
        model = Pipeline(steps)
        model.fit(X_train, y_train_binned)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test_binned, pred)
        prec = precision_score(y_test_binned, pred, average='weighted')
        rec = recall_score(y_test_binned, pred, average='weighted')
        f1 = f1_score(y_test_binned, pred, average='weighted')
        try:
            auc = roc_auc_score(y_test_binned, model.predict_proba(X_test), multi_class='ovr', average='weighted')
        except:
            auc = None
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric('Accuracy', f'{acc:.3f}')
        c2.metric('Precision', f'{prec:.3f}')
        c3.metric('Recall', f'{rec:.3f}')
        c4.metric('F1 Score', f'{f1:.3f}')
        if auc is not None:
            c5.metric('AUC', f'{auc:.3f}')
        else:
            c5.metric('AUC', 'N/A')
        st.subheader('Classification Report')
        report = classification_report(y_test_binned, pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        st.subheader('Model Interpretation')
        if model_choice == 'RandomForest':
            imp = pd.Series(model.named_steps['model'].feature_importances_, index=feats).sort_values()
            st.write('Feature Importance: Shows how much each feature contributes to classifying the target into Low/Medium/High categories.')
            st.plotly_chart(px.bar(imp.tail(20), orientation='h', labels={'value':'Importance','index':'Feature'}).update_layout(height=520), width='stretch')
        elif model_choice == 'LogisticRegression':
            # For multiclass, coef_ is (n_classes, n_features)
            coef_df = pd.DataFrame(model.named_steps['model'].coef_, columns=feats, index=['Low', 'Medium', 'High'])
            st.write('Coefficients: Shows the influence of each feature on each class. Positive values favor the class.')
            for cls in coef_df.index:
                st.write(f'**{cls} Class:**')
                coefs = coef_df.loc[cls].sort_values()
                st.plotly_chart(px.bar(coefs, orientation='h', labels={'value':'Coefficient','index':'Feature'}).update_layout(height=300, title=f'Coefficients for {cls}'), width='stretch')

with tabs[7]:
    st.subheader('Forecast (Future results)')
    st.caption('Baseline forecast: uses target lags; macro held constant at last observed value.')
    if not date_col or df[date_col].isna().all():
        st.stop()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    target_default = 'Home_Price_Index' if 'Home_Price_Index' in num_cols else num_cols[0]
    target = st.selectbox('Target to forecast', num_cols, index=num_cols.index(target_default), key='fc_target')
    exog = st.multiselect('Exogenous features (held constant)', [c for c in num_cols if c != target], default=[c for c in num_cols if c != target][:min(6,len(num_cols)-1)])
    horizon = st.slider('Horizon (months)', 6, 24, 12)
    lag_periods = st.multiselect('Lag periods for target', [1,3,6,12], default=[1,3,6,12])
    roll_windows = st.multiselect('Rolling windows for target', [3,6,12], default=[3,6,12])
    model_choice = st.selectbox('Forecast model', ['Ridge','RandomForest','LinearRegression','SVR','GradientBoosting'], index=0)
    use_scaling = st.checkbox('Normalize (forecast)', value=True, key='fc_norm')
    scaler_type = st.selectbox('Scaler (forecast)', ['StandardScaler (z-score)','MinMaxScaler (0-1)'], key='fc_scaler')
    scaler = StandardScaler() if scaler_type.startswith('Standard') else MinMaxScaler()
    d0 = df[[date_col, target] + exog].copy().sort_values(date_col).reset_index(drop=True)
    for f in exog:
        d0[f] = d0[f].ffill()
    sup = build_supervised_with_lags(d0, date_col, target, exog, lags=tuple(lag_periods), roll_windows=tuple(roll_windows)).dropna(subset=[target])
    feature_cols = [c for c in sup.columns if c not in [date_col, target]]
    split = max(10, len(sup) - horizon)
    train = sup.iloc[:split].copy()
    X_train, y_train = train[feature_cols], train[target]
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if use_scaling:
        steps.append(('scaler', scaler))
    if model_choice == 'Ridge':
        steps.append(('model', Ridge(alpha=1.0, random_state=0)))
    elif model_choice == 'RandomForest':
        steps.append(('model', RandomForestRegressor(n_estimators=800, random_state=0, n_jobs=-1, min_samples_leaf=2)))
    elif model_choice == 'LinearRegression':
        steps.append(('model', LinearRegression()))
    elif model_choice == 'SVR':
        steps.append(('model', SVR(kernel='rbf', C=1.0, epsilon=0.1)))
    elif model_choice == 'GradientBoosting':
        steps.append(('model', GradientBoostingRegressor(n_estimators=100, random_state=0)))
    model = Pipeline(steps)
    model.fit(X_train, y_train)
    last_date = d0[date_col].dropna().max()
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
    tmp = d0.copy()
    preds = []
    for dt in future_dates:
        new_row = {date_col: dt, target: np.nan}
        for f in exog:
            new_row[f] = tmp[f].iloc[-1] if len(tmp) else np.nan
        tmp = pd.concat([tmp, pd.DataFrame([new_row])], ignore_index=True)
        sup_tmp = build_supervised_with_lags(tmp, date_col, target, exog)
        last_feat = sup_tmp.iloc[[-1]][feature_cols]
        yhat = float(model.predict(last_feat)[0])
        tmp.loc[tmp.index[-1], target] = yhat
        preds.append(yhat)
    forecast_df = pd.DataFrame({date_col: future_dates, 'forecast': preds})
    hist = d0[[date_col, target]].dropna().copy()
    fig = px.line(hist, x=date_col, y=target, title='History + Forecast')
    fig.add_scatter(x=forecast_df[date_col], y=forecast_df['forecast'], mode='lines+markers', name='Forecast')
    fig.update_layout(height=520)
    st.plotly_chart(fig, width='stretch')
    st.dataframe(forecast_df, width='stretch')
    st.download_button('Download forecast CSV', data=forecast_df.to_csv(index=False).encode('utf-8'), file_name='forecast.csv', mime='text/csv')

with tabs[8]:
    st.subheader('Export')
    st.download_button('Download filtered dataset CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='filtered_us_housing.csv', mime='text/csv')

with tabs[9]:
    st.subheader('AI Chatbot')
    st.write('Chat with an AI assistant about your data, analysis, or general questions.')
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    if prompt := st.chat_input('Ask something...'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            full_response = ""
            # Prepare context about the app
            system_prompt = f"You are a helpful assistant for a Streamlit app analyzing US Housing data. The app has data from {source_name}, with columns: {list(df.columns)}. It includes EDA, ML models, forecasts, and comparisons between Trump and Biden administrations. Answer questions helpfully."
            messages = [{'role': 'system', 'content': system_prompt}] + st.session_state.messages
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7,
                    stream=True
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({'role': 'assistant', 'content': full_response})
            except Exception as e:
                st.error(f'Error: {e}')

with tabs[10]:
    st.subheader('OLAP Cube - Pivot Analysis')
    st.write('Perform multidimensional analysis with pivot tables.')
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist() + ['administration']
    index = st.multiselect('Rows (index)', cat_cols, default=[], key='olap_index')
    columns = st.multiselect('Columns', cat_cols, default=[], key='olap_columns')
    values = st.multiselect('Values (aggregate)', num_cols, default=num_cols[:1] if num_cols else [], key='olap_values')
    aggfunc = st.selectbox('Aggregation function', ['mean', 'sum', 'count', 'std', 'min', 'max'], key='olap_agg')
    if values and index:
        try:
            pivot = pd.pivot_table(df, values=values, index=index, columns=columns, aggfunc=aggfunc, fill_value=0, dropna=False)
            st.dataframe(pivot, width='stretch')
            st.download_button('Download Pivot CSV', data=pivot.reset_index().to_csv(index=False).encode('utf-8'), file_name='pivot_analysis.csv', mime='text/csv', key='download_pivot')
        except Exception as e:
            st.error(f'Error creating pivot: {e}')
    else:
        st.info('Select at least one row and one value to create the pivot table.')
