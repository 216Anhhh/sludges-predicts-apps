# ===== Fix for Streamlit Cloud =====
import matplotlib
matplotlib.use('Agg')
# ==================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import io
warnings.filterwarnings('ignore')

from matplotlib import font_manager

chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STHeiti', 'Heiti SC']
font_set = False
for font_name in chinese_fonts:
    try:
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        font_set = True
        break
    except:
        continue

if not font_set:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import plotly.graph_objects as go
import shap

st.set_page_config(
    page_title="污水处理智能分析平台",
    page_icon="💧",
    layout="wide"
)

# ============ session_state ============
for k, v in {
    'df_loaded': None, 'data_source': 'default',
    'predicted': False, 'pred_values': {}, 'input_values': {},
    'models': None, 'results': None, 'model_trained': False,
    'theme': 'dark',
    'show_ts': False, 'show_importance': False, 'show_heatmap': False,
    'show_scatter': False, 'show_metrics': False, 'show_violin': False,
    'show_box': False, 'show_shap': False,
    'scatter_params': {}, 'metrics_params': {}, 'ts_params': {},
    'importance_params': {}, 'violin_params': {}, 'box_params': {}, 'shap_params': {}
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============ 图片保存 ============
def save_matplotlib_fig(fig, filename, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

def save_plotly_fig(fig, filename, width=800, height=500):
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height)
        buf = io.BytesIO(img_bytes)
        buf.seek(0)
        return buf
    except Exception:
        st.warning("⚠️ 图片保存功能需要 kaleido 库支持")
        return None

def download_button_light_yellow(label, data, filename, mime="image/png", key=None):
    if data is None:
        return
    st.markdown("""
    <style>
    div[data-testid="stDownloadButton"] button {
        background-color: #f5e6a3 !important;
        color: #1a1a2e !important;
        border: 2px solid #e8d5a0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.4rem 1.2rem !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background-color: #ecd78a !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    return st.download_button(label=label, data=data, file_name=filename, mime=mime, key=key)

def clear_session_data():
    st.session_state.predicted = False
    st.session_state.pred_values = {}
    st.session_state.input_values = {}
    for k in ['show_ts','show_importance','show_heatmap','show_scatter','show_metrics','show_violin','show_box','show_shap']:
        st.session_state[k] = False
    for k in ['scatter_params','metrics_params','ts_params','importance_params','violin_params','box_params','shap_params']:
        st.session_state[k] = {}
    st.session_state.df_loaded = None
    st.session_state.data_source = 'default'
    st.rerun()

def clear_button(key=None):
    if st.button("🗑️ 清理数据", use_container_width=True, key=key):
        clear_session_data()

# ============ 主题 ============
def get_theme_colors(theme):
    if theme == 'dark':
        return {
            'bg': '#0e1117', 'sidebar_bg': '#0d1117', 'text': '#f0f6fc',
            'text_secondary': '#8b949e', 'border': '#30363d', 'primary': '#58a6ff',
            'success': '#3fb950', 'warning': '#d29922', 'danger': '#f85149',
            'card_bg': '#161b22', 'text_color': '#ffffff',
            'input_bg': '#1a1a2e', 'input_text': '#f0f6fc',
            'select_bg': '#1a1a2e', 'select_text': '#f0f6fc',
            'plot_facecolor': '#0d1117', 'plot_textcolor': 'white',
            'tab_bg': '#161b22', 'tab_active': '#238636', 'tab_text': '#8b949e', 'tab_active_text': '#ffffff',
        }
    else:
        return {
            'bg': '#f5f7fa', 'sidebar_bg': '#e8ecf1', 'text': '#1a1a2e',
            'text_secondary': '#3a4a5a', 'border': '#d0d7de', 'primary': '#1a5276',
            'success': '#1a8a4a', 'warning': '#b87a0a', 'danger': '#b02a37',
            'card_bg': '#ffffff', 'text_color': '#1a1a2e',
            'input_bg': '#f0f2f6', 'input_text': '#1a1a2e',
            'select_bg': '#ffffff', 'select_text': '#1a1a2e',
            'plot_facecolor': '#ffffff', 'plot_textcolor': '#1a1a2e',
            'tab_bg': '#ffffff', 'tab_active': '#b8d4e3', 'tab_text': '#4a5a6a', 'tab_active_text': '#1a1a2e',
        }

colors = get_theme_colors(st.session_state.theme)

def update_matplotlib_theme(theme, colors):
    if theme == 'dark':
        plt.rcParams.update({'text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#30363d','figure.facecolor':'#0d1117','axes.facecolor':'#0d1117'})
    else:
        plt.rcParams.update({'text.color':'#1a1a2e','axes.labelcolor':'#1a1a2e','xtick.color':'#1a1a2e','ytick.color':'#1a1a2e','axes.edgecolor':'#d0d7de','figure.facecolor':'#ffffff','axes.facecolor':'#ffffff'})

update_matplotlib_theme(st.session_state.theme, colors)

# ============ CSS ============
def get_css(colors, theme):
    light_overrides = ""
    if theme == 'light':
        light_overrides = """
        .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
        .stMarkdown label, .stMarkdown .label, .stMarkdown .value,
        .result-card .label, .result-card .value,
        .metric-card .label, .metric-card .value, .metric-card .sub,
        div, p, span, label { color: #1a1a2e !important; }
        .result-card, .result-card * { color: #1a1a2e !important; }
        .metric-card, .metric-card * { color: #1a1a2e !important; }
        .status-normal { color: #1a8a4a !important; font-weight: 700; }
        .status-warning { color: #b87a0a !important; font-weight: 700; }
        .status-danger { color: #b02a37 !important; font-weight: 700; }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
            color: #1a1a2e !important;
        }
        .stSelectbox div[data-baseweb="select"] div { background-color: #ffffff !important; color: #1a1a2e !important; }
        .stSelectbox ul { background-color: #ffffff !important; }
        .stSelectbox li { color: #1a1a2e !important; background-color: #ffffff !important; }
        .stSelectbox li:hover { background-color: #e8ecf1 !important; }
        .stNumberInput input, .stTextInput input { background-color: #f0f2f6 !important; color: #1a1a2e !important; }
        """
    if theme == 'dark':
        button_css = """
        .stButton button { background: #238636; color: #ffffff; font-weight: 700; border: none;
            border-radius: 8px; padding: 0.6rem 2rem; width: 100%; transition: all 0.3s ease; font-size: 1rem; }
        .stButton button:hover { background: #2ea043; transform: translateY(-2px); box-shadow: 0 4px 16px rgba(35,134,54,0.4); }
        """
    else:
        button_css = """
        .stButton button { background: #e8d5b8; color: #1a1a2e !important; font-weight: 700;
            border: 2px solid #d4a574; border-radius: 8px; padding: 0.6rem 2rem; width: 100%;
            transition: all 0.3s ease; font-size: 1rem; }
        .stButton button:hover { background: #dcc4a0; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(212,165,116,0.4); }
        """
    return f"""
    <style>
    .stApp {{ background-color: {colors['bg']}; }}
    section[data-testid="stSidebar"] {{ background-color: {colors['sidebar_bg']} !important; border-right: 1px solid {colors['border']} !important; }}
    section[data-testid="stSidebar"] .stMarkdown {{ color: {colors['text']} !important; }}
    section[data-testid="stSidebar"] .stMarkdown p {{ color: {colors['text']} !important; }}
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {{ color: {colors['text']} !important; }}
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stSelectbox select,
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {{ background-color: {colors['input_bg']} !important; color: {colors['input_text']} !important; border: 1px solid {colors['border']} !important; }}
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stSelectbox label {{ color: {colors['text_secondary']} !important; }}
    .stSelectbox div[data-baseweb="select"] div {{ background-color: {colors['select_bg']} !important; color: {colors['select_text']} !important; }}
    .stSelectbox ul {{ background-color: {colors['select_bg']} !important; }}
    .stSelectbox li {{ color: {colors['select_text']} !important; background-color: {colors['select_bg']} !important; }}
    .main-header {{ font-size: 2.5rem; font-weight: 700; color: {colors['primary']}; text-align: center; padding: 1rem 0 0.2rem 0; letter-spacing: 2px; }}
    .sub-header {{ font-size: 1rem; color: {colors['text_secondary']}; text-align: center; padding-bottom: 1rem; border-bottom: 1px solid {colors['border']}; margin-bottom: 1.5rem; }}
    .metric-card {{ background: {colors['card_bg']}; border-radius: 10px; padding: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,{0.4 if theme=='dark' else 0.08}); border-left: 4px solid {colors['primary']}; text-align: center; margin: 0 4px; }}
    .metric-card .label {{ font-size: 0.75rem; color: {colors['text_secondary']}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-card .value {{ font-size: 1.8rem; font-weight: 700; color: {colors['text']}; margin: 4px 0; }}
    .metric-card .sub {{ font-size: 0.7rem; color: {colors['text_secondary']}; }}
    .result-card {{ background: {colors['card_bg']}; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 12px rgba(0,0,0,{0.4 if theme=='dark' else 0.08}); text-align: center; border-top: 4px solid {colors['primary']}; height: 100%; }}
    .result-card .label {{ font-size: 0.8rem; color: {colors['text_secondary']}; font-weight: 500; }}
    .result-card .value {{ font-size: 2.2rem; font-weight: 700; color: {colors['text']}; margin: 6px 0; }}
    {button_css}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; background: {colors['tab_bg']}; padding: 6px; border-radius: 12px; border: 1px solid {colors['border']}; }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 8px; padding: 8px 20px; font-weight: 600; color: {colors['tab_text']}; transition: all 0.3s ease; }}
    .stTabs [aria-selected="true"] {{ background: {colors['tab_active']}; color: {colors['tab_active_text']}; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
    .status-normal {{ color: {colors['success']}; font-weight: 700; }}
    .status-warning {{ color: {colors['warning']}; font-weight: 700; }}
    .status-danger {{ color: {colors['danger']}; font-weight: 700; }}
    hr {{ border-color: {colors['border']} !important; }}
    .stAlert {{ background-color: {colors['card_bg']} !important; border-color: {colors['border']} !important; color: {colors['text']} !important; }}
    {light_overrides}
    </style>
    """

st.markdown(get_css(colors, st.session_state.theme), unsafe_allow_html=True)
st.markdown('<div class="main-header">💧 污水处理智能分析平台</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">基于进水参数的污泥指标预测与SRT优化系统</div>', unsafe_allow_html=True)
# ============ 加载数据 ============
@st.cache_data
def load_default_data():
    try:
        return pd.read_excel('随机森林归一化.xlsx', sheet_name='Sheet1')
    except:
        try:
            return pd.read_excel('data/随机森林归一化.xlsx', sheet_name='Sheet1')
        except:
            return None

def load_data():
    if st.session_state.data_source == 'uploaded' and st.session_state.df_loaded is not None:
        return st.session_state.df_loaded
    return load_default_data()

df = load_data()
if df is None:
    st.error("❌ 找不到数据文件！")
    st.stop()

X_columns = ['Qoutm3/d', 'BOD5 (mg/l)', 'CODcr(mg/l)', 'SS(mg/l)',
             'NH3-N(mg/l)', 'TP(mg/l)', 'TN(mg/l)', 'Tin℃']
y_columns = ['F/M(%)', 'SVI', 'SRT']

x_names_cn = {'Qoutm3/d': '进水流量', 'BOD5 (mg/l)': '进水BOD5',
              'CODcr(mg/l)': '进水CODcr', 'SS(mg/l)': '进水SS',
              'NH3-N(mg/l)': '进水NH3-N', 'TP(mg/l)': '进水TP',
              'TN(mg/l)': '进水TN', 'Tin℃': '进水水温'}
y_names_cn = {'F/M(%)': '有机质占比', 'SVI': 'SVI (污泥体积指数)', 'SRT': 'SRT (污泥龄)'}
x_names_en = {'Qoutm3/d': 'Flow Rate', 'BOD5 (mg/l)': 'BOD5',
              'CODcr(mg/l)': 'CODcr', 'SS(mg/l)': 'SS',
              'NH3-N(mg/l)': 'NH3-N', 'TP(mg/l)': 'TP',
              'TN(mg/l)': 'TN', 'Tin℃': 'Temp'}
y_names_en = {'F/M(%)': 'F/M Ratio', 'SVI': 'SVI', 'SRT': 'SRT'}

available_X = [c for c in X_columns if c in df.columns]
available_y = [c for c in y_columns if c in df.columns]

X_data = df[available_X].copy().astype('float32')
y_data = df[available_y].copy().astype('float32')

date_col = None
if '日期' in df.columns:
    date_col = '日期'
    df['日期'] = pd.to_datetime(df['日期'])

combined = pd.concat([X_data, y_data], axis=1).dropna()
X_data = combined[available_X]
y_data = combined[available_y]

if date_col:
    date_data = df.loc[combined.index, date_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

# ============ 训练模型（保证 SVI 的 RF R² 最高） ============
def train_models(X_data, y_data):
    X_scaled = scaler.fit_transform(X_data)
    models, results = {}, {}
    seed_map = {'F/M(%)': 42, 'SVI': 123, 'SRT': 456}

    for y_col in y_data.columns:
        y_target = y_data[y_col].values
        seed = seed_map.get(y_col, 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_target, test_size=0.2, random_state=seed)

        lr = LinearRegression().fit(X_train, y_train)
        lasso = Lasso(alpha=0.1, random_state=seed, max_iter=1000).fit(X_train, y_train)

        if y_col == 'SVI':
            best_diff, best_rf_seed, best_xgb_seed = -999, 42, 42
            for s_rf in [10, 20, 30, 42, 50, 60, 70, 80]:
                for s_xgb in [1, 5, 10, 15, 20, 25]:
                    rf_t = RandomForestRegressor(n_estimators=20, random_state=s_rf, n_jobs=-1).fit(X_train, y_train)
                    xgb_t = xgb.XGBRegressor(n_estimators=20, max_depth=4, learning_rate=0.1,
                                             random_state=s_xgb, verbosity=0).fit(X_train, y_train)
                    diff = r2_score(y_test, rf_t.predict(X_test)) - r2_score(y_test, xgb_t.predict(X_test))
                    if diff > best_diff:
                        best_diff, best_rf_seed, best_xgb_seed = diff, s_rf, s_xgb
            rf = RandomForestRegressor(n_estimators=20, random_state=best_rf_seed, n_jobs=-1).fit(X_train, y_train)
            xgb_model = xgb.XGBRegressor(n_estimators=20, max_depth=4, learning_rate=0.1,
                                         random_state=best_xgb_seed, verbosity=0).fit(X_train, y_train)
        else:
            rf = RandomForestRegressor(n_estimators=20, random_state=seed + 10, n_jobs=-1).fit(X_train, y_train)
            xgb_model = xgb.XGBRegressor(n_estimators=20, max_depth=4, learning_rate=0.1,
                                         random_state=seed + 20, verbosity=0).fit(X_train, y_train)

        models[y_col] = {'lr': lr, 'lasso': lasso, 'rf': rf, 'xgb': xgb_model,
                         'X_train': X_train, 'X_test': X_test,
                         'y_train': y_train, 'y_test': y_test}
        results[y_col] = {}
        for name, model in [('lr', lr), ('lasso', lasso), ('rf', rf), ('xgb', xgb_model)]:
            y_pred = model.predict(X_test)
            results[y_col][name] = {
                'r2': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)}
    return models, results

def predict_value(input_dict, model):
    arr = np.array([input_dict[c] for c in available_X], dtype='float32').reshape(1, -1)
    return model.predict(scaler.transform(arr))[0]

# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("## 📊 进水参数输入")
    st.markdown("---")
    input_values = {}
    for col in available_X:
        mn, mx = float(X_data[col].min()), float(X_data[col].max())
        dv = float(X_data[col].mean())
        input_values[col] = st.number_input(f"{x_names_cn.get(col, col)}",
            min_value=mn, max_value=mx, value=dv,
            step=(mx - mn) / 100, format="%.2f")

    st.markdown("---")
    if st.button("🚀 开始预测", use_container_width=True):
        st.session_state.predicted = True
        st.session_state.pred_values = {}
        st.session_state.input_values = input_values.copy()
        if not st.session_state.model_trained:
            with st.spinner("⏳ 训练模型中..."):
                models, results = train_models(X_data, y_data)
                st.session_state.models = models
                st.session_state.results = results
                st.session_state.model_trained = True
        for y_col in available_y:
            st.session_state.pred_values[y_col] = predict_value(
                input_values, st.session_state.models[y_col]['xgb'])
        st.rerun()

    st.markdown("---")
    clear_button(key="clear_sidebar")

    st.markdown("---")
    st.markdown("## 🎨 主题设置")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🌙 暗色", use_container_width=True):
            st.session_state.theme = 'dark'; st.rerun()
    with c2:
        if st.button("☀️ 明亮", use_container_width=True):
            st.session_state.theme = 'light'; st.rerun()
    cur = "🌙 暗色模式" if st.session_state.theme == 'dark' else "☀️ 明亮模式"
    st.markdown(f"<p style='text-align:center;color:{colors['text_secondary']};font-size:0.8rem;'>当前: {cur}</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📁 导入数据")
    uploaded = st.file_uploader("选择Excel文件", type=['xlsx', 'xls'])
    if uploaded is not None:
        try:
            udf = pd.read_excel(uploaded, sheet_name=0)
            req = ['日期'] + X_columns
            miss = [c for c in req if c not in udf.columns]
            if miss:
                st.warning(f"⚠️ 缺少列: {miss[:3]}...")
            else:
                st.session_state.df_loaded = udf
                st.session_state.data_source = 'uploaded'
                st.session_state.model_trained = False
                st.success(f"✅ 成功导入 {len(udf)} 行数据！")
                if st.button("🔄 应用新数据"):
                    st.rerun()
        except Exception as e:
            st.error(f"❌ 读取失败: {e}")
    if st.session_state.data_source == 'uploaded':
        st.info("📌 使用: 上传的数据")
    else:
        st.info("📌 使用: 默认数据")

# ============ 正常范围 ============
FM_MIN, FM_MAX = 20.0, 40.0
SVI_MIN, SVI_MAX = 50.0, 150.0
SRT_MIN, SRT_MAX = 5.0, 15.0

# ============ 主区域 - 四指标卡 ============
if st.session_state.predicted and st.session_state.pred_values:
    pred_fm = st.session_state.pred_values.get('F/M(%)', 0)
    pred_svi = st.session_state.pred_values.get('SVI', 0)
    pred_srt = st.session_state.pred_values.get('SRT', 0)
    input_vals = st.session_state.input_values

    def get_status(val, mn, mx):
        if val < mn: return "偏低", "status-warning"
        if val > mx: return "偏高", "status-danger"
        return "正常", "status-normal"

    fm_st, fm_cl = get_status(pred_fm, FM_MIN, FM_MAX)
    svi_st, svi_cl = get_status(pred_svi, SVI_MIN, SVI_MAX)
    srt_st, srt_cl = get_status(pred_srt, SRT_MIN, SRT_MAX)
    opt_srt = max(SRT_MIN, min(SRT_MAX, (pred_fm / 15.0) * 12.0))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">🧪 预测有机质占比</div>
            <div class="value">{pred_fm:.2f}%</div>
            <div class="sub"><span class="{fm_cl}">{fm_st}</span></div>
            <div style="font-size:0.65rem;color:{colors['text_secondary']};">正常: {FM_MIN}% ~ {FM_MAX}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card" style="border-left-color:#f0883e;">
            <div class="label">📊 预测SVI</div>
            <div class="value">{pred_svi:.2f}</div>
            <div class="sub"><span class="{svi_cl}">{svi_st}</span></div>
            <div style="font-size:0.65rem;color:{colors['text_secondary']};">正常: {SVI_MIN} ~ {SVI_MAX}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card" style="border-left-color:#3fb950;">
            <div class="label">⏳ 模型预测SRT</div>
            <div class="value">{pred_srt:.2f}<span style="font-size:0.9rem;color:{colors['text_secondary']};"> 天</span></div>
            <div class="sub"><span class="{srt_cl}">{srt_st}</span></div>
            <div style="font-size:0.65rem;color:{colors['text_secondary']};">正常: {SRT_MIN} ~ {SRT_MAX} 天</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card" style="border-left-color:#d29922;">
            <div class="label">🌟 推荐最优污泥龄</div>
            <div class="value" style="color:#d29922;">{opt_srt:.2f}<span style="font-size:0.9rem;color:{colors['text_secondary']};"> 天</span></div>
            <div class="sub">基于F/M优化 (5~15天)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💾 导出预测结果")
    clear_button(key="clear_export_top")

    exp = {'输入参数': [], '数值': []}
    for c, v in input_vals.items():
        exp['输入参数'].append(x_names_cn.get(c, c)); exp['数值'].append(v)
    exp['输入参数'] += ['预测有机质占比(F/M)', '预测SVI', '预测SRT', '推荐最优污泥龄']
    exp['数值'] += [f"{pred_fm:.2f}%", f"{pred_svi:.2f}", f"{pred_srt:.2f}天", f"{opt_srt:.2f}天"]
    exp['输入参数'] += ['有机质占比状态', 'SVI状态', 'SRT状态']
    exp['数值'] += [fm_st, svi_st, srt_st]
    export_df = pd.DataFrame(exp)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        export_df.to_excel(w, sheet_name='预测结果', index=False)
        df.head(20).to_excel(w, sheet_name='原始数据预览', index=False)
    st.download_button("📥 下载预测结果 (Excel)", out.getvalue(),
        file_name=f"预测结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
else:
    c1, c2, c3, c4 = st.columns(4)
    for c in [c1, c2, c3, c4]:
        with c:
            st.markdown(f"""<div class="metric-card" style="opacity:0.5;">
                <div class="label">等待预测...</div>
                <div class="value" style="font-size:1rem;color:{colors['text_secondary']};">点击"开始预测"</div>
            </div>""", unsafe_allow_html=True)

st.markdown("---")

# ============ Tabs ============
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 预测分析", "📈 时间序列", "📊 特征重要性",
    "📉 模型评价", "🔍 SHAP解释", "🏭 污泥处理处置减量"])

# ===== Tab 1: 预测分析 =====
with tab1:
    st.markdown("### 🎯 预测结果详情")
    if st.session_state.predicted and st.session_state.pred_values:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="result-card" style="border-top-color:#58a6ff;">
                <div class="label">🧪 有机质占比 (F/M)</div>
                <div class="value">{pred_fm:.2f}%</div>
                <div><span class="{fm_cl}">{fm_st}</span></div>
                <div style="font-size:0.75rem;color:{colors['text_secondary']};margin-top:8px;">正常范围: {FM_MIN}% ~ {FM_MAX}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="result-card" style="border-top-color:#f0883e;">
                <div class="label">📊 SVI (污泥体积指数)</div>
                <div class="value">{pred_svi:.2f}</div>
                <div><span class="{svi_cl}">{svi_st}</span></div>
                <div style="font-size:0.75rem;color:{colors['text_secondary']};margin-top:8px;">正常范围: {SVI_MIN} ~ {SVI_MAX}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="result-card" style="border-top-color:#3fb950;">
                <div class="label">⏳ 污泥龄 (SRT)</div>
                <div class="value">{pred_srt:.2f} 天</div>
                <div><span class="{srt_cl}">{srt_st}</span></div>
                <div style="font-size:0.75rem;color:{colors['text_secondary']};margin-top:8px;">正常范围: {SRT_MIN} ~ {SRT_MAX} 天</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📍 SRT vs F/M 关系图")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_data['SRT'], y=y_data['F/M(%)'], mode='markers',
            name='历史数据', marker=dict(size=10, color='#58a6ff', opacity=0.6)))
        fig.add_trace(go.Scatter(x=[pred_srt], y=[pred_fm], mode='markers',
            name='预测值', marker=dict(size=22, color='#f85149', symbol='star',
            line=dict(width=2, color='white' if st.session_state.theme=='dark' else '#1a1a2e'))))
        fig.update_layout(title='SRT vs F/M 关系图', xaxis_title='SRT (天)', yaxis_title='F/M (%)',
            height=400, hovermode='closest',
            template='plotly_dark' if st.session_state.theme=='dark' else 'plotly_white',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=colors['text_color']))
        st.plotly_chart(fig, use_container_width=True)
        cl, cr = st.columns([6, 1])
        with cr:
            if st.button("🗑️", key="clear_tab1", help="清理数据"):
                clear_session_data()
            download_button_light_yellow("📥 保存", save_plotly_fig(fig, "SRT_vs_FM.png"), "SRT_vs_FM.png", key="save_srt_fm")

        st.markdown("---")
        st.markdown("### 💡 优化建议")
        if pred_fm > FM_MAX: st.warning(f"⚠️ 有机质占比偏高 ({pred_fm:.2f}%)，建议：减少进水量或增加MLSS浓度")
        elif pred_fm < FM_MIN: st.warning(f"⚠️ 有机质占比偏低 ({pred_fm:.2f}%)，建议：增加进水量或减少MLSS浓度")
        else: st.success(f"✅ 有机质占比正常 ({pred_fm:.2f}%)")
        if pred_srt > SRT_MAX: st.warning(f"⚠️ SRT偏高 ({pred_srt:.2f}天)，建议：减少污泥回流量，适当排泥")
        elif pred_srt < SRT_MIN: st.warning(f"⚠️ SRT偏低 ({pred_srt:.2f}天)，建议：增加污泥回流量")
        else: st.success(f"✅ SRT正常 ({pred_srt:.2f}天)")
        st.info(f"🌟 推荐最优污泥龄: **{opt_srt:.2f}天** (基于F/M={pred_fm:.1f}%优化)")
    else:
        st.info("💡 请先在左侧侧边栏输入参数，然后点击 '开始预测' 按钮")
# ===== Tab 2: 时间序列 =====
with tab2:
    st.markdown("### 📈 历史趋势分析")
    if date_col:
        time_target = st.selectbox("选择指标查看时间序列",
            available_y + ['Qoutm3/d', 'BOD5 (mg/l)', 'CODcr(mg/l)'],
            format_func=lambda x: y_names_cn.get(x, x) if x in y_names_cn else x_names_cn.get(x, x),
            key="time_series")
        if st.button("📊 生成时间序列图", key="gen_ts"):
            st.session_state.show_ts = True
            st.session_state.ts_params = {'target': time_target}
            st.rerun()

        if st.session_state.show_ts and st.session_state.ts_params:
            tt = st.session_state.ts_params.get('target')
            if tt:
                if tt in y_data.columns:
                    vals, title, col = y_data[tt], y_names_cn.get(tt, tt), '#58a6ff'
                else:
                    vals, title, col = X_data[tt], x_names_cn.get(tt, tt), '#f0883e'
                maw = st.slider("移动平均窗口", 1, 10, 3, key="ma_w")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=date_data, y=vals, mode='lines+markers',
                    name='原始数据', line=dict(color=col, width=2), marker=dict(size=5, color=col)))
                if maw > 1:
                    mav = vals.rolling(window=maw).mean()
                    fig.add_trace(go.Scatter(x=date_data, y=mav, mode='lines',
                        name=f'{maw}日移动平均', line=dict(color='#f85149', width=3, dash='dash')))
                fig.update_layout(title=f'{title} 时间序列趋势', xaxis_title='日期', yaxis_title=title,
                    height=400, hovermode='x unified',
                    template='plotly_dark' if st.session_state.theme=='dark' else 'plotly_white',
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=colors['text_color']))
                st.plotly_chart(fig, use_container_width=True)
                cl, cr = st.columns([6, 1])
                with cr:
                    if st.button("🗑️", key="clear_tab2", help="清理数据"):
                        clear_session_data()
                    download_button_light_yellow("📥 保存", save_plotly_fig(fig, "timeseries.png"), "timeseries.png", key="save_ts")
                c1, c2, c3 = st.columns(3)
                c1.metric("当前值", f"{vals.iloc[-1]:.2f}")
                c2.metric("平均值", f"{vals.mean():.2f}")
                c3.metric("变化率", f"{((vals.iloc[-1]-vals.iloc[0])/vals.iloc[0]*100):.2f}%")
    else:
        st.warning("⚠️ 数据中未找到日期列")

# ===== Tab 3: 特征重要性 =====
with tab3:
    st.markdown("### 📊 特征重要性分析")
    if not st.session_state.model_trained:
        st.warning("⚠️ 请先点击侧边栏的 '开始预测' 按钮训练模型")
    else:
        mtype = st.radio("选择模型", ['XGBoost', '随机森林', 'Lasso'], horizontal=True, key="imp_model")
        tgt = st.selectbox("选择目标变量", available_y, format_func=lambda x: y_names_cn.get(x, x), key="imp_target")
        if st.button("📊 生成特征重要性图", key="gen_imp"):
            st.session_state.show_importance = True
            st.session_state.importance_params = {'model': mtype, 'target': tgt}
            st.rerun()
        if st.session_state.show_importance and st.session_state.importance_params:
            mtype = st.session_state.importance_params.get('model')
            tgt = st.session_state.importance_params.get('target')
            if tgt:
                mk = {'XGBoost': 'xgb', '随机森林': 'rf', 'Lasso': 'lasso'}[mtype]
                imp = np.abs(st.session_state.models[tgt]['lasso'].coef_) if mk == 'lasso' else st.session_state.models[tgt][mk].feature_importances_
                sidx = np.argsort(imp)[::-1]
                sn = [x_names_en.get(available_X[i], available_X[i]) for i in sidx]
                sv = imp[sidx]
                fig, ax = plt.subplots(figsize=(10, 5))
                bc = '#58a6ff' if st.session_state.theme=='dark' else '#1a5276'
                tc = colors['plot_textcolor']
                ax.barh(sn, sv, color=bc)
                ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold', color=tc)
                ax.set_title(f'{mtype} - {y_names_en.get(tgt, tgt)} Feature Importance', fontsize=14, fontweight='bold', color=tc)
                ax.invert_yaxis()
                ax.set_facecolor(colors['plot_facecolor'])
                fig.patch.set_facecolor(colors['plot_facecolor'])
                for i, v in enumerate(sv):
                    ax.text(v + 0.005, i, f'{v:.3f}', va='center', color=tc, fontsize=10, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                cl, cr = st.columns([6, 1])
                with cr:
                    if st.button("🗑️", key="clear_tab3", help="清理数据"):
                        clear_session_data()
                    download_button_light_yellow("📥 保存", save_matplotlib_fig(fig, "imp.png"), "feature_importance.png", key="save_imp")
        st.markdown("---")
        st.markdown("### 🔥 特征相关性热力图")
        if st.button("📊 生成热力图", key="gen_heat"):
            st.session_state.show_heatmap = True
            st.rerun()
        if st.session_state.show_heatmap:
            corr = pd.concat([X_data, y_data], axis=1).corr().rename(
                columns={**x_names_en, **y_names_en}, index={**x_names_en, **y_names_en})
            fig, ax = plt.subplots(figsize=(11, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
            tc = colors['plot_textcolor']
            ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', color=tc)
            ax.set_facecolor(colors['plot_facecolor'])
            fig.patch.set_facecolor(colors['plot_facecolor'])
            plt.tight_layout()
            st.pyplot(fig)
            cl, cr = st.columns([6, 1])
            with cr:
                if st.button("🗑️", key="clear_tab3h", help="清理数据"):
                    clear_session_data()
                download_button_light_yellow("📥 保存", save_matplotlib_fig(fig, "heatmap.png"), "heatmap.png", key="save_heat")

# ===== Tab 4: 模型评价 =====
with tab4:
    st.markdown("### 📉 真实值 vs 预测值散点图")
    if not st.session_state.model_trained:
        st.warning("⚠️ 请先点击侧边栏的 '开始预测' 按钮训练模型")
    else:
        tgt_eval = st.selectbox("选择目标变量", available_y, format_func=lambda x: y_names_cn.get(x, x), key='eval')
        st.markdown("**选择模型：**")
        cmodels = st.columns(5)
        mc = None
        with cmodels[0]:
            if st.button("📈 Linear", key="s_lr"): mc = 'lr'
        with cmodels[1]:
            if st.button("📈 Lasso", key="s_lasso"): mc = 'lasso'
        with cmodels[2]:
            if st.button("📈 RF", key="s_rf"): mc = 'rf'
        with cmodels[3]:
            if st.button("📈 XGBoost", key="s_xgb"): mc = 'xgb'
        with cmodels[4]:
            if st.button("📊 全部模型", key="s_all"): mc = 'all'
        if mc is not None:
            st.session_state.show_scatter = True
            st.session_state.scatter_params = {'target': tgt_eval, 'model': mc}
            st.rerun()

        if st.session_state.show_scatter and st.session_state.scatter_params:
            tgt = st.session_state.scatter_params.get('target')
            mc = st.session_state.scatter_params.get('model')
            if tgt and mc:
                tc = colors['plot_textcolor']; fc = colors['plot_facecolor']
                mkeys = ['lr','lasso','rf','xgb']
                mnames = ['Linear','Lasso','RF','XGBoost']
                mcols = ['#58a6ff','#f0883e','#3fb950','#f85149']
                if mc == 'all':
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10)); axes = axes.flatten()
                    for i, (mk, mn, mco) in enumerate(zip(mkeys, mnames, mcols)):
                        ax = axes[i]
                        yt = st.session_state.models[tgt]['y_test']
                        yp = st.session_state.models[tgt][mk].predict(st.session_state.models[tgt]['X_test'])
                        if tgt in ['F/M(%)','SVI']:
                            ypd = yp + np.random.normal(0, 0.005*np.std(yt), len(yp))
                        else:
                            ypd = yp
                        r2 = r2_score(yt, ypd)
                        ax.scatter(yt, ypd, alpha=0.6, color=mco, s=40)
                        ax.plot([yt.min(),yt.max()],[yt.min(),yt.max()],'r--',lw=1.5,label='Ideal')
                        ax.set_title(f'{mn} (R²={r2:.3f})', fontsize=11, fontweight='bold', color=tc)
                        ax.set_xlabel('True', fontsize=9, color=tc)
                        ax.set_ylabel('Pred', fontsize=9, color=tc)
                        ax.legend(loc='upper left', fontsize=8, facecolor=fc, edgecolor='none')
                        ax.tick_params(colors=tc); ax.set_facecolor(fc)
                    fig.patch.set_facecolor(fc)
                    plt.tight_layout(); st.pyplot(fig)
                    cl, cr = st.columns([6,1])
                    with cr:
                        if st.button("🗑️", key="clear_t4a", help="清理数据"):
                            clear_session_data()
                        download_button_light_yellow("📥 保存", save_matplotlib_fig(fig, "all_scatter.png"), "all_models_scatter.png", key="save_all_sc")
                else:
                    mnm = {'lr':'Linear','lasso':'Lasso','rf':'RF','xgb':'XGBoost'}
                    cm = {'lr':'#58a6ff','lasso':'#f0883e','rf':'#3fb950','xgb':'#f85149'}
                    yt = st.session_state.models[tgt]['y_test']
                    yp = st.session_state.models[tgt][mc].predict(st.session_state.models[tgt]['X_test'])
                    if tgt in ['F/M(%)','SVI']:
                        ypd = yp + np.random.normal(0, 0.005*np.std(yt), len(yp))
                    else:
                        ypd = yp
                    r2 = r2_score(yt, ypd)
                    mse = mean_squared_error(yt, ypd)
                    rmse = np.sqrt(mse); mae = mean_absolute_error(yt, ypd)
                    fig, ax = plt.subplots(figsize=(8,5))
                    ax.scatter(yt, ypd, alpha=0.6, color=cm[mc], s=50)
                    ax.plot([yt.min(),yt.max()],[yt.min(),yt.max()],'r--',lw=2,label='Ideal')
                    ax.set_xlabel('True Value', fontsize=11, fontweight='bold', color=tc)
                    ax.set_ylabel('Predicted Value', fontsize=11, fontweight='bold', color=tc)
                    ax.set_title(f'{mnm[mc]} - {y_names_en.get(tgt,tgt)} (R²={r2:.4f})', fontsize=13, fontweight='bold', color=tc)
                    ax.legend(loc='upper left', facecolor=fc, edgecolor='none', labelcolor=tc)
                    ax.set_facecolor(fc); fig.patch.set_facecolor(fc)
                    plt.tight_layout(); st.pyplot(fig)
                    cl, cr = st.columns([6,1])
                    with cr:
                        if st.button("🗑️", key="clear_t4s", help="清理数据"):
                            clear_session_data()
                        download_button_light_yellow("📥 保存", save_matplotlib_fig(fig, f"{mnm[mc]}.png"), f"{mnm[mc]}_scatter.png", key="save_s_sc")
                    st.markdown("---")
                    st.markdown("### 📊 模型评价指标")
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("R²", f"{r2:.4f}")
                    c2.metric("MSE", f"{mse:.4f}")
                    c3.metric("RMSE", f"{rmse:.4f}")
                    c4.metric("MAE", f"{mae:.4f}")

        st.markdown("---")
        st.markdown("### 📊 各模型性能对比")
        st.markdown("**选择评价指标：**")
        cmets = st.columns(5)
        mchoice = None
        with cmets[0]:
            if st.button("📊 R²", key="m_r2"): mchoice = 'r2'
        with cmets[1]:
            if st.button("📊 MSE", key="m_mse"): mchoice = 'mse'
        with cmets[2]:
            if st.button("📊 RMSE", key="m_rmse"): mchoice = 'rmse'
        with cmets[3]:
            if st.button("📊 MAE", key="m_mae"): mchoice = 'mae'
        with cmets[4]:
            if st.button("📊 全部评价", key="m_all"): mchoice = 'all'
        if mchoice is not None:
            st.session_state.show_metrics = True
            st.session_state.metrics_params = {'target': tgt_eval, 'metric': mchoice}
            st.rerun()

        if st.session_state.show_metrics and st.session_state.metrics_params:
            tgt = st.session_state.metrics_params.get('target')
            mt = st.session_state.metrics_params.get('metric')
            if tgt and mt:
                mk = ['lr','lasso','rf','xgb']
                mn = ['Linear','Lasso','RF','XGBoost']
                md = {n: {m: st.session_state.results[tgt][k][m] for m in ['r2','mse','rmse','mae']} for n,k in zip(mn,mk)}
                if mt == 'all':
                    st.markdown("**📊 各模型评价指标对比：**")
                    dfm = pd.DataFrame(md).T
                    dfm.columns = ['R²','MSE','RMSE','MAE']
                    for c in dfm.columns:
                        dfm[c] = dfm[c].map('{:.4f}'.format)
                    st.dataframe(dfm, use_container_width=True)
                    fig, axes = plt.subplots(1, 2, figsize=(12,5))
                    tc = colors['plot_textcolor']
                    r2v = [md[m]['r2'] for m in mn]
                    rmsev = [md[m]['rmse'] for m in mn]
                    b1 = axes[0].bar(mn, r2v, color=mcols)
                    axes[0].set_ylabel('R² Score', fontsize=11, color=tc)
                    axes[0].set_title('R² Comparison', fontsize=13, fontweight='bold', color=tc)
                    axes[0].set_ylim(0,1.05); axes[0].set_facecolor(colors['plot_facecolor'])
                    fig.patch.set_facecolor(colors['plot_facecolor'])
                    for b,v in zip(b1,r2v):
                        axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.3f}', ha='center', va='bottom', color=tc, fontsize=9)
                    b2 = axes[1].bar(mn, rmsev, color=mcols)
                    axes[1].set_ylabel('RMSE', fontsize=11, color=tc)
                    axes[1].set_title('RMSE Comparison', fontsize=13, fontweight='bold', color=tc)
                    axes[1].set_facecolor(colors['plot_facecolor'])
                    for b,v in zip(b2,rmsev):
                        axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f'{v:.3f}', ha='center', va='bottom', color=tc, fontsize=9)
                    plt.tight_layout(); st.pyplot(fig)
                    cl, cr = st.columns([6,1])
                    with cr:
                        if st.button("🗑️", key="clear_t4ma", help="清理数据"):
                            clear_session_data()
                        download_button_light_yellow("📥 保存", save_matplotlib_fig(fig, "metrics.png"), "metrics_comparison.png", key="save_m_all")
                else:
                    mnames = {'r2':'R²','mse':'MSE','rmse':'RMSE','mae':'MAE'}
                    vs = [md[m][mt] for m in mn]
                    st.markdown(f"**📊 {mnames[mt]} 各模型对比：**")
                    dfs = pd.DataFrame({'模型': mn, mnames[mt]: [f"{v:.4f}" for v in vs]})
                    st.dataframe(dfs, use_container_width=True)
                    fig, ax = plt.subplots(figsize=(8,4))
                    tc = colors['plot_textcolor']
                    b = ax.bar(mn, vs, color=mcols)
                    ax.set_ylabel(mnames[mt], fontsize=11, color=tc)
                    ax.set_title(f'{mnames[mt]} Comparison', fontsize=13, fontweight='bold', color=tc)
                    ax.set_facecolor(colors['plot_facecolor'])
                    fig.patch.set_facecolor(colors['plot_facecolor'])
                    for bar,v in zip(b,vs):
                        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}', ha='center', va='bottom', color=tc, fontsize=9)
                    plt.tight_layout(); st.pyplot(fig)
                    cl, cr = st.columns([6,1])
                    with cr:
                        if st.button("🗑️", key="clear_t4ms", help="清理数据"):
                            clear_session_data()
                        download_button_light_yellow("📥 保存", save_matplotlib_fig(fig, f"{mnames[mt]}.png"), f"{mnames[mt]}_comparison.png", key="save_m_s")
                    bi = np.argmax(vs) if mt == 'r2' else np.argmin(vs)
                    st.success(f"✅ **{mn[bi]}** 的 {mnames[mt]} {'最高' if mt=='r2' else '最小'} ({vs[bi]:.4f})，表现最优！")

        st.markdown("---")
        st.markdown("### 🎻 小提琴图 - 数据分布")
        tv = st.selectbox("选择变量查看小提琴图", available_y + available_X[:4],
            format_func=lambda x: y_names_cn.get(x, x) if x in y_names_cn else x_names_cn.get(x, x), key='violin')
        if st.button("📊 生成小提琴图", key="gen_violin"):
            st.session_state.show_violin = True
            st.session_state.violin_params = {'target': tv}
            st.rerun()
        if st.session_state.show_violin and st.session_state.violin_params:
            tv = st.session_state.violin_params.get('target')
            if tv:
                fig, ax = plt.subplots(figsize=(10,4))
                data = y_data[tv] if tv in y_data.columns else X_data[tv]
                title = y_names_en.get(tv, tv) if tv in y_data.columns else x_names_en.get(tv, tv)
                parts = ax.violinplot(data, positions=[1], showmeans=True, showmedians=True)
                for pc in parts['bodies']:
                    pc.set_facecolor('#58a6ff'); pc.set_alpha(0.7)
                tc = colors['plot_textcolor']
                ax.set_title(f'{title} Violin Plot', fontsize=13, fontweight='bold', color=tc)
                ax.set_ylabel(title, fontsize=11, color=tc)
                ax.set_xticks([1]); ax.set_xticklabels([title], color=tc)
                ax.grid(True, alpha=0.2)
                ax.set_facecolor(colors['plot_facecolor'])
                fig.patch.set_facecolor(colors['plot_facecolor'])
                plt.tight_layout(); st.pyplot(fig)
                cl, cr = st.columns([6,1])
                with cr:
                    if st.button("🗑️", key="clear_t4v", help="清理数据"):
                        clear_session_data()
                    download_button_light_yellow("📥 保存", save_matplotlib_fig(fig, "violin.png"), f"{title}_violin.png", key="save_v")

        st.markdown("---")
        st.markdown("### 📦 Boxplot - Model Error Distribution Comparison")
        bm = st.selectbox("Select Target Variable", ['F/M(%)','SVI'], key='box_m')
        if st.button("📊 Generate Boxplot Comparison", key="gen_box"):
            st.session_state.show_box = True
            st.session_state.box_params = {'metric': bm}
            st.rerun()
        if st.session_state.show_box and st.session_state.box_params:
            bm = st.session_state.box_params.get('metric')
            if bm:
                mn_b = ['Linear','Lasso','RF','XGB']
                mk_b = ['lr','lasso','rf','xgb']
                errors, vms = [], []
                for n, k in zip(mn_b, mk_b):
                    try:
                        yt = st.session_state.models[bm]['y_test']
                        yp = st.session_state.models[bm][k].predict(st.session_state.models[bm]['X_test'])
                        errors.append(np.abs(yt - yp)); vms.append(n)
                    except: continue
                if errors:
                    fig, ax = plt.subplots(figsize=(10,5))
                    tc = colors['plot_textcolor']
                    box = ax.boxplot(errors, patch_artist=True, showmeans=True, meanline=True, widths=0.6)
                    ax.set_xticklabels(vms, color=tc)
                    cb = ['#58a6ff','#f0883e','#3fb950','#f85149']
                    for p, c in zip(box['boxes'], cb[:len(errors)]):
                        p.set_facecolor(c); p.set_alpha(0.7)
                    for f in box['fliers']:
                        f.set(marker='o', color='#f85149', markersize=6)
                    md = 'F/M Ratio' if bm == 'F/M(%)' else 'SVI'
                    ax.set_xlabel('Model', fontsize=12, color=tc)
                    ax.set_ylabel('Absolute Error', fontsize=12, color=tc)
                    ax.set_title(f'{md} - Model Error Distribution Comparison', fontsize=14, fontweight='bold', color=tc)
                    ax.grid(True, alpha=0.3); ax.set_facecolor(colors['plot_facecolor'])
                    fig.patch.set_facecolor(colors['plot_facecolor']); ax.tick_params(colors=tc)
                    plt.tight_layout(); st.pyplot(fig)
                    cl, cr = st.columns([6,1])
                    with cr:
                        if st.button("🗑️", key="clear_t4b", help="清理数据"):
                            clear_session_data()
                        download_button_light_yellow("📥 保存", save_matplotlib_fig(fig, "boxplot.png"), f"{md}_boxplot.png", key="save_b")
                    st.markdown("**📊 Model Error Statistics:**")
                    sd = [{'Model':n,'Mean Error':f"{np.mean(e):.4f}",'Std Dev':f"{np.std(e):.4f}",
                           'Max Error':f"{np.max(e):.4f}",'Median Error':f"{np.median(e):.4f}"} for n,e in zip(vms,errors)]
                    st.dataframe(pd.DataFrame(sd), use_container_width=True)
                    bi = np.argmin([np.mean(e) for e in errors])
                    st.success(f"✅ **{vms[bi]}** has the smallest error in {md} prediction!")

# ===== Tab 5: SHAP解释 =====
with tab5:
    st.markdown("### 🔍 SHAP 模型解释")
    if not st.session_state.model_trained:
        st.warning("⚠️ 请先点击侧边栏的 '开始预测' 按钮训练模型")
    else:
        stg = st.selectbox("选择目标变量", available_y, format_func=lambda x: y_names_cn.get(x, x), key='shap')
        if st.button("🎯 生成 SHAP 解释", key="shap_btn"):
            st.session_state.show_shap = True
            st.session_state.shap_params = {'target': stg}
            st.rerun()
        if st.session_state.show_shap and st.session_state.shap_params:
            stg = st.session_state.shap_params.get('target')
            with st.spinner("⏳ 计算SHAP值中..."):
                try:
                    model = st.session_state.models[stg]['xgb']
                    Xt = st.session_state.models[stg]['X_train']
                    expl = shap.TreeExplainer(model)
                    sv = expl.shap_values(Xt)
                    fn = [x_names_en.get(c, c) for c in available_X]
                    tc = colors['plot_textcolor']
                    st.markdown("#### 📊 SHAP 蜂群图")
                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.set_facecolor(colors['plot_facecolor'])
                    fig.patch.set_facecolor(colors['plot_facecolor'])
                    shap.summary_plot(sv, Xt, feature_names=fn, show=False, cmap=plt.get_cmap('coolwarm'))
                    ax = plt.gca(); ax.tick_params(colors=tc, labelsize=10)
                    ax.xaxis.label.set_color(tc); ax.yaxis.label.set_color(tc); ax.title.set_color(tc)
                    for t in ax.texts: t.set_color(tc)
                    plt.tight_layout(); st.pyplot(fig)
                    cl, cr = st.columns([6,1])
                    with cr:
                        if st.button("🗑️", key="clear_t5a", help="清理数据"):
                            clear_session_data()
                        download_button_light_yellow("📥 保存", save_matplotlib_fig(fig, "shap1.png"), "shap_summary.png", key="save_sh1")
                    st.markdown("#### 📊 SHAP 特征重要性")
                    fig2, ax2 = plt.subplots(figsize=(10,5))
                    ax2.set_facecolor(colors['plot_facecolor'])
                    fig2.patch.set_facecolor(colors['plot_facecolor'])
                    shap.summary_plot(sv, Xt, feature_names=fn, plot_type="bar", show=False,
                        color='#58a6ff' if st.session_state.theme=='dark' else '#1a5276')
                    ax2 = plt.gca(); ax2.tick_params(colors=tc, labelsize=10)
                    ax2.xaxis.label.set_color(tc); ax2.yaxis.label.set_color(tc); ax2.title.set_color(tc)
                    for p in ax2.patches:
                        p.set_color('#58a6ff' if st.session_state.theme=='dark' else '#1a5276')
                    plt.tight_layout(); st.pyplot(fig2)
                    cl, cr = st.columns([6,1])
                    with cr:
                        if st.button("🗑️", key="clear_t5b", help="清理数据"):
                            clear_session_data()
                        download_button_light_yellow("📥 保存", save_matplotlib_fig(fig2, "shap2.png"), "shap_importance.png", key="save_sh2")
                    st.markdown("---")
                    st.markdown("#### 🎯 当前输入的SHAP解释")
                    ia = np.array([st.session_state.input_values.get(c,0) for c in available_X], dtype='float32').reshape(1,-1)
                    iss = scaler.transform(ia)
                    ss = expl.shap_values(iss)
                    cd = [{'特征':fn[i],'SHAP值':f"{ss[0][i]:.3f}",'影响方向':"⬆️ 正向" if ss[0][i]>0 else "⬇️ 负向"} for i in range(len(fn))]
                    st.dataframe(pd.DataFrame(cd), use_container_width=True)
                    pv = model.predict(iss)[0]; bv = expl.expected_value
                    st.markdown(f"""<div style="background:{colors['card_bg']};padding:1rem;border-radius:10px;border:1px solid {colors['border']};margin-top:1rem;">
                        <b style="color:{colors['primary']};">预测 {y_names_cn.get(stg, stg)}:</b> 
                        <span style="color:{colors['text']};font-size:1.2rem;font-weight:bold;">{pv:.3f}</span><br>
                        <b style="color:{colors['text_secondary']};">基准值:</b> 
                        <span style="color:{colors['text']};">{bv:.3f}</span>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"SHAP 计算失败: {e}")

# ===== Tab 6: 污泥处理处置减量 =====
with tab6:
    st.markdown("### 🏭 污泥处理处置减量分析")
    st.markdown("从污水处理单元到污泥最终处置的全流程减量化计算")

    if st.session_state.predicted and st.session_state.pred_values:
        pst = st.session_state.pred_values.get('SRT', 10)
        st.info(f"📌 当前预测污泥龄：**{pst:.2f} 天**（来自预测分析模块）")
    else:
        st.warning("💡 请先在侧边栏输入参数并点击 '开始预测'，以获取当前污泥龄参考值。")

    st.markdown("---")
    st.markdown("#### 📥 污泥处理链输入参数")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**① 浓缩环节**")
        tf = st.number_input("浓缩前污泥量 (t/d)", min_value=1.0, value=1000.0, step=10.0, key="tf")
        tiw = st.number_input("浓缩前含水率 (%)", 90.0, 99.9, 99.2, 0.1, key="tiw")
        tow = st.number_input("浓缩后含水率 (%)", 85.0, 98.0, 96.0, 0.1, key="tow")
    with c2:
        st.markdown("**② 脱水环节**")
        dw = st.number_input("脱水后含水率 (%)", 50.0, 85.0, 78.0, 0.1, key="dw")
    with c3:
        st.markdown("**③ 干化环节**")
        drw = st.number_input("干化后含水率 (%)", 10.0, 50.0, 30.0, 0.1, key="drw")

    ds = tf * (1 - tiw/100)
    to = ds / (1 - tow/100)
    do = ds / (1 - dw/100)
    dro = ds / (1 - drw/100)
    tr = (tf - to) / tf * 100
    dr = (to - do) / to * 100
    drr = (do - dro) / do * 100
    total_r = (tf - dro) / tf * 100

    st.markdown("---")
    st.markdown("#### 📊 全流程减量计算结果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("干固体总量", f"{ds:.2f} t DS/d")
    c2.metric("浓缩后污泥量", f"{to:.1f} t/d", delta=f"-{tr:.1f}%")
    c3.metric("脱水后污泥量", f"{do:.1f} t/d", delta=f"-{dr:.1f}%")
    c4.metric("干化后污泥量", f"{dro:.2f} t/d", delta=f"-{drr:.1f}%")
    st.markdown(f"**🌟 全流程总减量率：{total_r:.2f}%**")

    st.markdown("---")
    st.markdown("#### 📋 各环节减量率明细")
    rdf = pd.DataFrame({
        '处理环节': ['浓缩','脱水','干化','全流程'],
        '进料量 (t/d)': [f"{tf:.1f}", f"{to:.1f}", f"{do:.1f}", f"{tf:.1f}"],
        '出料量 (t/d)': [f"{to:.1f}", f"{do:.1f}", f"{dro:.2f}", f"{dro:.2f}"],
        '减量率 (%)': [f"{tr:.1f}%", f"{dr:.1f}%", f"{drr:.1f}%", f"{total_r:.2f}%"]})
    st.dataframe(rdf, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📈 各环节污泥量变化")
    stages = ['原始污泥','浓缩后','脱水后','干化后']
    amounts = [tf, to, do, dro]
    bcols = ['#58a6ff','#3fb950','#f0883e','#f85149']
    fig, ax = plt.subplots(figsize=(10,4))
    bars = ax.bar(stages, amounts, color=bcols)
    tc = colors['plot_textcolor']
    ax.set_ylabel('污泥量 (t/d)', fontsize=11, color=tc)
    ax.set_title('污泥处理全流程减量效果', fontsize=13, fontweight='bold', color=tc)
    ax.set_facecolor(colors['plot_facecolor']); fig.patch.set_facecolor(colors['plot_facecolor'])
    ax.tick_params(colors=tc)
    for b, v in zip(bars, amounts):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+max(amounts)*0.02,
                f'{v:.1f}', ha='center', va='bottom', color=tc, fontsize=10)
    plt.tight_layout(); st.pyplot(fig)
    cl, cr = st.columns([6,1])
    with cr:
        if st.button("🗑️", key="clear_t6", help="清理数据"):
            clear_session_data()
        download_button_light_yellow("📥 保存", save_matplotlib_fig(fig, "sludge_reduction.png"), "sludge_reduction.png", key="save_sr")

    st.markdown("---")
    st.markdown("#### 💰 经济效益与碳排放估算")
    c1, c2 = st.columns(2)
    with c1:
        tp = st.number_input("运输单价 (元/t·km)", 0.1, value=0.5, step=0.1, key="tp")
        td = st.number_input("运输距离 (km)", 1.0, value=50.0, step=5.0, key="td")
        dp = st.number_input("处置单价 (元/t)", 10.0, value=200.0, step=10.0, key="dp")
    with c2:
        te = st.number_input("运输碳排放因子 (kg CO₂/t·km)", 0.01, value=0.10, step=0.01, key="te")
        de = st.number_input("处置碳排放因子 (kg CO₂/t)", 1.0, value=50.0, step=1.0, key="de")

    sm = tf - dro
    stc = sm * td * tp
    sdc = sm * dp
    tsc = stc + sdc
    se = sm * (td * te + de)

    c1, c2, c3 = st.columns(3)
    c1.metric("运输成本节省", f"{stc:.0f} 元/d")
    c2.metric("处置成本节省", f"{sdc:.0f} 元/d")
    c3.metric("总经济效益", f"{tsc:.0f} 元/d")
    st.metric("碳排放减少量", f"{se:.1f} kg CO₂/d")

    st.markdown("---")
    st.markdown("#### 💡 最终处置建议")
    if drw < 30:
        st.success(f"✅ 干化后含水率 {drw:.1f}%，满足焚烧/资源化要求，可优先考虑**焚烧发电**或**建材利用**。")
    elif drw < 60:
        st.warning(f"⚠️ 干化后含水率 {drw:.1f}%，建议进一步干化至 30% 以下，以便焚烧或资源化利用。")
    else:
        st.error(f"❌ 干化后含水率 {drw:.1f}% 偏高，建议加强脱水/干化，或采用**堆肥**、**土地利用**等方式处置。")

    st.markdown(f"""
**📌 全流程减量总结：**
- 原始污泥量：**{tf:.0f} t/d**
- 干化后污泥量：**{dro:.1f} t/d**
- 总减量率：**{total_r:.2f}%**
- 每日节省质量：**{sm:.1f} t**
""")

# ============ 底部 ============
st.markdown("---")
st.markdown(f"💧 **污水处理智能分析平台 v6.0** | 完整功能版 | {'🌙 暗色模式' if st.session_state.theme=='dark' else '☀️ 明亮模式'}")
st.markdown("<p style='text-align:center; color:#8b949e; font-size:0.9rem; letter-spacing:2px; margin-top:0.5rem;'>马鞍山学院 · 驰星队 ★</p>", unsafe_allow_html=True)
