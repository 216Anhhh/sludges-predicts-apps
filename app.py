# ===== 必须在所有import之前添加 =====
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
# ====================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="污水处理智能分析平台",
    page_icon="💧",
    layout="wide"
)

# 自定义CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a73e8;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        padding-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<div class="main-header">💧 污水处理智能分析平台</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">基于XGBoost / 随机森林 / Lasso 的多模型预测与优化系统</div>',
            unsafe_allow_html=True)


# 加载数据
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('数据/随机森林归一化.xlsx', sheet_name='Sheet1')
        return df
    except:
        # 尝试读取当前目录
        try:
            df = pd.read_excel('随机森林归一化.xlsx', sheet_name='Sheet1')
            return df
        except:
            st.error("❌ 找不到数据文件！请确保 '随机森林归一化.xlsx' 在 'data' 文件夹或当前目录下。")
            return None


df = load_data()

if df is not None:
    # 定义列名
    X_columns = ['Qoutm3/d', 'BOD5 (mg/l)', 'CODcr(mg/l)', 'SS(mg/l)',
                 'NH3-N(mg/l)', 'TP(mg/l)', 'TN(mg/l)', 'Tin℃']
    y_columns = ['SRT', 'F/M(%)', 'SVI']

    # 检查列是否存在
    available_X = [col for col in X_columns if col in df.columns]
    available_y = [col for col in y_columns if col in df.columns]

    # 提取数据
    X_data = df[available_X].copy()
    y_data = df[available_y].copy()

    # 删除缺失值
    combined = pd.concat([X_data, y_data], axis=1).dropna()
    X_data = combined[available_X]
    y_data = combined[available_y]

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    # 中文名称映射
    x_names_cn = {
        'Qoutm3/d': '出水量 (m³/d)',
        'BOD5 (mg/l)': 'BOD5 (mg/L)',
        'CODcr(mg/l)': 'CODcr (mg/L)',
        'SS(mg/l)': 'SS (mg/L)',
        'NH3-N(mg/l)': 'NH3-N (mg/L)',
        'TP(mg/l)': 'TP (mg/L)',
        'TN(mg/l)': 'TN (mg/L)',
        'Tin℃': '进水温度 (°C)'
    }
    y_names_cn = {
        'SRT': 'SRT (污泥龄)',
        'F/M(%)': '有机质占比 (F/M)',
        'SVI': 'SVI (污泥体积指数)'
    }

    # ============ 侧边栏：输入参数 ============
    st.sidebar.markdown("## 📊 输入参数")
    st.sidebar.markdown("---")

    input_values = {}
    for col in available_X:
        min_val = float(X_data[col].min())
        max_val = float(X_data[col].max())
        default_val = float(X_data[col].mean())
        input_values[col] = st.sidebar.number_input(
            f"{x_names_cn.get(col, col)}",
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=(max_val - min_val) / 100,
            format="%.2f"
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🤖 模型选择")

    # 模型选择
    show_lasso = st.sidebar.checkbox("查看 Lasso 回归", value=True)
    show_rf = st.sidebar.checkbox("查看 随机森林", value=True)
    show_xgb = st.sidebar.checkbox("查看 XGBoost", value=True)
    show_heatmap = st.sidebar.checkbox("查看 热力图", value=True)

    # 一键勾选
    if st.sidebar.button("📌 一键勾选全部"):
        st.session_state.show_all = True
        st.rerun()

    if st.sidebar.button("🗑️ 取消全部选择"):
        st.session_state.show_all = False
        st.rerun()

    if 'show_all' in st.session_state and st.session_state.show_all:
        show_lasso = show_rf = show_xgb = show_heatmap = True


    # ============ 训练模型 ============
    @st.cache_resource
    def train_models(X_data, y_data):
        X_scaled = scaler.fit_transform(X_data)
        models = {}
        results = {}

        for y_col in y_data.columns:
            y_target = y_data[y_col].values
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_target, test_size=0.2, random_state=42
            )

            # 线性回归
            lr = LinearRegression()
            lr.fit(X_train, y_train)

            # Lasso
            lasso = Lasso(alpha=0.1, random_state=42)
            lasso.fit(X_train, y_train)

            # 随机森林
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)

            # XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1,
                                         max_depth=5, random_state=42)
            xgb_model.fit(X_train, y_train)

            models[y_col] = {
                'lr': lr, 'lasso': lasso, 'rf': rf, 'xgb': xgb_model,
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }

            # 评估
            results[y_col] = {}
            for name, model in [('lr', lr), ('lasso', lasso), ('rf', rf), ('xgb', xgb_model)]:
                y_pred = model.predict(X_test)
                results[y_col][name] = {
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                }

        return models, results


    models, results = train_models(X_data, y_data)


    # ============ 预测 ============
    def predict_value(input_dict, model, scaler):
        input_array = np.array([input_dict[col] for col in available_X]).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        return model.predict(input_scaled)[0]


    # ============ 主区域 ============
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 特征重要性分析",
        "📊 模型对比",
        "🎯 预测与评估",
        "🔧 优化建议",
        "📋 数据总览"
    ])

    # ===== Tab 1: 特征重要性分析 =====
    with tab1:
        st.markdown("## 📈 特征重要性分析")
        st.markdown("展示各输入特征对三个目标变量的影响程度")


        # 训练模型并获取特征重要性
        def get_feature_importance(models_dict, y_col, model_type='xgb'):
            if model_type == 'xgb':
                return models_dict[y_col]['xgb'].feature_importances_
            elif model_type == 'rf':
                return models_dict[y_col]['rf'].feature_importances_
            else:
                return np.abs(models_dict[y_col]['lasso'].coef_)


        # 选择模型类型
        model_type_importance = st.radio(
            "选择模型类型",
            ['XGBoost', '随机森林', 'Lasso'],
            horizontal=True
        )

        model_map = {'XGBoost': 'xgb', '随机森林': 'rf', 'Lasso': 'lasso'}
        model_key = model_map[model_type_importance]

        # 创建三个子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{model_type_importance} 特征重要性分析', fontsize=16, fontweight='bold')

        for idx, y_col in enumerate(available_y):
            if model_key == 'lasso':
                importance = get_feature_importance(models, y_col, 'lasso')
            else:
                importance = get_feature_importance(models, y_col, model_key)

            # 排序
            sorted_idx = np.argsort(importance)[::-1]
            sorted_names = [available_X[i] for i in sorted_idx]
            sorted_values = importance[sorted_idx]
            sorted_cn_names = [x_names_cn.get(name, name) for name in sorted_names]

            ax = axes[idx]
            bars = ax.barh(sorted_cn_names, sorted_values, color='steelblue')
            ax.set_xlabel('特征重要性', fontsize=11)
            ax.set_title(f'{y_names_cn.get(y_col, y_col)}', fontsize=12)
            ax.invert_yaxis()

            for bar, val in zip(bars, sorted_values):
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', fontsize=9)
            ax.set_xlim(0, max(sorted_values) * 1.15)

        plt.tight_layout()
        st.pyplot(fig)

        # 额外：单独的大图
        st.markdown("---")
        st.markdown("### 详细视图 - 点击展开")

        selected_y = st.selectbox("选择目标变量", available_y, format_func=lambda x: y_names_cn.get(x, x))

        if selected_y:
            fig2, ax = plt.subplots(figsize=(10, 6))

            if model_key == 'lasso':
                importance = get_feature_importance(models, selected_y, 'lasso')
            else:
                importance = get_feature_importance(models, selected_y, model_key)

            sorted_idx = np.argsort(importance)[::-1]
            sorted_names = [available_X[i] for i in sorted_idx]
            sorted_values = importance[sorted_idx]
            sorted_cn_names = [x_names_cn.get(name, name) for name in sorted_names]

            bars = ax.barh(sorted_cn_names, sorted_values, color='steelblue')
            ax.set_xlabel('特征重要性', fontsize=12)
            ax.set_title(f'{model_type_importance} - {y_names_cn.get(selected_y, selected_y)} 特征重要性', fontsize=14)
            ax.invert_yaxis()

            for bar, val in zip(bars, sorted_values):
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', fontsize=10)
            ax.set_xlim(0, max(sorted_values) * 1.15)

            plt.tight_layout()
            st.pyplot(fig2)

    # ===== Tab 2: 模型对比 =====
    with tab2:
        st.markdown("## 📊 模型性能对比")

        # 显示评估结果
        col1, col2 = st.columns(2)

        for idx, y_col in enumerate(available_y):
            with col1 if idx % 2 == 0 else col2:
                st.markdown(f"### {y_names_cn.get(y_col, y_col)}")
                data = []
                for model_name, metrics in results[y_col].items():
                    model_display = {'lr': '线性回归', 'lasso': 'Lasso', 'rf': '随机森林', 'xgb': 'XGBoost'}[model_name]
                    data.append({
                        '模型': model_display,
                        'R²': f"{metrics['r2']:.4f}",
                        'RMSE': f"{metrics['rmse']:.4f}"
                    })
                st.table(pd.DataFrame(data))

        # 模型对比图
        st.markdown("---")
        st.markdown("### 模型性能可视化")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, y_col in enumerate(available_y):
            ax = axes[idx]
            model_names = ['线性回归', 'Lasso', '随机森林', 'XGBoost']
            r2_values = [results[y_col]['lr']['r2'], results[y_col]['lasso']['r2'],
                         results[y_col]['rf']['r2'], results[y_col]['xgb']['r2']]

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            bars = ax.bar(model_names, r2_values, color=colors)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('R² Score', fontsize=11)
            ax.set_title(f'{y_names_cn.get(y_col, y_col)}', fontsize=12)
            ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='良好阈值')

            for bar, val in zip(bars, r2_values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            ax.legend()

        plt.tight_layout()
        st.pyplot(fig)

        # ===== 如果勾选了热力图 =====
        if show_heatmap:
            st.markdown("---")
            st.markdown("### 🔥 特征相关性热力图")

            # 计算相关性
            corr_data = pd.concat([X_data, y_data], axis=1)
            corr_matrix = corr_data.corr()

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                        fmt='.2f', square=True, linewidths=0.5, ax=ax)
            ax.set_title('所有变量相关性热力图', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

    # ===== Tab 3: 预测与评估 =====
    with tab3:
        st.markdown("## 🎯 预测与评估")
        st.markdown("基于输入的参数，预测三个目标变量的值")

        if st.button("🚀 执行预测", type="primary"):
            col1, col2, col3 = st.columns(3)

            for idx, y_col in enumerate(available_y):
                model = models[y_col]['xgb']  # 使用XGBoost
                pred_val = predict_value(input_values, model, scaler)

                # 获取实际值范围
                actual_min = y_data[y_col].min()
                actual_max = y_data[y_col].max()
                actual_mean = y_data[y_col].mean()

                # 判断状态
                if pred_val > actual_max:
                    status = "🔴 偏高"
                    status_color = "danger-box"
                elif pred_val < actual_min:
                    status = "🔵 偏低"
                    status_color = "warning-box"
                else:
                    status = "🟢 正常"
                    status_color = "success-box"

                with [col1, col2, col3][idx]:
                    st.markdown(f"### {y_names_cn.get(y_col, y_col)}")
                    st.markdown(f"""
                    <div class="{status_color}" style="padding:1rem;border-radius:10px;">
                        <h3 style="margin:0;">{pred_val:.2f}</h3>
                        <p style="margin:0;">状态: {status}</p>
                        <p style="margin:0;font-size:0.9rem;">正常范围: {actual_min:.2f} ~ {actual_max:.2f}</p>
                        <p style="margin:0;font-size:0.9rem;">平均值: {actual_mean:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # ===== 散点图：显示预测点在真实数据中的位置 =====
            st.markdown("---")
            st.markdown("### 📍 预测值在真实数据中的位置")

            # 用户选择查看哪个目标变量的散点图
            selected_scatter = st.selectbox(
                "选择目标变量查看散点图",
                available_y,
                format_func=lambda x: y_names_cn.get(x, x)
            )

            if selected_scatter:
                # 创建散点图
                fig = go.Figure()

                # 原始数据点
                fig.add_trace(go.Scatter(
                    x=y_data[selected_scatter],
                    y=[0] * len(y_data),
                    mode='markers',
                    name='原始数据',
                    marker=dict(size=10, color='blue', opacity=0.6),
                    hovertemplate='<b>值</b>: %{x:.2f}<br><b>索引</b>: %{text}<extra></extra>',
                    text=y_data.index
                ))

                # 预测点
                pred_val = predict_value(input_values, models[selected_scatter]['xgb'], scaler)
                fig.add_trace(go.Scatter(
                    x=[pred_val],
                    y=[0],
                    mode='markers',
                    name='预测值',
                    marker=dict(size=18, color='red', symbol='star')
                ))

                fig.update_layout(
                    title=f'{y_names_cn.get(selected_scatter, selected_scatter)} 预测值分布',
                    xaxis_title='值',
                    yaxis_title='',
                    yaxis=dict(showticklabels=False),
                    height=300,
                    hovermode='closest'
                )

                st.plotly_chart(fig, use_container_width=True)

                # 显示统计数据
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("预测值", f"{pred_val:.2f}")
                with col2:
                    st.metric("最小值", f"{y_data[selected_scatter].min():.2f}")
                with col3:
                    st.metric("最大值", f"{y_data[selected_scatter].max():.2f}")

    # ===== Tab 4: 优化建议 =====
    with tab4:
        st.markdown("## 🔧 优化建议")
        st.markdown("根据当前污泥龄和F/M值，提供优化方案")

        # 获取当前预测值
        if st.button("🔄 生成优化方案", key="optimize"):
            pred_srt = predict_value(input_values, models['SRT']['xgb'], scaler)
            pred_fm = predict_value(input_values, models['F/M(%)']['xgb'], scaler)
            pred_svi = predict_value(input_values, models['SVI']['xgb'], scaler)

            # 获取正常范围
            srt_min, srt_max = y_data['SRT'].min(), y_data['SRT'].max()
            fm_min, fm_max = y_data['F/M(%)'].min(), y_data['F/M(%)'].max()
            svi_min, svi_max = y_data['SVI'].min(), y_data['SVI'].max()

            st.markdown("### 📊 当前状态评估")

            col1, col2, col3 = st.columns(3)


            def get_status(val, min_val, max_val):
                if val < min_val:
                    return "偏低 ⬇️", "warning-box"
                elif val > max_val:
                    return "偏高 ⬆️", "danger-box"
                else:
                    return "正常 ✅", "success-box"


            status_srt, color_srt = get_status(pred_srt, srt_min, srt_max)
            status_fm, color_fm = get_status(pred_fm, fm_min, fm_max)
            status_svi, color_svi = get_status(pred_svi, svi_min, svi_max)

            with col1:
                st.markdown(f"""
                <div class="{color_srt}" style="padding:1rem;border-radius:10px;">
                    <h4>SRT (污泥龄)</h4>
                    <h2>{pred_srt:.2f}</h2>
                    <p>状态: {status_srt}</p>
                    <p style="font-size:0.8rem;">正常范围: {srt_min:.2f} ~ {srt_max:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="{color_fm}" style="padding:1rem;border-radius:10px;">
                    <h4>有机质占比 (F/M)</h4>
                    <h2>{pred_fm:.2f}%</h2>
                    <p>状态: {status_fm}</p>
                    <p style="font-size:0.8rem;">正常范围: {fm_min:.2f} ~ {fm_max:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="{color_svi}" style="padding:1rem;border-radius:10px;">
                    <h4>SVI (污泥体积指数)</h4>
                    <h2>{pred_svi:.2f}</h2>
                    <p>状态: {status_svi}</p>
                    <p style="font-size:0.8rem;">正常范围: {svi_min:.2f} ~ {svi_max:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            # ===== 优化方案 =====
            st.markdown("---")
            st.markdown("### 💡 优化方案")

            # 分析问题并给出建议
            issues = []
            if pred_srt < srt_min:
                issues.append({
                    'param': 'SRT (污泥龄)',
                    'issue': '偏低',
                    'suggestion': '建议增加污泥回流量，延长污泥在系统内的停留时间'
                })
            elif pred_srt > srt_max:
                issues.append({
                    'param': 'SRT (污泥龄)',
                    'issue': '偏高',
                    'suggestion': '建议减少污泥回流量，适当排泥'
                })

            if pred_fm < fm_min:
                issues.append({
                    'param': '有机质占比 (F/M)',
                    'issue': '偏低',
                    'suggestion': '建议增加进水量或减少MLSS浓度，提高有机负荷'
                })
            elif pred_fm > fm_max:
                issues.append({
                    'param': '有机质占比 (F/M)',
                    'issue': '偏高',
                    'suggestion': '建议减少进水量或增加MLSS浓度，降低有机负荷'
                })

            if pred_svi < svi_min:
                issues.append({
                    'param': 'SVI (污泥体积指数)',
                    'issue': '偏低',
                    'suggestion': '污泥沉降性能良好，维持当前运行参数'
                })
            elif pred_svi > svi_max:
                issues.append({
                    'param': 'SVI (污泥体积指数)',
                    'issue': '偏高',
                    'suggestion': '存在污泥膨胀风险，建议增加曝气量或调整营养比'
                })

            if issues:
                for issue in issues:
                    st.markdown(f"""
                    <div class="warning-box" style="margin-bottom:0.5rem;">
                        <b>⚠️ {issue['param']}: {issue['issue']}</b><br>
                        📌 {issue['suggestion']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    ✅ 所有参数均在正常范围内，系统运行良好！
                    <br>📌 建议：维持当前运行参数，定期监测水质指标。
                </div>
                """, unsafe_allow_html=True)

            # ===== 污泥龄与F/M/SVI的关系 =====
            st.markdown("---")
            st.markdown("### 🔄 污泥龄与F/M、SVI的关系")

            # 模拟不同污泥龄下的F/M和SVI变化
            srt_range = np.linspace(srt_min * 0.5, srt_max * 1.5, 30)
            fm_values = []
            svi_values = []

            # 使用训练好的模型预测不同SRT下的值
            base_input = input_values.copy()
            for srt_val in srt_range:
                # 这里简化处理：假设SRT与F/M和SVI有负相关关系
                # 实际应用中可以用模型预测
                fm_values.append(pred_fm * (1 - 0.1 * (srt_val - pred_srt) / pred_srt))
                svi_values.append(pred_svi * (1 - 0.08 * (srt_val - pred_srt) / pred_srt))

            fig, ax1 = plt.subplots(figsize=(10, 6))

            color1 = '#1f77b4'
            ax1.set_xlabel('SRT (污泥龄)', fontsize=12)
            ax1.set_ylabel('F/M (%)', color=color1, fontsize=12)
            ax1.plot(srt_range, fm_values, color=color1, linewidth=2, label='F/M')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.axvline(x=pred_srt, color='red', linestyle='--', alpha=0.7, label='当前SRT')

            ax2 = ax1.twinx()
            color2 = '#d62728'
            ax2.set_ylabel('SVI', color=color2, fontsize=12)
            ax2.plot(srt_range, svi_values, color=color2, linewidth=2, label='SVI')
            ax2.tick_params(axis='y', labelcolor=color2)

            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

            plt.title('SRT对F/M和SVI的影响关系', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

            # 优化建议总结
            st.markdown("---")
            st.markdown("### 📋 优化总结")

            # 计算目标调整
            target_srt = pred_srt
            if pred_fm > fm_max:
                target_srt = pred_srt * 1.2  # 增加SRT降低F/M
            elif pred_fm < fm_min:
                target_srt = pred_srt * 0.8  # 减少SRT增加F/M

            st.markdown(f"""
            <div class="result-box">
                <h4>🎯 建议调整目标</h4>
                <ul>
                    <li><b>当前SRT</b>: {pred_srt:.2f} → <b>目标SRT</b>: {target_srt:.2f}</li>
                    <li><b>调整建议</b>: {'增加污泥回流量' if target_srt > pred_srt else '减少污泥回流量'}</li>
                    <li><b>预期效果</b>: 将F/M调整至正常范围，改善污泥沉降性能</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ===== Tab 5: 数据总览 =====
    with tab5:
        st.markdown("## 📋 数据总览")

        st.markdown("### 原始数据")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("---")
        st.markdown("### 数据统计")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 自变量统计")
            st.dataframe(X_data.describe())
        with col2:
            st.markdown("#### 因变量统计")
            st.dataframe(y_data.describe())

        st.markdown("---")
        st.markdown("### 📊 数据分布")

        # 选择变量查看分布
        selected_col = st.selectbox("选择变量查看分布", df.columns.tolist())
        if selected_col:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(df[selected_col].dropna(), bins=30, color='steelblue', edgecolor='white')
            ax.set_title(f'{selected_col} 分布', fontsize=14)
            ax.set_xlabel(selected_col)
            ax.set_ylabel('频数')
            st.pyplot(fig)

else:
    st.error("无法加载数据，请检查文件路径。")

st.markdown("---")
st.markdown("💧 **污水处理智能分析平台 v2.0** | 基于机器学习的多模型预测系统")