import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# 全局页面配置
st.set_page_config(page_title="污泥指标AI预测分析平台", layout="wide")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 初始化会话缓存
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train_dict" not in st.session_state:
    st.session_state.y_train_dict = {}
if "y_test_dict" not in st.session_state:
    st.session_state.y_test_dict = {}
if "model_dict" not in st.session_state:
    st.session_state.model_dict = {}
if "single_input" not in st.session_state:
    st.session_state.single_input = None

# ---------------------- 页面标题 ----------------------
st.title("🧪 进水水质-污泥指标机器学习交互预测平台")
st.subheader("流程：数据输入 → 模型训练验证 → 在线单样本预测 → 多模型可视化分析")
st.divider()

# ---------------------- Step1 数据输入模块 ----------------------
st.header("Step 1 数据录入与单样本参数输入")
col_upload, col_input = st.columns([1, 1])

# 1.1 上传Excel数据集（云端适配修改）
with col_upload:
    st.subheader("1.1 上传 data.xlsx 数据集")
    upload_file = st.file_uploader("上传数据集文件（data.xlsx）", type=["xlsx"])
    if upload_file is not None:
        df = pd.read_excel(upload_file)
        # 筛选原始真实字段（剔除归一化列）
        use_cols = [
            "Qoutm3/d", "BOD5 (mg/l)", "CODcr(mg/l)", "SS(mg/l)",
            "TP(mg/l)", "TN(mg/l)", "Tin℃",
            "SRT", "MLVSS/MLSS", "SVI"
        ]
        df_clean = df[use_cols].copy()
        df_clean = df_clean.dropna()
        st.session_state.df_raw = df_clean
        st.success("数据集加载完成！")
        st.dataframe(df_clean.head(8), height=220)

# 1.2 手动输入单条水质参数
with col_input:
    st.subheader("1.2 手动输入待预测水质参数")
    Qout = st.number_input("出水量 Qout(m³/d)", value=350000.0, min_value=200000.0, max_value=450000.0)
    BOD5 = st.number_input("BOD5 (mg/L)", value=140.0, min_value=70.0, max_value=320.0)
    COD = st.number_input("CODcr (mg/L)", value=260.0, min_value=160.0, max_value=750.0)
    SS = st.number_input("SS (mg/L)", value=130.0, min_value=50.0, max_value=430.0)
    TP = st.number_input("TP (mg/L)", value=4.0, min_value=1.5, max_value=20.0)
    TN = st.number_input("TN (mg/L)", value=36.0, min_value=27.0, max_value=80.0)
    Tin = st.number_input("进水温度 Tin(℃)", value=24.0, min_value=14.0, max_value=27.0)
    input_arr = np.array([[Qout, BOD5, COD, SS, TP, TN, Tin]])
    st.session_state.single_input = input_arr

# 字段命名映射
feature_names = ["出水量", "BOD5", "CODcr", "SS", "TP", "TN", "进水温度"]
feature_cols = ["Qoutm3/d", "BOD5 (mg/l)", "CODcr(mg/l)", "SS(mg/l)", "TP(mg/l)", "TN(mg/l)", "Tin℃"]
target_names = ["SRT污泥龄", "有机质占比MLVSS/MLSS", "SVI污泥指数"]
target_cols = ["SRT", "MLVSS/MLSS", "SVI"]

st.divider()

# ---------------------- Step2 模型训练与验证 ----------------------
st.header("Step 2 模型一键训练与精度验证")
if st.session_state.df_raw is None:
    st.warning("请先在Step1上传data.xlsx数据集，才能训练模型！")
else:
    df = st.session_state.df_raw
    X = df[feature_cols]
    Y = df[target_cols]
    train_btn = st.button("🚀 一键划分训练集/测试集 + 训练全部模型")
    if train_btn:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        # 保存至缓存
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train_dict = {t: y_train[t] for t in target_cols}
        st.session_state.y_test_dict = {t: y_test[t] for t in target_cols}

        # 批量训练3类模型，分别对3个因变量建模
        model_storage = {}
        model_list = ["线性回归", "Lasso", "随机森林", "XGBoost"]
        for m_name in model_list:
            target_models = {}
            for t_col in target_cols:
                y_tr = st.session_state.y_train_dict[t_col]
                if m_name == "线性回归":
                    md = LinearRegression()
                elif m_name == "Lasso":
                    md = Lasso(alpha=0.02, random_state=42)
                elif m_name == "随机森林":
                    md = RandomForestRegressor(n_estimators=100, random_state=42)
                else:
                    md = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
                md.fit(X_train, y_tr)
                target_models[t_col] = md
            model_storage[m_name] = target_models
        st.session_state.model_dict = model_storage
        st.success("全部模型训练完成！")

        # 输出模型精度指标表格
        metric_table = []
        for m_name, t_md in model_storage.items():
            row = {"模型": m_name}
            for t_col in target_cols:
                y_pred = t_md[t_col].predict(X_test)
                r2 = round(r2_score(st.session_state.y_test_dict[t_col], y_pred), 3)
                rmse = round(np.sqrt(mean_squared_error(st.session_state.y_test_dict[t_col], y_pred)), 3)
                row[f"{t_col}_R²"] = r2
                row[f"{t_col}_RMSE"] = rmse
            metric_table.append(row)
        st.dataframe(pd.DataFrame(metric_table), use_container_width=True)

st.divider()

# ---------------------- Step3 在线预测 + 预警 + 污泥龄优化 ----------------------
st.header("Step 3 单样本预测、指标预警与运行优化方案")
if len(st.session_state.model_dict) == 0:
    st.warning("请先完成Step2模型训练！")
else:
    select_pred_model = st.radio("选择用于预测的模型", ["线性回归", "Lasso", "随机森林", "XGBoost"])
    run_pred = st.button("执行预测并生成优化方案")
    if run_pred:
        pred_model = st.session_state.model_dict[select_pred_model]
        input_x = st.session_state.single_input
        pred_res = {}
        # 预测三个指标
        for t in target_cols:
            pred_res[t] = pred_model[t].predict(input_x)[0]
        srt_pred = pred_res["SRT"]
        org_pred = pred_res["MLVSS/MLSS"]
        svi_pred = pred_res["SVI"]

        # 展示预测数值
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("预测SRT污泥龄(天)", f"{srt_pred:.2f}")
        with c2:
            st.metric("预测有机质占比", f"{org_pred:.3f}")
        with c3:
            st.metric("预测SVI污泥指数", f"{svi_pred:.1f}")

        # 阈值预警（污水厂常规合理区间）
        org_low, org_high = 0.35, 0.65
        svi_low, svi_high = 70, 150
        warn_text = []
        if org_pred < org_low:
            warn_text.append(f"⚠️ 有机质占比偏低({org_pred:.3f})，污泥活性不足")
        elif org_pred > org_high:
            warn_text.append(f"⚠️ 有机质占比偏高({org_pred:.3f})，F/M负荷过大易膨胀")
        if svi_pred < svi_low:
            warn_text.append(f"⚠️ SVI偏低({svi_pred:.1f})，污泥老化细碎")
        elif svi_pred > svi_high:
            warn_text.append(f"⚠️ SVI偏高({svi_pred:.1f})，存在污泥膨胀风险")
        if len(warn_text) > 0:
            st.error("\n".join(warn_text))
        else:
            st.success("✅ 有机质占比、SVI均处于正常运行区间")

        # 污泥龄优化逻辑：指标越大，推荐SRT越大；越小则越小
        base_srt = srt_pred
        factor = ((org_pred / 0.5) + (svi_pred / 110)) / 2
        opt_srt = round(base_srt * factor, 2)
        st.subheader("📋 污泥运行优化方案")
        st.metric("推荐最优污泥龄SRT(天)", opt_srt)
        if factor > 1.05:
            st.info(f"有机质/SVI偏高，建议**延长污泥龄至{opt_srt}天**，增加污泥总量，降低F/M负荷")
        elif factor < 0.95:
            st.info(f"有机质/SVI偏低，建议**缩短污泥龄至{opt_srt}天**，减少污泥总量，提升污泥活性")
        else:
            st.info("当前工况均衡，污泥龄无需大幅调整")

st.divider()

# ---------------------- Step4 可视化绘图控制面板 ----------------------
st.header("Step 4 多模型可视化分析面板")
if len(st.session_state.model_dict) == 0:
    st.warning("请先完成Step2模型训练，才能生成图表！")
else:
    # 绘图勾选控件
    st.subheader("图表选择（线性回归散点图为强制标配）")
    col_check, col_btn = st.columns([3, 1])
    with col_check:
        show_heat = st.checkbox("显示相关性热力图")
        show_lasso = st.checkbox("显示Lasso系数图")
        show_rf = st.checkbox("显示随机森林特征重要度")
        show_xgb = st.checkbox("显示XGBoost特征重要度")
    with col_btn:
        if st.button("一键勾选全部图表"):
            show_heat = True
            show_lasso = True
            show_rf = True
            show_xgb = True

    # 选择绘图对应的因变量
    select_target = st.radio("选择要可视化的目标污泥指标", target_cols, horizontal=True)
    X_test = st.session_state.X_test
    y_test_t = st.session_state.y_test_dict[select_target]
    input_x = st.session_state.single_input
    df_raw = st.session_state.df_raw

    # ========== 1. 强制标配：线性回归真实-预测散点图 ==========
    st.subheader("【标配】线性回归真实值vs预测值散点图")
    lr_model = st.session_state.model_dict["线性回归"][select_target]
    y_pred_test = lr_model.predict(X_test)
    y_pred_single = lr_model.predict(input_x)[0]
    fig_scatter = px.scatter(
        x=y_test_t.values,
        y=y_pred_test,
        labels={"x": f"真实{select_target}", "y": f"线性回归预测{select_target}"},
        title=f"线性回归拟合效果：{select_target}（蓝=历史样本，红=当前输入工况）",
        hover_data={"真实值": y_test_t.values, "预测值": y_pred_test}
    )
    min_v = min(y_test_t.min(), y_pred_test.min())
    max_v = max(y_test_t.max(), y_pred_test.max())
    fig_scatter.add_scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", line_dash="dash", name="理想拟合线")
    fig_scatter.add_scatter(
        x=[y_pred_single], y=[y_pred_single],
        marker_color="red", marker_size=12, name="当前输入工况",
        hover_data={"输入预测值": y_pred_single}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ========== 2. 可选：相关性热力图 ==========
    if show_heat:
        st.subheader("相关性热力图（含显著性标记）")
        corr_df = df_raw[feature_cols + target_cols].copy()
        corr_mat = corr_df.corr()
        p_mat = np.zeros_like(corr_mat)
        for i in range(len(corr_mat.columns)):
            for j in range(len(corr_mat.columns)):
                _, p = pearsonr(corr_df.iloc[:, i], df_raw.iloc[:, j])
                p_mat[i, j] = p
        def sig_star(p):
            if p < 0.001: return "***"
            elif p < 0.01: return "**"
            elif p < 0.05: return "*"
            else: return ""
        annot_text = np.vectorize(lambda r, p: f"{r:.2f}{sig_star(p)}")(corr_mat.values, p_mat)
        fig_heat, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(corr_mat, annot=annot_text, fmt="", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
        ax.set_title("变量相关性热力图 ***p<0.001,**p<0.01,*p<0.05", fontsize=14)
        st.pyplot(fig_heat)

    # ========== 3. 可选：Lasso回归系数图 ==========
    if show_lasso:
        st.subheader(f"Lasso回归特征系数（{select_target}）")
        lasso_md = st.session_state.model_dict["Lasso"][select_target]
        coefs = lasso_md.coef_
        fig_lasso, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(feature_names, coefs, color=np.where(coefs>0, "#d62728", "#1f77b4"))
        ax.set_xlabel("Lasso标准化系数（正值正向影响，负值负向影响）")
        ax.set_title(f"Lasso各进水参数对{select_target}的驱动系数")
        st.pyplot(fig_lasso)

    # ========== 4. 可选：随机森林特征重要度 ==========
    if show_rf:
        st.subheader(f"随机森林特征重要度（{select_target}）")
        rf_md = st.session_state.model_dict["随机森林"][select_target]
        imp_rf = rf_md.feature_importances_
        fig_rf, ax = plt.subplots(figsize=(10, 5))
        ax.barh(feature_names, imp_rf, color="#2ca02c")
        ax.set_xlabel("特征重要度")
        ax.set_title(f"7项进水参数对{select_target}贡献大小排序")
        st.pyplot(fig_rf)

    # ========== 5. 可选：XGBoost特征重要度 ==========
    if show_xgb:
        st.subheader(f"XGBoost特征重要度（{select_target}）")
        xgb_md = st.session_state.model_dict["XGBoost"][select_target]
        imp_xgb = xgb_md.feature_importances_
        fig_xgb, ax = plt.subplots(figsize=(10, 5))
        ax.barh(feature_names, imp_xgb, color="#ff7f0e")
        ax.set_xlabel("特征重要度")
        ax.set_title(f"XGBoost各进水参数对{select_target}影响权重")
        st.pyplot(fig_xgb)