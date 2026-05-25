import warnings
# 過濾掉 Sklearn 的欄位名稱警告，保持後台 LOG 清潔
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sklearn
from scipy.optimize import minimize

st.set_page_config(page_title="玻璃預測與設計系統", layout="centered")

st.title("🔬 玻璃物理性質 AI 預測與設計系統")

# 狀態檢查清單
with st.sidebar:
    st.header("系統狀態檢查")
    files_to_check = ["optimized_glass_model.joblib", "sciglass_database.csv", "interglad_database.csv"]
    env_ok = True
    for f in files_to_check:
        if os.path.exists(f):
            st.success(f"✅ {f} 已就緒")
        else:
            st.error(f"❌ 缺少 {f}")
            env_ok = False
    st.info(f"sklearn version: {sklearn.__version__}")

if not env_ok:
    st.warning("請確保所有必要檔案都已上傳到 GitHub 根目錄。")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load("optimized_glass_model.joblib")

with st.spinner('正在載入 AI 模型，請稍候...'):
    try:
        model_pkg = get_model()
        st.sidebar.success("🔥 模型載入成功")
    except Exception as e:
        st.error(f"模型載入失敗: {e}")
        st.stop()

# 正向預測邏輯 (純 NumPy 高速版)
def predict(comp, pkg):
    f_cols = pkg["input_features"]
    feat_to_idx = {col: i for i, col in enumerate(f_cols)}
    arr = np.zeros(len(f_cols))
    
    for k, v in comp.items():
        col = k if k.endswith("_mass_pct") else k + "_mass_pct"
        if col in feat_to_idx:
            arr[feat_to_idx[col]] = float(v)
            
    total = np.sum(arr)
    if total > 0:
        arr = (arr / total) * 100
        
    X_arr = arr.reshape(1, -1)
    preds = {}
    for t in pkg["models"]:
        preds[t] = pkg["models"][t].predict(X_arr)[0]
        
    if preds.get("cte_1e-6_per_C", 10) < 3.5:
        corr = 0.15 + (0.03 * comp.get("B2O3", 0))
        preds["cte_1e-6_per_C"] = max(preds["cte_1e-6_per_C"] - corr, 0.1)
    return preds

# --- 核心升級：稀疏正規化 (Sparsity Penalty) + 誠實驗收機制 ---
def inverse_predict(target_cte, target_young, target_viscosity, allowed_oxides, custom_bounds, pkg):
    def loss_fn(weights, oxides_list):
        comp = {ox: w for ox, w in zip(oxides_list, weights)}
        preds = predict(comp, pkg)
        loss = 0.0
        
        # 1. 物理性質相對誤差平方和
        if target_cte > 0:
            loss += ((preds.get("cte_1e-6_per_C", 0) - target_cte) / target_cte) ** 2
        if target_young > 0:
            loss += ((preds.get("young_modulus_GPa", 0) - target_young) / target_young) ** 2
        if target_viscosity > 0:
            loss += ((preds.get("T_at_1E3_dPas_C", 0) - target_viscosity) / target_viscosity) ** 2
            
        # 2. 加入稀疏懲罰 (L0.5 Penalty)：自然壓抑微量元素的產生
        # 讓演算法知道「用一堆 0.1% 的元素」會被扣分，強迫它收斂到少數主要元素
        sparsity_penalty = sum((w / 100.0) ** 0.5 for w in weights)
        loss += 0.01 * sparsity_penalty 
        
        return loss

    bounds = [custom_bounds.get(ox, (0.0, 100.0)) for ox in allowed_oxides]
    constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 100.0}

    # 多起點探索：提供合乎常理的初始值，避免算太久或卡死
    seeds = []
    mid_seed = [(b[0] + b[1]) / 2.0 for b in bounds]
    s_mid = sum(mid_seed)
    seeds.append([w / s_mid * 100 for w in mid_seed] if s_mid > 0 else [100.0/len(bounds)]*len(bounds))
    
    if "SiO2" in allowed_oxides:
        idx = allowed_oxides.index("SiO2")
        s2 = [custom_bounds[ox][0] for ox in allowed_oxides]
        s2[idx] = max(65.0, custom_bounds["SiO2"][0])
        tot = sum(s2)
        seeds.append([w / tot * 100 for w in s2] if tot > 0 else seeds[0])

    # 執行優化尋找最佳解
    best_loss = float('inf')
    best_weights = seeds[0]

    for seed_init in seeds:
        res = minimize(
            loss_fn, seed_init, method='SLSQP', bounds=bounds, constraints=constraints,
            args=(allowed_oxides,), tol=1e-3, options={'maxiter': 50, 'ftol': 1e-3}
        )
        if res.fun < best_loss:
            best_loss = res.fun
            best_weights = res.x

    # 封裝配方
    opt_comp = {}
    s_f = sum(best_weights)
    if s_f > 0:
        best_weights = [w / s_f * 100 for w in best_weights]
        
    for ox, w in zip(allowed_oxides, best_weights):
        # 由於稀疏懲罰發揮作用，極小值(例如 < 0.05%)會自然產生，這裡單純做四捨五入顯示過濾
        if round(w, 2) > 0:
            opt_comp[ox] = round(w, 2)

    # --- 誠實驗收機制 ---
    # 預測最終配方的實際數值
    final_preds = predict(opt_comp, pkg)
    is_success = True
    
    # 嚴格驗證：只要有任何一個指標誤差超過 10%，就判定為「未達標」
    if target_cte > 0 and abs(final_preds.get("cte_1e-6_per_C", target_cte) - target_cte) / target_cte > 0.10:
        is_success = False
    if target_young > 0 and abs(final_preds.get("young_modulus_GPa", target_young) - target_young) / target_young > 0.10:
        is_success = False
    if target_viscosity > 0 and abs(final_preds.get("T_at_1E3_dPas_C", target_viscosity) - target_viscosity) / target_viscosity > 0.10:
        is_success = False

    return opt_comp, is_success, final_preds

# 建立分頁標籤
tab1, tab2 = st.tabs(["🔮 正向性質預測", "🧩 逆向配方設計"])
all_oxides = sorted([c.replace("_mass_pct", "") for c in model_pkg["input_features"]])

# --- Tab 1: 正向預測 ---
with tab1:
    st.subheader("1. 輸入玻璃配方 (mass %)")
    selected = st.multiselect("選擇組分", all_oxides, default=["SiO2", "Al2O3", "B2O3", "CaO", "MgO", "Na2O"], key="forward_select")

    input_data = {}
    cols = st.columns(3)
    for i, ox in enumerate(selected):
        with cols[i % 3]:
            input_data[ox] = st.number_input(f"{ox}", value=0.0, step=0.1, key=f"f_{ox}")

    if st.button("🚀 開始預測", use_container_width=True):
        res = predict(input_data, model_pkg)
        st.divider()
        st.subheader("2. 預測結果")
        r1, r2, r3 = st.columns(3)
        r1.metric("CTE", f"{res['cte_1e-6_per_C']:.4f}")
        r2.metric("Young's Modulus", f"{res['young_modulus_GPa']:.2f} GPa")
        r3.metric("Viscosity (T10^3)", f"{res['T_at_1E3_dPas_C']:.1f} °C")

# --- Tab 2: 逆向設計 ---
with tab2:
    st.subheader("1. 設定目標物理性質")
    t_cols = st.columns(3)
    with t_cols[0]:
        target_cte = st.number_input("目標 CTE", value=8.5, step=0.1, min_value=0.0)
    with t_cols[1]:
        target_young = st.number_input("目標 Young's Modulus (GPa)", value=70.0, step=1.0, min_value=0.0)
    with t_cols[2]:
        target_viscosity = st.number_input("目標 Viscosity (T10^3) (°C)", value=1000.0, step=10.0, min_value=0.0)
        
    st.subheader("2. 決定逆向配方搜尋範圍")
    search_mode = st.radio(
        "請選擇評估範圍：",
        ["使用資料庫中【所有支援的氧化物】進行全面評估 (跳脫常見配方限制)", "僅在【自行指定的氧化物】中進行調配搜尋"],
        index=0
    )

    if search_mode == "使用資料庫中【所有支援的氧化物】進行全面評估 (跳脫常見配方限制)":
        inverse_selected = all_oxides
        st.info("💡 已啟用所有元素搜尋。演算法將透過數學模型自動尋找最佳化且組分精簡的配方。")
    else:
        inverse_selected = st.multiselect(
            "選擇允許演算法調配的指定組分", 
            all_oxides, 
            default=["SiO2", "Al2O3", "B2O3", "CaO", "MgO", "Na2O"],
            key="inverse_select_custom"
        )

    st.subheader("3. 新增特定的組分含量限制 (選填)")
    st.caption("您可以新增需要嚴格限制範圍的元素。未設定的組分，預設容許範圍為 0% ~ 100%。")
    
    limit_selected = st.multiselect(
        "點擊下方選擇要設定範圍限制的氧化物組分：",
        inverse_selected,
        help="選擇您需要特別指定上限或下限的組分"
    )

    active_bounds = {}
    has_bound_error = False
    
    if len(limit_selected) > 0:
        for ox in limit_selected:
            b_cols = st.columns([2, 3, 3])
            with b_cols[0]:
                st.markdown(f"<div style='padding-top: 25px;'><b>{ox}</b></div>", unsafe_allow_html=True)
            with b_cols[1]:
                min_v = st.number_input(f"{ox} 下限 (Min)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"min_{ox}")
            with b_cols[2]:
                max_v = st.number_input(f"{ox} 上限 (Max)", min_value=0.0, max_value=100.0, value=100.0, step=0.1, key=f"max_{ox}")
            
            if min_v > max_v:
                st.error(f"❌ 錯誤：{ox} 的下限不能大於上限！")
                has_bound_error = True
            active_bounds[ox] = (min_v, max_v)
            
    full_bounds = {}
    for ox in inverse_selected:
        if ox in active_bounds:
            full_bounds[ox] = active_bounds[ox]
        else:
            full_bounds[ox] = (0.0, 100.0)

    st.divider()
    if st.button("🧩 開始逆向尋找配方", use_container_width=True):
        if len(inverse_selected) == 0:
            st.error("請確保搜尋範圍內至少包含一種組分！")
        elif has_bound_error:
            st.error("請先修正上方組分上下限設定錯誤，再執行預測。")
        else:
            with st.spinner('AI 正在尋找符合物理性質的精簡配方...'):
                opt_comp, success, actual_res = inverse_predict(
                    target_cte, target_young, target_viscosity, 
                    inverse_selected, full_bounds, model_pkg
                )
                
                if success:
                    st.success("✨ AI 成功找到高度符合您目標的玻璃配方！")
                else:
                    st.warning("⚠️ 在當前範圍限制下，AI 未能找到完美達標的配方（誤差超過 10%）。以下提供最接近的妥協解：")
                
                # 顯示配方結果
                st.subheader("💡 模擬玻璃配方 (Mass %)")
                
                if len(opt_comp) == 0:
                    st.error("演算法無法生成有效配方，請放寬限制條件。")
                else:
                    df_comp = pd.DataFrame(list(opt_comp.items()), columns=["氧化物組分", "推薦比例 (mass %)"])
                    df_comp = df_comp.sort_values(by="推薦比例 (mass %)", ascending=False)
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)
                    
                    # 顯示此配方的預測性質與目標的差距
                    st.subheader("🎯 該配方之模擬物理性質與目標對比")
                    a1, a2, a3 = st.columns(3)
                    a1.metric("預測 CTE", f"{actual_res['cte_1e-6_per_C']:.4f}", delta=f"目標: {target_cte}", delta_color="off")
                    a2.metric("預測 Young's Modulus", f"{actual_res['young_modulus_GPa']:.2f} GPa", delta=f"目標: {target_young}", delta_color="off")
                    a3.metric("預測 Viscosity (T10^3)", f"{actual_res['T_at_1E3_dPas_C']:.1f} °C", delta=f"目標: {target_viscosity}", delta_color="off")