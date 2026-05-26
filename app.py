import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sklearn
import random
from scipy.optimize import minimize

st.set_page_config(page_title="玻璃預測與設計系統", layout="centered")
st.title("🔬 玻璃物理性質 AI 預測與設計系統")

# --- 狀態檢查 ---
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

# --- 正向預測邏輯 ---
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
    preds = {t: pkg["models"][t].predict(X_arr)[0] for t in pkg["models"]}
        
    if preds.get("cte_1e-6_per_C", 10) < 3.5:
        corr = 0.15 + (0.03 * comp.get("B2O3", 0))
        preds["cte_1e-6_per_C"] = max(preds["cte_1e-6_per_C"] - corr, 0.1)
    return preds

# --- 逆向優化邏輯 (蒙地卡羅混合優化版) ---
def inverse_predict(target_cte, target_young, target_viscosity, allowed_oxides, custom_bounds, pkg):
    def loss_fn(weights, oxides_list):
        comp = {ox: w for ox, w in zip(oxides_list, weights)}
        preds = predict(comp, pkg)
        loss = 0.0
        
        # 誤差權重可以根據需求調整，這裡讓 CTE 的達成優先度稍微提高
        if target_cte > 0: loss += (((preds.get("cte_1e-6_per_C", 0) - target_cte) / target_cte) ** 2) * 2.0
        if target_young > 0: loss += ((preds.get("young_modulus_GPa", 0) - target_young) / target_young) ** 2
        if target_viscosity > 0: loss += ((preds.get("T_at_1E3_dPas_C", 0) - target_viscosity) / target_viscosity) ** 2
        return loss

    bounds = [custom_bounds.get(ox, (0.0, 100.0)) for ox in allowed_oxides]
    
    # 【核心修正 1】：蒙地卡羅隨機尋找最佳起點 (打破平坦梯度陷阱)
    best_mc_loss = float('inf')
    best_mc_guess = None
    
    for _ in range(300):
        guess = []
        for i, ox in enumerate(allowed_oxides):
            b_min, b_max = bounds[i]
            if ox == "SiO2": 
                guess.append(random.uniform(max(40, b_min), min(85, b_max)))
            elif ox in ["B2O3", "Al2O3"]: 
                guess.append(random.uniform(b_min, min(25, b_max)))
            else: 
                # 讓鹼金屬等助熔劑有 50% 機率直接趨近下限，有助於尋找低 CTE 配方
                if random.random() > 0.5:
                    guess.append(random.uniform(b_min, min(5, b_max)))
                else:
                    guess.append(b_min + random.uniform(0, 0.5))
                    
        # 歸一化並檢查邊界
        s = sum(guess)
        if s == 0: continue
        guess = [w / s * 100 for w in guess]
        
        valid = all(bounds[i][0] <= w <= bounds[i][1] for i, w in enumerate(guess))
        if valid:
            current_loss = loss_fn(guess, allowed_oxides)
            if current_loss < best_mc_loss:
                best_mc_loss = current_loss
                best_mc_guess = guess

    # 如果隨機撒網失敗，給一個預設的合法值
    if best_mc_guess is None:
        best_mc_guess = [(b[0] + b[1]) / 2.0 for b in bounds]
        s = sum(best_mc_guess)
        best_mc_guess = [w / s * 100 for w in best_mc_guess]

    # 【核心修正 2】：從最佳起點進行 SLSQP 微調，並加大步伐 (eps=0.5) 避免卡死
    constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 100.0}
    res_initial = minimize(
        loss_fn, best_mc_guess, method='SLSQP', bounds=bounds, constraints=constraints,
        args=(allowed_oxides,), tol=1e-3, options={'maxiter': 50, 'eps': 0.5}
    )
    
    # 階段 2：實用化修剪 (保留 > 0.01% 的元素，最多保留 Top 10)
    ox_weights = list(zip(allowed_oxides, res_initial.x))
    ox_weights = [x for x in ox_weights if x[1] >= 0.01 or custom_bounds.get(x[0], (0, 100))[0] > 0]
    ox_weights.sort(key=lambda x: x[1], reverse=True)
    ox_weights = ox_weights[:10]

    pruned_oxides = [x[0] for x in ox_weights]
    pruned_init = [x[1] for x in ox_weights]
    pruned_bounds = [custom_bounds.get(ox, (0.0, 100.0)) for ox in pruned_oxides]
    
    s_p = sum(pruned_init)
    pruned_init = [w / s_p * 100 for w in pruned_init] if s_p > 0 else [100.0]

    # 階段 3：對精簡後的組合進行最後精確逼近
    res_final = minimize(
        loss_fn, pruned_init, method='SLSQP', bounds=pruned_bounds, 
        constraints={'type': 'eq', 'fun': lambda w: sum(w) - 100.0},
        args=(pruned_oxides,), tol=1e-4, options={'maxiter': 40, 'eps': 0.5}
    )

    # 階段 4：實驗室級數據清理 (小數點 2 位，湊齊 100%)
    final_w = [round(w, 2) for w in res_final.x]
    s_f = sum(final_w)
    
    if final_w:
        max_idx = final_w.index(max(final_w))
        final_w[max_idx] = round(final_w[max_idx] + (100.0 - s_f), 2)

    opt_comp = {ox: w for ox, w in zip(pruned_oxides, final_w) if w > 0}
    final_preds = predict(opt_comp, pkg)
    
    return opt_comp, final_preds

# --- UI 介面 ---
tab1, tab2 = st.tabs(["🔮 正向性質預測", "🧩 逆向配方設計"])
all_oxides = sorted([c.replace("_mass_pct", "") for c in model_pkg["input_features"]])

with tab1:
    st.subheader("1. 輸入玻璃配方 (mass %)")
    selected = st.multiselect("選擇組分", all_oxides, default=["SiO2", "Al2O3", "B2O3", "CaO", "MgO", "Na2O"], key="f_sel")
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
        ["全元素自動篩選 (AI 將優先選擇常規玻璃成分，限制最多 10 種)", "自訂指定氧化物"],
        index=0
    )

    if search_mode == "全元素自動篩選 (AI 將優先選擇常規玻璃成分，限制最多 10 種)":
        inverse_selected = all_oxides
    else:
        inverse_selected = st.multiselect(
            "選擇允許演算法調配的指定組分", all_oxides, 
            default=["SiO2", "Al2O3", "B2O3", "CaO", "MgO", "Na2O"], key="inv_sel"
        )

    st.subheader("3. 新增特定的組分含量限制 (選填)")
    limit_selected = st.multiselect("選擇要設定上下限的組分：", inverse_selected)

    active_bounds = {}
    has_bound_error = False
    
    if len(limit_selected) > 0:
        for ox in limit_selected:
            b_cols = st.columns([2, 3, 3])
            with b_cols[0]:
                st.markdown(f"<div style='padding-top: 25px;'><b>{ox}</b></div>", unsafe_allow_html=True)
            with b_cols[1]:
                min_v = st.number_input(f"{ox} 下限", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"min_{ox}")
            with b_cols[2]:
                max_v = st.number_input(f"{ox} 上限", min_value=0.0, max_value=100.0, value=100.0, step=1.0, key=f"max_{ox}")
            if min_v > max_v:
                st.error(f"❌ 錯誤：{ox} 的下限不能大於上限！")
                has_bound_error = True
            active_bounds[ox] = (min_v, max_v)
            
    full_bounds = {ox: active_bounds.get(ox, (0.0, 100.0)) for ox in inverse_selected}

    st.divider()
    if st.button("🧩 生成實用配方", use_container_width=True):
        if len(inverse_selected) == 0:
            st.error("請至少包含一種組分！")
        elif has_bound_error:
            st.error("請修正上下限設定。")
        else:
            with st.spinner('AI 正在尋找配方 (啟動蒙地卡羅全域搜索以確保目標達成)...'):
                opt_comp, actual_res = inverse_predict(
                    target_cte, target_young, target_viscosity, 
                    inverse_selected, full_bounds, model_pkg
                )
                
                st.success("✨ 配方生成完畢！")
                
                st.subheader("💡 實驗室推薦配方 (Mass %)")
                df_comp = pd.DataFrame(list(opt_comp.items()), columns=["氧化物", "質量百分比 (%)"])
                df_comp = df_comp.sort_values(by="質量百分比 (%)", ascending=False)
                st.dataframe(df_comp, use_container_width=True, hide_index=True)
                
                st.subheader("🎯 此配方的預測性質")
                a1, a2, a3 = st.columns(3)
                a1.metric("預測 CTE", f"{actual_res['cte_1e-6_per_C']:.4f}", delta=f"目標: {target_cte}", delta_color="off")
                a2.metric("預測 Young's Modulus", f"{actual_res['young_modulus_GPa']:.1f} GPa", delta=f"目標: {target_young}", delta_color="off")
                a3.metric("預測 Viscosity (T10^3)", f"{actual_res['T_at_1E3_dPas_C']:.0f} °C", delta=f"目標: {target_viscosity}", delta_color="off")