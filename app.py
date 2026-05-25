import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sklearn
from scipy.optimize import minimize  # 引入優化演算法模組

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

# 使用 st.spinner 處理耗時任務
with st.spinner('正在載入 AI 模型，請稍候...'):
    try:
        model_pkg = get_model()
        st.sidebar.success("🔥 模型載入成功")
    except Exception as e:
        st.error(f"模型載入失敗: {e}")
        st.stop()

# 正向預測邏輯
def predict(comp, pkg):
    f_cols = pkg["input_features"]
    x = pd.DataFrame([{c: 0.0 for c in f_cols}])
    for k, v in comp.items():
        col = k if k.endswith("_mass_pct") else k + "_mass_pct"
        if col in x.columns: x.loc[0, col] = float(v)
    # 歸一化
    total = x[f_cols].sum(axis=1).iloc[0]
    if total > 0: x[f_cols] = x[f_cols].div(total, axis=0) * 100
    
    preds = {t: pkg["models"][t].predict(x[f_cols])[0] for t in pkg["models"]}
    # CTE 偏差修正
    if preds.get("cte_1e-6_per_C", 10) < 3.5:
        corr = 0.15 + (0.03 * comp.get("B2O3", 0))
        preds["cte_1e-6_per_C"] = max(preds["cte_1e-6_per_C"] - corr, 0.1)
    return preds

# 逆向優化邏輯
def inverse_predict(target_cte, target_young, target_viscosity, allowed_oxides, custom_bounds, pkg):
    def loss_fn(weights):
        comp = {ox: w for ox, w in zip(allowed_oxides, weights)}
        preds = predict(comp, pkg)
        
        loss = 0.0
        # 使用相對誤差平方和
        if target_cte > 0:
            loss += ((preds.get("cte_1e-6_per_C", 0) - target_cte) / target_cte) ** 2
        if target_young > 0:
            loss += ((preds.get("young_modulus_GPa", 0) - target_young) / target_young) ** 2
        if target_viscosity > 0:
            loss += ((preds.get("T_at_1E3_dPas_C", 0) - target_viscosity) / target_viscosity) ** 2
        return loss

    n = len(allowed_oxides)
    if n == 0:
        return {}, False

    # 建立邊界與初始猜測
    init_guess = []
    bounds = []
    for ox in allowed_oxides:
        b_min, b_max = custom_bounds.get(ox, (0.0, 100.0))
        bounds.append((b_min, b_max))
        init_guess.append((b_min + b_max) / 2.0)
        
    # 限制所有成分總和必須等於 100%
    constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 100.0}
    
    # 執行 SLSQP 有限優化演算法
    res = minimize(
        loss_fn, 
        init_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        tol=1e-2,                  
        options={
            'maxiter': 100,        # 提高至 100 次以應對全組分（多變數）的搜尋空間
            'ftol': 1e-2
        }
    )
    
    best_weights = res.x
    total = sum(best_weights)
    if total > 0:
        best_weights = [w / total * 100 for w in best_weights]
        
    opt_comp = {ox: round(w, 2) for ox, w in zip(allowed_oxides, best_weights)}
    return opt_comp, res.success

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
        st.info(f"💡 系統已啟用模型支援的全部 {len(all_oxides)} 種氧化物組分進行高維度優化，這能幫您跳脫傳統思維，尋找更創新的潛在配方！")
    else:
        inverse_selected = st.multiselect(
            "選擇允許演算法調配的指定組分", 
            all_oxides, 
            default=["SiO2", "Al2O3", "B2O3", "CaO", "MgO", "Na2O"],
            key="inverse_select_custom"
        )

    # --- 新增功能：動態新增特定組分範圍限制 ---
    st.subheader("3. 新增特定的組分含量限制 (選填)")
    st.caption("您可以新增需要嚴格限制範圍的元素。未新增設定的組分，系統預設容許範圍為 0% ~ 100%。")
    
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
                st.markdown(f"<div style='padding-top: 25px;'><b>{ox}</b></div>", unsafe_html=True)
            with b_cols[1]:
                min_v = st.number_input(f"{ox} 下限 (Min)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"min_{ox}")
            with b_cols[2]:
                max_v = st.number_input(f"{ox} 上限 (Max)", min_value=0.0, max_value=100.0, value=100.0, step=0.1, key=f"max_{ox}")
            
            if min_v > max_v:
                st.error(f"❌ 錯誤：{ox} 的下限不能大於上限！")
                has_bound_error = True
            active_bounds[ox] = (min_v, max_v)
            
    # 將所有組分的邊界矩陣補齊（沒被選到的預設就是 0.0 ~ 100.0）
    full_bounds = {}
    for ox in inverse_selected:
        if ox in active_bounds:
            full_bounds[ox] = active_bounds[ox]
        else:
            full_bounds[ox] = (0.0, 100.0)

    # 可行性防呆檢查
    sum_min = sum([b[0] for b in full_bounds.values()])
    sum_max = sum([b[1] for b in full_bounds.values()])
    if sum_min > 100.0:
        st.warning(f"⚠️ 警告：目前設定的【下限總和】已達 {sum_min:.1f}% (超過 100%)，演算法可能找不到合理解。")
    if sum_max < 100.0:
        st.warning(f"⚠️ 警告：目前設定的【上限總和】僅 {sum_max:.1f}% (未達 100%)，配方總和將無法湊滿 100%。")

    # 觸發逆向優化按鈕
    st.divider()
    if st.button("🧩 開始逆向尋找配方", use_container_width=True):
        if len(inverse_selected) == 0:
            st.error("請確保搜尋範圍內至少包含一種組分！")
        elif has_bound_error:
            st.error("請先修正上方組分上下限設定錯誤，再執行預測。")
        else:
            with st.spinner('AI 正在全組分高維空間中進行反向調配優化（約需 3-5 秒）...'):
                opt_comp, success = inverse_predict(target_cte, target_young, target_viscosity, inverse_selected, full_bounds, model_pkg)
                
                st.success("✨ AI 已計算出最貼近目標與範圍限制的玻璃配方！")
                
                # 計算該優化配方實際對應的預測性質
                actual_res = predict(opt_comp, model_pkg)
                
                # 顯示推薦配方結果
                st.subheader("💡 AI 推薦玻璃配方 (Mass %)")
                
                # 過濾掉 0% 的成分，只顯示有含量的主要組分（大於 0.01%）
                filtered_comp = {k: v for k, v in opt_comp.items() if v > 0.01}
                if len(filtered_comp) == 0:
                    st.warning("在當前限制條件下未配置出有效配方，請放寬組分範圍。")
                else:
                    df_comp = pd.DataFrame(list(filtered_comp.items()), columns=["氧化物組分", "推薦比例 (mass %)"])
                    # 按照比例由高到低排序，看起來更專業
                    df_comp = df_comp.sort_values(by="推薦比例 (mass %)", ascending=False)
                    st.dataframe(df_comp, use_container_width=True, hide_index=True)
                    
                    # 顯示此配方的預測性質與目標的差距
                    st.subheader("🎯 該配方之模擬物理性質")
                    a1, a2, a3 = st.columns(3)
                    a1.metric("預測 CTE", f"{actual_res['cte_1e-6_per_C']:.4f}", delta=f"目標: {target_cte}")
                    a2.metric("預測 Young's Modulus", f"{actual_res['young_modulus_GPa']:.2f} GPa", delta=f"目標: {target_young}")
                    a3.metric("預測 Viscosity (T10^3)", f"{actual_res['T_at_1E3_dPas_C']:.1f} °C", delta=f"目標: {target_viscosity}")