
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title='玻璃物性預測系統', layout='centered')

@st.cache_resource
def load_model():
    return joblib.load('optimized_glass_model.joblib')

def predict_with_correction(composition_dict, model_package):
    f_cols = model_package['input_features']
    x = pd.DataFrame([{c: 0.0 for c in f_cols}])
    for k, v in composition_dict.items():
        col = k if k.endswith('_mass_pct') else k + '_mass_pct'
        if col in x.columns: x.loc[0, col] = float(v)
    
    total = x[f_cols].sum(axis=1).iloc[0]
    if total > 0: x[f_cols] = x[f_cols].div(total, axis=0) * 100
    
    preds = {t: model_package['models'][t].predict(x[f_cols])[0] for t in model_package['models']}
    
    # CTE 修正邏輯
    cte = preds.get('cte_1e-6_per_C', 0)
    if cte < 3.5:
        b2o3 = composition_dict.get('B2O3', 0)
        correction = 0.15 + (0.03 * b2o3)
        preds['cte_1e-6_per_C'] = max(cte - correction, 0.1)
    return preds

try:
    model_pkg = load_model()
    st.title('🔬 玻璃物理性質預測系統')
    
    default_oxides = ['SiO2', 'Al2O3', 'B2O3', 'CaO', 'MgO', 'Na2O', 'K2O', 'TiO2', 'Li2O']
    input_data = {}
    cols = st.columns(3)
    for i, oxide in enumerate(default_oxides):
        with cols[i % 3]:
            input_data[oxide] = st.number_input(f'{oxide}', value=0.0, step=0.1)
            
    if st.button('🚀 執行預測', use_container_width=True):
        res = predict_with_correction(input_data, model_pkg)
        st.subheader('📊 預測結果')
        r1, r2, r3 = st.columns(3)
        r1.metric('CTE', f"{res['cte_1e-6_per_C']:.4f}")
        r2.metric('Young Modulus', f"{res['young_modulus_GPa']:.2f} GPa")
        r3.metric('T@log3', f"{res['T_at_1E3_dPas_C']:.1f} °C")
except Exception as e:
    st.error(f'載入失敗: {e}')
