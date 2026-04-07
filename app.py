import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Aarogya Live Command Center", page_icon="🚨", layout="wide")
st.title("🚨 Aarogya RL: Live ICU Command Center")

REPORT_FILE = "final_clinical_report.csv"

if os.path.exists(REPORT_FILE):
    # Load data and strip any accidental spaces from headers
    df = pd.read_csv(REPORT_FILE)
    df.columns = df.columns.str.strip()

    # --- DYNAMIC COLUMN PICKER ---
    # Finds the ID column (patient_id or id)
    id_col = next((c for c in ['patient_id', 'id'] if c in df.columns), df.columns[0])
    # Finds the Score column (stability_score or score)
    score_col = next((c for c in ['stability_score', 'score'] if c in df.columns), None)

    if score_col:
        # Use Session State so the Emergency Button works instantly
        if 'data' not in st.session_state:
            st.session_state.data = df
        
        working_df = st.session_state.data

        # --- CRITICAL ALERTS SECTION ---
        st.subheader("⚠️ Active Critical Alerts")
        critical = working_df[working_df[score_col] < 0.7]

        if not critical.empty:
            for idx, row in critical.iterrows():
                c1, c2, c3 = st.columns([1, 2, 1])
                c1.error(f"PATIENT: {row[id_col]}")
                c2.warning(f"Current Stability: {row[score_col]:.2f}")
                if c3.button(f"⚡ STABILIZE", key=f"btn_{row[id_col]}"):
                    # Manual Override Logic
                    st.session_state.data.at[idx, score_col] = 0.95
                    st.balloons()
                    st.success(f"Emergency meds administered to {row[id_col]}!")
                    st.rerun()
        else:
            st.success("All patients are currently stable under AI monitoring.")

        # --- FULL TABLE ---
        st.divider()
        st.subheader("📋 Patient Population Overview")
        st.dataframe(working_df, use_container_width=True)
    else:
        st.error(f"Could not find a score column. Found: {list(df.columns)}")
else:
    st.error(f"File '{REPORT_FILE}' not found. Please run 'master_audit.py' first.")
    