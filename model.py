import streamlit as st
import pandas as pd

st.set_page_config(page_title="Excel Validator with Dimension Control", layout="wide")
st.title("📊 Cognos vs Power BI Column Validator")

# Upload files
cognos_file = st.file_uploader("Upload Cognos Excel", type=["xlsx"])
pbi_file = st.file_uploader("Upload Power BI Excel", type=["xlsx"])

if cognos_file and pbi_file:
    try:
        df_cognos = pd.read_excel(cognos_file)
        df_pbi = pd.read_excel(pbi_file)

        cognos_cols = set(df_cognos.columns)
        pbi_cols = set(df_pbi.columns)

        shared_cols = sorted(list(cognos_cols & pbi_cols))
        if not shared_cols:
            st.warning("No common columns found between Cognos and Power BI.")
        else:
            st.subheader("Column Match & Dimension Control")
            st.markdown("Check the columns you want to use as dimensions (will not be validated). These will be renamed with `_id` suffix.")

            # Checklist for each common column
            selected_dims = []
            dim_checkboxes = {}

            for col in shared_cols:
                checkbox = st.checkbox(f"Use as dimension: {col}", key=f"dim_{col}")
                if checkbox:
                    selected_dims.append(col)
                    dim_checkboxes[col] = True
                else:
                    dim_checkboxes[col] = False

            # Rename columns in both dataframes to *_id if selected as dimension
            for col in selected_dims:
                df_cognos.rename(columns={col: f"{col}_id"}, inplace=True)
                df_pbi.rename(columns={col: f"{col}_id"}, inplace=True)

            st.success(f"Selected {len(selected_dims)} column(s) as dimensions. These were renamed with `_id` suffix.")

            st.markdown("### ✅ Ready for validation logic here (e.g., difference checking, export, etc.)")
            st.dataframe(df_cognos.head(), use_container_width=True)
            st.dataframe(df_pbi.head(), use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
