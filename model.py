import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Cognos vs PBI Validator", layout="wide")
st.title("📊 Cognos vs Power BI Excel Validation Tool")

# File upload
cognos_file = st.file_uploader("Upload Cognos Excel File", type=["xlsx"], key="cognos")
pbi_file = st.file_uploader("Upload Power BI Excel File", type=["xlsx"], key="pbi")

def load_excel(file):
    xls = pd.read_excel(file, sheet_name=None)
    return xls

def get_common_columns(df1, df2):
    return list(set(df1.columns) & set(df2.columns))

def generate_validation_report(cognos_df, pbi_df, dims):
    shared_cols = list(set(cognos_df.columns) & set(pbi_df.columns))
    if not shared_cols:
        raise ValueError("No common columns between Cognos and PBI sheets.")

    if not dims:
        st.warning("⚠️ No dimensions selected. Using all shared columns as fallback.")
        dims = shared_cols

    cognos_df[dims] = cognos_df[dims].fillna('NAN')
    pbi_df[dims] = pbi_df[dims].fillna('NAN')

    cognos_measures = [col for col in shared_cols if col not in dims and np.issubdtype(cognos_df[col].dtype, np.number)]
    pbi_measures = [col for col in shared_cols if col not in dims and np.issubdtype(pbi_df[col].dtype, np.number)]
    all_measures = list(set(cognos_measures) & set(pbi_measures))

    if not all_measures:
        raise ValueError("No common numeric columns found for measure comparison.")

    cognos_agg = cognos_df.groupby(dims)[all_measures].sum().reset_index()
    pbi_agg = pbi_df.groupby(dims)[all_measures].sum().reset_index()

    cognos_agg['unique_key'] = cognos_agg[dims].astype(str).agg('-'.join, axis=1).str.upper()
    pbi_agg['unique_key'] = pbi_agg[dims].astype(str).agg('-'.join, axis=1).str.upper()

    validation_report = pd.DataFrame({'unique_key': list(set(cognos_agg['unique_key']) | set(pbi_agg['unique_key']))})

    for dim in dims:
        validation_report[dim] = validation_report['unique_key'].map(dict(zip(cognos_agg['unique_key'], cognos_agg[dim])))
        validation_report[dim].fillna(validation_report['unique_key'].map(dict(zip(pbi_agg['unique_key'], pbi_agg[dim]))), inplace=True)

    validation_report['presence'] = validation_report['unique_key'].apply(
        lambda key: 'Present in Both' if key in cognos_agg['unique_key'].values and key in pbi_agg['unique_key'].values
        else ('Present in Cognos' if key in cognos_agg['unique_key'].values else 'Present in PBI')
    )

    for measure in all_measures:
        validation_report[f'{measure}_Cognos'] = validation_report['unique_key'].map(dict(zip(cognos_agg['unique_key'], cognos_agg[measure])))
        validation_report[f'{measure}_PBI'] = validation_report['unique_key'].map(dict(zip(pbi_agg['unique_key'], pbi_agg[measure])))
        validation_report[f'{measure}_Diff'] = validation_report[f'{measure}_PBI'].fillna(0) - validation_report[f'{measure}_Cognos'].fillna(0)

    column_order = ['unique_key'] + dims + ['presence'] + \
                   [col for measure in all_measures for col in 
                    [f'{measure}_Cognos', f'{measure}_PBI', f'{measure}_Diff']]
    validation_report = validation_report[column_order]

    return validation_report

if cognos_file and pbi_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_cognos, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_pbi:

        tmp_cognos.write(cognos_file.read())
        tmp_pbi.write(pbi_file.read())
        tmp_cognos.flush()
        tmp_pbi.flush()

        cognos_sheets = load_excel(tmp_cognos.name)
        pbi_sheets = load_excel(tmp_pbi.name)

        selected_cognos_sheet = st.selectbox("Select Cognos Sheet", options=list(cognos_sheets.keys()))
        selected_pbi_sheet = st.selectbox("Select Power BI Sheet", options=list(pbi_sheets.keys()))

        cognos_df = cognos_sheets[selected_cognos_sheet]
        pbi_df = pbi_sheets[selected_pbi_sheet]

        shared_columns = get_common_columns(cognos_df, pbi_df)
        st.markdown("### 🧩 Select Dimensions (keys to group by)")
        dims_selected = st.multiselect("Select columns to use as dimensions", options=shared_columns)

        if st.button("🧪 Run Validation"):
            try:
                report = generate_validation_report(cognos_df, pbi_df, dims_selected)
                st.success("Validation complete!")
                st.dataframe(report)

                csv = report.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Validation Report", data=csv, file_name="validation_report.csv", mime="text/csv")

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
