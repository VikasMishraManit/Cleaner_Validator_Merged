import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
from fuzzywuzzy import process

st.set_page_config(layout="wide")
st.title("Cognos vs Power BI Column Checklist")

# Upload files separately
cognos_file = st.file_uploader("Upload Cognos Excel file", type=["xlsx"], key="cognos")
pbi_file = st.file_uploader("Upload Power BI Excel file", type=["xlsx"], key="pbi")

model_name = st.text_input("Enter Model Name")
report_name = st.text_input("Enter Report Name")

def get_best_match(c_row, unmatched_pbi, dims):
    combined = '-'.join([str(c_row[dim]) if pd.notnull(c_row[dim]) else '' for dim in dims])
    choices = [
        '-'.join([str(p_row[dim]) if pd.notnull(p_row[dim]) else '' for dim in dims])
        for _, p_row in unmatched_pbi.iterrows()
    ]
    if not choices:
        return None, 0
    best_match_str, score = process.extractOne(combined, choices)
    return best_match_str, score

if cognos_file and pbi_file:
    cognos_df = pd.read_excel(cognos_file)
    pbi_df = pd.read_excel(pbi_file)

    cognos_columns = list(cognos_df.columns)
    pbi_columns = list(pbi_df.columns)
    common_columns = [col for col in cognos_columns if col in pbi_columns]

    st.markdown("### Select ID Columns")
    id_columns = []

    for col in common_columns:
        checked = st.checkbox(f"{col} is ID column", key=col)
        if checked:
            cognos_df.rename(columns={col: f"{col}_id"}, inplace=True)
            pbi_df.rename(columns={col: f"{col}_id"}, inplace=True)
            id_columns.append(f"{col}_id")
        else:
            id_columns.append(col)

    # Clean strings
    cognos_df = cognos_df.apply(lambda x: x.str.upper().str.strip() if x.dtype == "object" else x)
    pbi_df = pbi_df.apply(lambda x: x.str.upper().str.strip() if x.dtype == "object" else x)

    def generate_validation_report(cognos_df, pbi_df):
        dims = [col for col in cognos_df.columns if col in pbi_df.columns and 
                (cognos_df[col].dtype == 'object' or '_id' in col.lower() or '_key' in col.lower())]

        cognos_df[dims] = cognos_df[dims].fillna('NAN')
        pbi_df[dims] = pbi_df[dims].fillna('NAN')

        cognos_measures = [col for col in cognos_df.columns if col not in dims and np.issubdtype(cognos_df[col].dtype, np.number)]
        pbi_measures = [col for col in pbi_df.columns if col not in dims and np.issubdtype(pbi_df[col].dtype, np.number)]
        all_measures = list(set(cognos_measures) & set(pbi_measures))

        cognos_df['unique_key'] = cognos_df[dims].astype(str).agg('-'.join, axis=1).str.upper()
        pbi_df['unique_key'] = pbi_df[dims].astype(str).agg('-'.join, axis=1).str.upper()

        matched_keys = set(cognos_df['unique_key']) & set(pbi_df['unique_key'])

        matched_cognos = cognos_df[cognos_df['unique_key'].isin(matched_keys)].copy()
        matched_pbi = pbi_df[pbi_df['unique_key'].isin(matched_keys)].copy()

        unmatched_cognos = cognos_df[~cognos_df['unique_key'].isin(matched_keys)].copy()
        unmatched_pbi = pbi_df[~pbi_df['unique_key'].isin(matched_keys)].copy()

        fuzzy_matches = []
        used_pbi_keys = set()

        for _, c_row in unmatched_cognos.iterrows():
            match_str, score = get_best_match(c_row, unmatched_pbi, dims)
            if score > 80 and match_str not in used_pbi_keys:
                p_match = unmatched_pbi[unmatched_pbi[dims].astype(str).agg('-'.join, axis=1).str.upper() == match_str]
                if not p_match.empty:
                    fuzzy_matches.append((c_row, p_match.iloc[0]))
                    used_pbi_keys.add(match_str)

        for c_row, p_row in fuzzy_matches:
            matched_cognos = pd.concat([matched_cognos, pd.DataFrame([c_row])])
            matched_pbi = pd.concat([matched_pbi, pd.DataFrame([p_row])])

        matched_cognos.reset_index(drop=True, inplace=True)
        matched_pbi.reset_index(drop=True, inplace=True)

        report = matched_cognos[dims + all_measures].copy()
        for col in dims + all_measures:
            report[f'{col}_Cognos'] = matched_cognos[col]
            report[f'{col}_PBI'] = matched_pbi[col]
            if col in all_measures:
                report[f'{col}_Diff'] = matched_pbi[col] - matched_cognos[col]
            elif col in dims:
                report[f'{col}_Diff'] = matched_pbi[col] != matched_cognos[col]

        return report

    def column_checklist(cognos_df, pbi_df):
        cognos_columns = cognos_df.columns.tolist()
        pbi_columns = pbi_df.columns.tolist()
        checklist_df = pd.DataFrame({
            'Cognos Columns': cognos_columns + [''] * (max(len(pbi_columns), len(cognos_columns)) - len(cognos_columns)),
            'PowerBI Columns': pbi_columns + [''] * (max(len(pbi_columns), len(cognos_columns)) - len(pbi_columns))
        })
        checklist_df['Match'] = checklist_df.apply(lambda row: row['Cognos Columns'] == row['PowerBI Columns'], axis=1)
        return checklist_df

    def generate_diff_checker(report):
        diff_cols = [col for col in report.columns if col.endswith("_Diff")]
        return pd.DataFrame({
            "Diff Column Name": diff_cols,
            "Sum of Difference": [report[col].sum() if report[col].dtype != "bool" else report[col].sum() for col in diff_cols]
        })

    validation_report = generate_validation_report(cognos_df, pbi_df)
    column_checklist_df = column_checklist(cognos_df, pbi_df)
    diff_checker_df = generate_diff_checker(validation_report)

    checklist_data = {
        "S.No": range(1, 18),
        "Checklist": [
            "Database & Warehouse is parameterized (In case of DESQL Reports)",
            "All the columns of Cognos replicated in PBI (No extra columns)",
            "All the filters of Cognos replicated in PBI",
            "Filters working as expected (single/multi select as usual)",
            "Column names matching with Cognos",
            "Currency symbols to be replicated",
            "Filters need to be aligned vertically/horizontally",
            "Report Name & Package name to be written",
            "Entire model to be refreshed before publishing to PBI service",
            "Date Last refreshed to be removed from filter/table",
            "Table's column header to be bold",
            "Table style to not have grey bars",
            "Pre-applied filters while generating validation report?",
            "Dateformat to be YYYY-MM-DD [hh:mm:ss] in refresh date as well",
            "Sorting is replicated",
            "Filter pane to be hidden before publishing to PBI service",
            "Mentioned the exception in our validation document like numbers/columns/values mismatch (if any)"
        ],
        "Status - Level1": ["" for _ in range(17)],
        "Status - Level2": ["" for _ in range(17)]
    }
    checklist_df = pd.DataFrame(checklist_data)

    st.markdown("---")
    st.subheader("Validation Report Preview")
    st.dataframe(validation_report)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        checklist_df.to_excel(writer, sheet_name='Checklist', index=False)
        validation_report.to_excel(writer, sheet_name='Validation_Report', index=False)
        column_checklist_df.to_excel(writer, sheet_name='Column Checklist', index=False)
        diff_checker_df.to_excel(writer, sheet_name='Diff Checker', index=False)

    output.seek(0)
    today_date = datetime.today().strftime('%Y-%m-%d')
    dynamic_filename = f"{model_name}_{report_name}_ValidationReport_{today_date}.xlsx" if model_name and report_name else f"ValidationReport_{today_date}.xlsx"

    st.download_button(
        label="Download Excel Report",
        data=output,
        file_name=dynamic_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
