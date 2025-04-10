import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
from fuzzywuzzy import process

st.set_page_config(layout="wide")
st.title("Cognos vs Power BI Column Checklist")

cognos_file = st.file_uploader("Upload Cognos Excel file", type=["xlsx"], key="cognos")
pbi_file = st.file_uploader("Upload Power BI Excel file", type=["xlsx"], key="pbi")

model_name = st.text_input("Enter Model Name")
report_name = st.text_input("Enter Report Name")

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

    # Clean string columns
    cognos_df = cognos_df.apply(lambda x: x.str.upper().str.strip() if x.dtype == "object" else x)
    pbi_df = pbi_df.apply(lambda x: x.str.upper().str.strip() if x.dtype == "object" else x)

    def get_best_match(row, candidates, dims):
        """Return the best matching row from candidates using dimension similarity"""
        combined = row[dims].astype(str).agg("-".join, axis=0)
        choices = candidates[dims].astype(str).agg("-".join, axis=1).tolist()
        best_match_str, score = process.extractOne(combined, choices)
        best_row = candidates[candidates[dims].astype(str).agg("-".join, axis=1) == best_match_str]
        return best_row if not best_row.empty else None

    def generate_validation_report(cognos_df, pbi_df):
        dims = [col for col in cognos_df.columns if col in pbi_df.columns and 
                (cognos_df[col].dtype == 'object' or '_id' in col.lower() or '_key' in col.lower())]

        cognos_df[dims] = cognos_df[dims].fillna('NAN')
        pbi_df[dims] = pbi_df[dims].fillna('NAN')

        measures = list(set(cognos_df.select_dtypes(include=np.number).columns) & set(pbi_df.select_dtypes(include=np.number).columns))

        matched_rows = []

        unmatched_cognos = cognos_df.copy()
        unmatched_pbi = pbi_df.copy()

        for _, c_row in cognos_df.iterrows():
            match = get_best_match(c_row, unmatched_pbi, dims)
            if match is not None:
                p_row = match.iloc[0]
                diff = {f'{col}_Diff': p_row[col] - c_row[col] for col in measures}
                matched_rows.append({**c_row.to_dict(), **{f'{col}_PBI': p_row[col] for col in measures}, **diff, 'presence': 'Similar Match'})
                unmatched_pbi = unmatched_pbi.drop(match.index)
                unmatched_cognos = unmatched_cognos.drop(_)

        # Add remaining unmatched rows
        for _, row in unmatched_cognos.iterrows():
            diff = {f'{col}_Diff': -row[col] for col in measures}
            matched_rows.append({**row.to_dict(), **{f'{col}_PBI': np.nan for col in measures}, **diff, 'presence': 'Present in Cognos'})

        for _, row in unmatched_pbi.iterrows():
            diff = {f'{col}_Diff': row[col] for col in measures}
            base_row = {col: np.nan for col in cognos_df.columns}
            matched_rows.append({**base_row, **{f'{col}_PBI': row[col] for col in measures}, **diff, 'presence': 'Present in PBI'})

        return pd.DataFrame(matched_rows)

    def column_checklist(cognos_df, pbi_df):
        cognos_columns = cognos_df.columns.tolist()
        pbi_columns = pbi_df.columns.tolist()
        checklist_df = pd.DataFrame({
            'Cognos Columns': cognos_columns + [''] * (max(len(pbi_columns), len(cognos_columns)) - len(cognos_columns)),
            'PowerBI Columns': pbi_columns + [''] * (max(len(pbi_columns), len(cognos_columns)) - len(pbi_columns))
        })
        checklist_df['Match'] = checklist_df.apply(lambda row: row['Cognos Columns'] == row['PowerBI Columns'], axis=1)
        return checklist_df

    def generate_diff_checker(validation_report):
        diff_columns = [col for col in validation_report.columns if col.endswith('_Diff')]
        diff_checker = pd.DataFrame({
            'Diff Column Name': diff_columns,
            'Sum of Difference': [validation_report[col].sum() for col in diff_columns]
        })
        presence_summary = {
            'Diff Column Name': 'All rows present in both',
            'Sum of Difference': 'Yes' if all(validation_report['presence'] == 'Similar Match') else 'No'
        }
        return pd.concat([diff_checker, pd.DataFrame([presence_summary])], ignore_index=True)

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

    validation_report = generate_validation_report(cognos_df, pbi_df)
    column_checklist_df = column_checklist(cognos_df, pbi_df)
    diff_checker_df = generate_diff_checker(validation_report)
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
    filename = f"{model_name}_{report_name}_ValidationReport_{today_date}.xlsx" if model_name and report_name else f"ValidationReport_{today_date}.xlsx"

    st.download_button(
        label="Download Excel Report",
        data=output,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
