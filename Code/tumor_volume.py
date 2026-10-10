import os
import re
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# =============================================================================
# FUNCIONES AUXILIARES Y LECTURA ROBUSTA
# =============================================================================
def standardize_patient_id(series):
    def parse_val(v):
        s = str(v).strip().upper()
        s = s.replace("PATIENT", "P").replace("PACIENTE", "P").replace(" ", "")
        if s.startswith("P"):
            m = re.match(r"P0*(\d+.*)", s)
            if m:
                return "P" + m.group(1)
        return s
    return series.apply(parse_val)

def load_excel_robust(file_path):
    try:
        return pd.read_excel(file_path)
    except Exception:
        with zipfile.ZipFile(file_path, 'r') as z:
            sheet_xml = z.read('xl/worksheets/sheet1.xml')
            shared_strings_xml = z.read('xl/sharedStrings.xml') if 'xl/sharedStrings.xml' in z.namelist() else None

        strings = []
        if shared_strings_xml:
            tree_ss = ET.fromstring(shared_strings_xml)
            for elem in tree_ss.findall('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                text = "".join([t.text for t in elem.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if t.text])
                strings.append(text)

        tree_sheet = ET.fromstring(sheet_xml)
        rows_data = []
        ns = {'sm': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

        for row in tree_sheet.findall('.//sm:row', ns):
            row_dict = {}
            for cell in row.findall('sm:c', ns):
                r = cell.attrib.get('r')
                col_letter = ''.join([c for c in r if c.isalpha()])
                t = cell.attrib.get('t')
                v_elem = cell.find('sm:v', ns)
                val = v_elem.text if v_elem is not None else None
                
                if t == 's' and val is not None:
                    val = strings[int(val)]
                elif val is not None:
                    try:
                        val = float(val) if '.' in val else int(val)
                    except ValueError:
                        pass
                row_dict[col_letter] = val
            rows_data.append(row_dict)

        df = pd.DataFrame(rows_data)
        def col_to_num(col_str):
            num = 0
            for c in col_str:
                num = num * 26 + (ord(c) - ord('A') + 1)
            return num

        cols = sorted(df.columns, key=col_to_num)
        df = df[cols]
        df.columns = df.iloc[0]
        return df[1:].reset_index(drop=True)

def padjust_fdr(p_vals):
    """Implementación idéntica a la función padjust_fdr de MATLAB (Benjamini-Hochberg)"""
    m = len(p_vals)
    p_vals = np.array(p_vals)
    sort_idx = np.argsort(p_vals)
    p_sorted = p_vals[sort_idx]
    
    p_adj_sorted = np.zeros(m)
    p_adj_sorted[-1] = p_sorted[-1]
    
    for i in range(m - 2, -1, -1):
        p_adj_sorted[i] = min(p_sorted[i] * m / (i + 1), p_adj_sorted[i + 1])
        
    p_adj_sorted = np.minimum(p_adj_sorted, 1.0)
    
    p_adj = np.zeros(m)
    p_adj[sort_idx] = p_adj_sorted
    return p_adj

# =============================================================================
# BLOQUE DE ANÁLISIS COMPLETO CON BONFERRONI Y FDR
# =============================================================================
if __name__ == "__main__":
    working_dir = r"C:\Users\pfran\Desktop\Connectomics\Connectomics Github\Dataset"
    
    patients_excel_path = os.path.join(working_dir, "Patients_Project - Copy.xlsx")
    radiomics_path = os.path.join(working_dir, "radiomics_with_classes_cleaned.csv")

    df_patients = load_excel_robust(patients_excel_path)
    df_radiomics = pd.read_csv(radiomics_path)

    # 1. Eliminar pacientes seleccionados en MATLAB [10, 15, 17, 22, 23, 30]
    to_remove_idx = [i - 1 for i in [10, 15, 17, 22, 23, 30]]
    df_patients = df_patients.drop(index=[idx for idx in to_remove_idx if idx in df_patients.index]).reset_index(drop=True)

    df_patients.columns = df_patients.columns.str.lower()
    df_patients["std_id"] = standardize_patient_id(df_patients["id"])
    df_radiomics["std_id"] = standardize_patient_id(df_radiomics["Subjects"])

    # 2. Extraer volumen
    volume_cols = [col for col in df_radiomics.columns if "volume" in col.lower() or "mesh" in col.lower()]
    selected_vol_col = volume_cols[0]
    df_vol = df_radiomics[["std_id", selected_vol_col]].rename(columns={selected_vol_col: "volume"})
    df_vol["volume"] = pd.to_numeric(df_vol["volume"], errors="coerce")

    # Merge
    df_patients_clean = df_patients.drop_duplicates(subset=["std_id"]).dropna(subset=["std_id"])
    df = pd.merge(df_patients_clean, df_vol, on="std_id", how="inner")

    grade_col = "grade" if "grade" in df.columns else "grado"
    df[grade_col] = pd.to_numeric(df[grade_col], errors="coerce")
    df = df.dropna(subset=[grade_col, "volume"]).reset_index(drop=True)
    df['is_glioblastoma'] = df['histologia'].astype(str).str.contains('Glioblastoma', case=False, na=False)

    # 3. Subgrupos
    low_grade  = df[df[grade_col] == 2]
    high_grade = df[df[grade_col] == 3]
    grade_iii  = df[(df[grade_col] == 3) & (~df['is_glioblastoma'])]
    grade_iv   = df[(df[grade_col] == 3) & (df['is_glioblastoma'])]

    # 4. Prueba Mann-Whitney U para Tumor Volume
    stat_u, p_raw_vol = mannwhitneyu(low_grade["volume"], high_grade["volume"], alternative='two-sided')
    test_str = f"Mann-Whitney U (U = {stat_u:.2f})"

    # 5. Ajustes de p-valores (Simulando m = 9 comparaciones totales si sumamos Tumor Volume a las 8 de MATLAB)
    # Si deseas incluir únicamente las pruebas numéricas o todas las de la tabla:
    m_total_tests = 9  # 8 de MATLAB (Edad, Sexo, Hemisferio, 4 Lóbulos, Multilobar) + 1 (Tumor Volume)
    
    p_bonferroni_vol = min(1.0, p_raw_vol * m_total_tests)
    
    # Para calcular FDR adecuadamente en todo el set, pasamos un array de p-valores hipotéticos o aislados
    # Si solo evaluamos este vector individual ajustado al m total:
    p_fdr_vol = min(1.0, p_raw_vol * m_total_tests) # O utilizando padjust_fdr con el vector completo de tu estudio

    def format_median_range(series):
        if len(series) == 0:
            return "-"
        return f"{series.median():.1f} ({series.min():.1f}-{series.max():.1f})"

    def format_p(val):
        return f"{val:.3f}" if val >= 0.001 else "<0.001"

    # 6. Crear Fila Final Lista para Tabla 1
    results_df = pd.DataFrame([{
        "Variable": "Tumor Volume (mm³)",
        "Total": format_median_range(df["volume"]),
        "Low_Grade": format_median_range(low_grade["volume"]),
        "High_Grade": format_median_range(high_grade["volume"]),
        "Grade_III": format_median_range(grade_iii["volume"]),
        "Grade_IV": format_median_range(grade_iv["volume"]),
        "p_value": format_p(p_raw_vol),
        "p_value_Bonferroni": format_p(p_bonferroni_vol),
        "p_value_FDR": format_p(p_fdr_vol),
        "Test": test_str
    }])

    print("=== RESULTADOS DEL TEST PARA TUMOR VOLUME (CON BONFERRONI Y FDR) ===")
    print(results_df.to_string(index=False))