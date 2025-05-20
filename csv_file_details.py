import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="CSV Data Profiler", layout="wide")

st.title("📊 CSV Data Profiler & Data Quality Dashboard")
st.markdown("""
A professional, interactive overview and data quality scan for your dataset.
- **Blue**: Info
- **Green**: OK
- **Yellow**: Warning
- **Red**: Critical Issue
""")

# Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# --- 1. Basic Info ---
with st.expander("1️⃣ Basic Information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Columns:**")
        st.write(df.columns.tolist())
        st.markdown("**Shape:**")
        st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
    with col2:
        st.markdown("**Data Types:**")
        st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={'index': 'Column', 0: 'Type'}))

# --- 2. Missing & Infinite Values ---
with st.expander("2️⃣ Missing & Infinite Values", expanded=True):
    missing = df.isnull().sum()
    inf = np.isinf(df.select_dtypes(include=[np.number])).sum()
    if missing.any():
        st.warning("⚠️ Missing values detected!")
        st.dataframe(missing[missing > 0].reset_index().rename(columns={0: 'Missing Count', 'index': 'Column'}))
    else:
        st.success("No missing values detected.")
    if inf.any():
        st.error("❗ Infinite values detected!")
        st.dataframe(inf[inf > 0].reset_index().rename(columns={0: 'Inf Count', 'index': 'Column'}))
    else:
        st.info("No infinite values detected.")

# --- 3. Duplicates ---
with st.expander("3️⃣ Duplicated Rows", expanded=True):
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        st.warning(f"⚠️ There are {duplicates} duplicated rows.")
    else:
        st.success("No duplicated rows detected.")

# --- 4. Descriptive Statistics ---
with st.expander("4️⃣ Descriptive Statistics", expanded=False):
    st.markdown("**Numerical Columns:**")
    st.dataframe(df.describe().T)
    st.markdown("**Categorical Columns:**")
    st.dataframe(df.describe(include=['object']).T)

# --- 5. Unique Values & Cardinality ---
with st.expander("5️⃣ Unique Values & Cardinality", expanded=False):
    for col in df.columns:
        nunique = df[col].nunique()
        if df[col].dtype == 'object':
            if nunique > 50:
                st.error(f"🔴 {col}: {nunique} unique values (High cardinality!)")
            elif nunique > 10:
                st.warning(f"🟡 {col}: {nunique} unique values")
            else:
                st.info(f"🔵 {col}: {nunique} unique values")
            st.write(df[col].unique()[:20])
        else:
            if nunique == 1:
                st.error(f"🔴 {col}: Only one unique value (Constant column)")
            elif nunique == df.shape[0]:
                st.info(f"🔵 {col}: All values unique (Possible ID column)")

# --- 6. Correlation Matrix ---
with st.expander("6️⃣ Correlation Matrix (Numerical Columns)", expanded=False):
    st.dataframe(df.corr(numeric_only=True).round(2))

# --- 7. Advanced Data Quality Scan ---
with st.expander("7️⃣ Advanced Data Quality Scan", expanded=True):
    issues = False

    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        st.error(f"🔴 Columns with only one unique value: {constant_cols}")
        issues = True

    # All-unique columns
    all_unique = [col for col in df.columns if df[col].nunique() == len(df)]
    if all_unique:
        st.info(f"🔵 Columns with all unique values (possible IDs): {all_unique}")

    # Negative values in numeric columns
    neg_cols = [col for col in df.select_dtypes(include=['number']).columns if (df[col] < 0).any()]
    if neg_cols:
        st.warning(f"🟡 Negative values detected in columns (check if expected): {neg_cols}")
        issues = True

    # Outlier scan (z-score > 4)
    from scipy.stats import zscore
    outlier_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if (np.abs(zscore(df[col].dropna())) > 4).sum() > 0:
            outlier_cols.append(col)
    if outlier_cols:
        st.warning(f"🟡 Possible outliers detected in: {outlier_cols}")

    # Skewness
    skewed = df.select_dtypes(include=[np.number]).skew().abs()
    highly_skewed = skewed[skewed > 2]
    if not highly_skewed.empty:
        st.warning(f"🟡 Highly skewed columns: {list(highly_skewed.index)}")

    # Mixed types in columns
    mixed_type_cols = []
    for col in df.columns:
        types = df[col].apply(lambda x: type(x)).nunique()
        if types > 1:
            mixed_type_cols.append(col)
    if mixed_type_cols:
        st.error(f"🔴 Columns with mixed data types: {mixed_type_cols}")
        issues = True

    if not issues and not missing.any() and duplicates == 0:
        st.success("No obvious data bugs or faults detected.")

# --- 8. Sample Data ---
with st.expander("8️⃣ Sample Data", expanded=False):
    st.dataframe(df)

st.markdown("---")
st.caption("Generated by GitHub Copilot · Streamlit Data Profiler")