import pandas as pd


def load_tabular_data(uploaded_file):
    """
    读取 CSV 或 Excel 文件。

    Args:
        uploaded_file: Streamlit UploadedFile 对象

    Returns:
        pd.DataFrame: 读取的数据表

    Raises:
        ValueError: 不支持的文件格式
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("仅支持 CSV、XLS、XLSX 文件")
