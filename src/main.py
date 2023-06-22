# %% 
# BASE PARA WORKFLOW
import logging

logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="basic.log",
    )

logging.info("Starting workflow")

# %%
# PARTE DE IMPORTAR PARSEAR LOS SHEETS DE CAMI
import gspread
import pandas as pd

def sheet_to_dataframe(
    key : str,          # The Google Sheet name key
    gc : gspread.Client # The client for Google Sheets # TODO: default client in environment?
    ) -> pd.DataFrame:
    """Reads a Google Sheet into a DataFrame"""

    # Now you can open your Google Sheets document
    spreadsheet = gc.open_by_key( key )  # carga todo el google sheet

    # Get the list of all worksheets in the spreadsheet
    worksheets = spreadsheet.worksheets()  # saca todas las worksheets
    logging.info(f"Reading from Google Sheets data\n\tFILE {spreadsheet}\n\tSHEETS {worksheets}")

    # Create a list to hold all dataframes
    df_list : list[pd.DataFrame] = []

    # Loop through all worksheets and convert each into a dataframe
    for worksheet in worksheets[3:]:  # saqué las primeras 3 pq no eran de los tiempos de los pacientes

        # DATA INTO A DATAFRAME
        # Get all the values in the worksheet
        data : list[list] = worksheet.get_all_values()
        # Convert the data to a dataframe
        df = pd.DataFrame(data)
        # Set the column names to the values in the first row
        df.columns = df.iloc[0]
        # eliminar del df la primera columna
        df = df.iloc[1:]

        # CLEANSE OF THE COLUMN NAMES
        # sacar los espacios y que quede todo en minuscula los nombres de las  columnas
        df.columns = df.columns.str.strip().str.lower()
        # eliminar cualquier numero que esté antes de una palabra
        df.columns = df.columns.str.replace(r"^\b(?:13|12|11|10|[1-9])\b", " ", regex=True)
        # en minusculas eliminar los espacios
        df.columns = df.columns.str.strip().str.lower()

        logging.info(
            f"SHEET {worksheet} has {df.shape[0]} entries, {df.shape[1]} features\n\t {df.columns}"
        )

        # Add the dataframe to the list
        df_list.append(df)  # una lista de dataframes

    # PARSE ALL COLUMNS INTO A COMMON LIST
    all_columns = pd.Index([])
    for df in df_list:
        all_columns = all_columns.union(df.columns)

    logging.info(
        f"Final list of columns: {all_columns}"
    )

    final_df = (pd.concat(df_list, axis='index')
        .set_index(['tiempo','código'])
        .apply(pd.to_numeric, errors = 'ignore')
        .convert_dtypes()
    )

    return final_df


def compare_dataframes(df1 : pd.DataFrame, df2 : pd.DataFrame) -> None:
    """Sanity check for comparing dataframes"""
    # Read CSV files into pandas DataFrames
    # Compare number of columns
    if len(df1.columns) != len(df2.columns):
        print(f"Number of columns are different. CSV1 has {len(df1.columns)} columns and CSV2 has {len(df2.columns)} columns.")
    else:
        print(f"Both CSV files have the same number of columns: {len(df1.columns)}")


    # Compare column names
    if set(df1.columns) != set(df2.columns):
        print("Column names are different:")
        print("Columns in first CSV but not in second: ", set(df1.columns) - set(df2.columns))
        print("Columns in second CSV but not in first: ", set(df2.columns) - set(df1.columns))
    else:
        print("Column names are the same.")

    # Compare column order
    if df1.columns.tolist() != df2.columns.tolist():
        print("Column order is different.")
        print("First CSV columns: ", df1.columns.tolist())
        print("Second CSV columns: ", df2.columns.tolist())
    else:
        print("Column order is the same.")

    # Compare column data types
    dtype_diff = False
    for column in df1.columns.intersection(df2.columns):
        if df1[column].dtype != df2[column].dtype:
            dtype_diff = True
            print(f"Column '{column}' has different data types:")
            print(f"Data type in first CSV: {df1[column].dtype}")
            print(f"Data type in second CSV: {df2[column].dtype}")
    if not dtype_diff:
        print("All common columns have the same data types.")

# %% ---- PROCESADO EN SI
# Conexion a Google Sheets
gc = gspread.service_account(
    # PARAM: service account for google sheets
    filename='/home/manu/.config/google-cloud/credentials/project-30463-38031804e4b0.json'
)

# df1 --- Cohorte de Chile
df1 = (
    sheet_to_dataframe(key = "1aHCPVXG9Lv9Eg_V7bGd9Zj9jGbXcS8m0lDp9s8HDF-c", gc=gc)
    .drop('nut status', axis = 'columns')
)
df1.to_csv('data/tirosinemia_1aHCPVXG9Lv9Lv.csv')

# df2 --- Cohorte de Italia
df2 = (
    sheet_to_dataframe(key = "1vpFD-SLCcub_QuUFohRyt9W7LI7vAWiGTQRSJxVnAro", gc=gc)
    .drop('data record (not for analysis)', axis = 'columns')
)
df2.to_csv('data/tirosinemia_italia_1vpFD-SLCcub.csv')


# %%
# PARTE DE PARSEAR DICHOS SHEETS A PARQUET DE ALEJANDRO

# %%
# PARTE DE USAR ESTO PARA CONSTRUIR Y TESTEAR MODELOS DE MANU
import ray

ray.init(ignore_reinit_error=True)  # Initialize ray cluster
