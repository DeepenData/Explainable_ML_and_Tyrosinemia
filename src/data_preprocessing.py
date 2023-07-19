import logging
from functools import cache

import gspread
import pandas as pd

@cache
def sheet_to_dataframe(
    key : str,          # The Google Sheet name key
    gc : gspread.Client # The client for Google Sheets # TODO: default client in environment?
    ) -> pd.DataFrame:
    """Reads a Google Sheet into a DataFrame"""

    # Now you can open your Google Sheets document
    spreadsheet = gc.open_by_key( key )  # carga todo el google sheet

    # Get the list of all worksheets in the spreadsheet
    worksheets = spreadsheet.worksheets()  # saca todas las worksheets
    logging.debug(f"Reading from Google Sheets data\n\tFILE {spreadsheet}\n\tSHEETS {worksheets}")

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
        .set_index(['tiempo','código','pacientes'])
        .apply(pd.to_numeric, axis='index', errors = 'coerce')
        .convert_dtypes()
        .astype(dtype='float64', errors='ignore') # BUG: THis is to coherce to 'float64' as Numpy doesn't support pd.NaT
    )

    return final_df


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Sanity check for comparing dataframes"""
    # Read CSV files into pandas DataFrames
    # Compare number of columns
    if len(df1.columns) != len(df2.columns):
        logging.debug(
            f"Number of columns are different. CSV1 has {len(df1.columns)} columns and CSV2 has {len(df2.columns)} columns."
        )
    else:
        logging.debug(f"Both CSV files have the same number of columns: {len(df1.columns)}")

    # Compare column names
    if set(df1.columns) != set(df2.columns):
        logging.debug("Column names are different:")
        logging.debug("Columns in first CSV but not in second: ", set(df1.columns) - set(df2.columns))
        logging.debug("Columns in second CSV but not in first: ", set(df2.columns) - set(df1.columns))
    else:
        logging.debug("Column names are the same.")

    # Compare column order
    if df1.columns.tolist() != df2.columns.tolist():
        logging.debug("Column order is different.")
        logging.debug("First CSV columns: ", df1.columns.tolist())
        logging.debug("Second CSV columns: ", df2.columns.tolist())
    else:
        logging.debug("Column order is the same.")

    # Compare column data types
    dtype_diff = False
    for column in df1.columns.intersection(df2.columns):
        if df1[column].dtype != df2[column].dtype:
            dtype_diff = True
            logging.debug(f"Column '{column}' has different data types:")
            logging.debug(f"Data type in first CSV: {df1[column].dtype}")
            logging.debug(f"Data type in second CSV: {df2[column].dtype}")
    if not dtype_diff:
        logging.debug("All common columns have the same data types.")


# Common filtering of the cohorts, so they pass the same names
def cohort_filter(df : pd.DataFrame, OPBG : list[str], REQUIERE_COMPLETE : list[str]) -> pd.DataFrame:
    """Drops the rows (samples) that have NaNs in any of the REQUIERE_COMPLETE features"""
    return df[ OPBG ].dropna(axis='rows', subset=REQUIERE_COMPLETE)