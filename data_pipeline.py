"""
ASSESSMENT 2
filename: datapipeline.py
"""

# LIBRARIES
import csv
import pandas as pd
import numpy as np
from IPython.display import display


'''
Data Cleaning and Sanitization 

'''

def determine_efficiency_group(energy) -> str:
    """
    Categorizes into the appropriate efficiency group based on energy consumption
    :param energy: of each record (in kWh)
    :return: str
    """
    if energy < 14:
        return "Hyper-Efficient"
    elif 14 <= energy < 16:
        return "Highly Efficient"
    elif 16 <= energy < 20:
        return "Standard"
    elif energy >= 20:
        return "Low Efficiency"
    else:
        return "Standard"


def is_peak_hour_flag (start, duration, overstay) -> int :
    """
    Calculate if the charging occurred during peak hours (12:00 - 17:00)
    :param start: start hour
    :param duration: in minutes
    :param overstay: in minutes
    :return: Boolean (True if stayed in peak hour)
    """
    total_duration_hour = (duration + overstay) / 60
    end_hour = start + total_duration_hour
    if 12 < start < 18:
        return 1
    elif 12 < end_hour < 18:
        return 1
    else:
        return 0


def compute_charging_power(df) -> pd.DataFrame:
    """
        Determine Efficiency group of each record by applying lambda function and create new feature "Efficiency_Group"
        Compute charging power rate using (`Power_kW = Energy_kWh / (Duration_Mins / 60)`)
        Store Charging Power in DataFrame
        Add a feature/column named Peak_Hour_Flag as Boolean

        Statistical Aggregation:
        group-by summaries tables comparing average energy consumption and temperature

        :param df: Dataframe from clean_data()
        :return: pandas DataFrame
    """
    df["Efficiency_Group"] = df["Energy_kWh"].apply(lambda x: determine_efficiency_group(x))

    df["Power_kW"] = round(df["Energy_kWh"] / (df["Duration_Mins"] / 60 ), 2)

    df["Peak_Hour_Flag"] =df.apply(lambda row:
                                   is_peak_hour_flag(row["Start_Hour"],
                                                     row["Duration_Mins"],
                                                     row["Overstay_Mins"]),
                                   axis=1)

    charger_group_summary = df[["Power_kW", "Battery_Temp_C", "Charger_Type"]].groupby("Charger_Type").mean()
    peak_hour_summary = df[["Power_kW", "Battery_Temp_C", "Peak_Hour_Flag"]].groupby("Peak_Hour_Flag").mean()
    print("\n")
    print("----- SUMMARY TABLES ------")
    print("----- Across Charger Types ------")
    display(charger_group_summary)
    print("----- Peak Hour Flags ------")
    display(peak_hour_summary)

    return df

def clean_data(df) -> pd.DataFrame:
    """
        Returns a list of column names with missing/null values
        Fill in all missing values with median imputation
        Remove duplicate session records with drop.duplicates()

        :return: pandas DataFrame
    """

    print("-----FOUND MISSING VALUES AT COLUMN: -----")
    columns_with_null = df.columns[df.isna().sum() > 0].tolist()
    print(columns_with_null)

    for col in columns_with_null:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # This removes all duplicate records from the Dataframe
    df.drop_duplicates()

    return df


def data_pipeline(file_in, file_out) -> None:
    """
    Open file_in with pandas read_csv -> DataFrame
    Convert numerical features to appropriate numerical types
    Pass Dataframe to clean_data -> DataFrame
    This cleans the data off missing values and duplicates

    Pass the cleaned DF to compute_charging_power() -> DataFrame
    This function adds the columns : Efficiency_Group, Power_kW, Peak_Hour_Flag

    Store the final cleaned DataFrame to `assets/cleaned_ev_dataset.csv`

    :return: None
    """

    try:
        df = pd.read_csv(file_in, index_col=0)

        df['Start_Hour'] = pd.to_numeric(df['Start_Hour'], errors='coerce')
        df['Duration_Mins'] = pd.to_numeric(df['Duration_Mins'], errors='coerce')
        df['Energy_kWh'] = pd.to_numeric(df['Energy_kWh'], errors='coerce')
        df['Overstay_Mins'] = pd.to_numeric(df['Overstay_Mins'], errors='coerce')
        df['Battery_Temp_C'] = pd.to_numeric(df['Battery_Temp_C'], errors='coerce')

        # Clean dataframe
        cleaned = clean_data(df)
        # Add other features
        computed_charging_power = compute_charging_power(cleaned)
        # Write dataframe to new file
        computed_charging_power.to_csv(file_out, index=False)

    except FileNotFoundError:
        print("[FILE ERROR] File not found. Check filename or file path.")
    else:
        print(f'[FILE SUCCESS] File `{file_in}` was successfully cleaned.')
    finally:
        print('Files are closed.')


# START HERE
if __name__ == '__main__':
    # Declare file paths of the infile and outfile here
    ev_in = 'assets/ev_campus_dataset.csv'
    ev_out ='assets/cleaned_ev_dataset.csv'
    data_pipeline(ev_in, ev_out)

