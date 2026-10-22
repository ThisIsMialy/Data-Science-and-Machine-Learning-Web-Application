"""
filename: datapipeline.py
"""

# LIBRARIES
import csv
import pandas as pd
import numpy as np
from IPython.display import display

'''
Data Cleaning and Sanitization 
1. Detect missing values 

Median Imputation: Replaces the missing value with the middle value
all the non-missing values 

Remove duplicate session records

'''

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
        Compute charging power rate using (`Power_kW = Energy_kWh / (Duration_Mins / 60)`)
        Store Charging Power in DataFrame

        Add a column named Peak Hour Flag as Boolean


        Statistical Aggregation
        group-by summaries tables comparing average energy consumption and temperature

        :param df: Dataframe from clean_data()
        :return: pandas DataFrame
    """

    print("COMPUTING CHARGING POWER RATE")
    df["Power_kW"] = df["Energy_kWh"] / (df["Duration_Mins"] / 60 )


    df["Peak_Hour_Flag"] =df.apply(lambda row:
                                   is_peak_hour_flag(row["Start_Hour"],
                                                     row["Duration_Mins"],
                                                     row["Overstay_Mins"]),
                                   axis=1)

    charger_group_summary = df[["Power_kW", "Battery_Temp_C", "Charger_Type"]].groupby("Charger_Type").mean()
    peak_hour_summary = df[["Power_kW", "Battery_Temp_C", "Peak_Hour_Flag"]].groupby("Peak_Hour_Flag").mean()

    # reverse = df[["Power_kW", "Battery_Temp_C", "Charger_Type"]].groupby("Battery_Temp_C").mean()

    print(display(charger_group_summary))
    display(peak_hour_summary)



    return df

def clean_data(headers, csv_reader) -> pd.DataFrame:

    """
        Open CSV file with open() and skips headers (store for later)
        Read file with csv.reader -> <class '_csv.reader'>
        Convert to list then to Pandas DataFrame
        Turn numerical values into numerical values with
        Returns a list of column names with missing/null values

        Fill in all missing values with median imputation

        Remove duplicate session records with drop.duplicates()

        :return: pandas DataFrame
    """

    df = pd.DataFrame(csv_reader, columns=headers)

    df['Start_Hour'] = pd.to_numeric(df['Start_Hour'], errors='coerce')
    df['Duration_Mins'] = pd.to_numeric(df['Duration_Mins'], errors='coerce')
    df['Energy_kWh'] = pd.to_numeric(df['Energy_kWh'], errors='coerce')
    df['Overstay_Mins'] = pd.to_numeric(df['Overstay_Mins'], errors='coerce')
    df['Battery_Temp_C'] = pd.to_numeric(df['Battery_Temp_C'], errors='coerce')

    print("FOUND MISSING VALUES AT COLUMN:")
    columns_with_null = df.columns[df.isna().sum() > 0].tolist()
    print(columns_with_null)

    for col in columns_with_null:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # median_duration = df['Duration_Mins'].median()
    # median_energy = df['Energy_kWh'].median()
    #
    # print(f"REPLACING MISSING VALUES AT COLUMN: Duration_Mins")
    # df['Duration_Mins'] = df['Duration_Mins'].fillna(median_duration)
    # df['Energy_kWh'] = df['Energy_kWh'].fillna(median_energy)
    #
    # df = df.fillna({
    #     'Duration_Mins': df['Duration_Mins'].median(),
    #     'Energy_kWh': df['Energy_kWh'].median(),
    # })

    # Before
    # df.info()

    df.drop_duplicates()

    # After
    # df.info()

    return df

    # clean_data = list(filter(
    #     lambda row: all(item for item in row),
    #     csv_reader
    # ))
    # csv_writer.writerow(headers)



def main():

    try:
        with (open(file='files/ev_campus_dataset.csv', encoding='utf-8', mode='r') as ev_file,
              open(file='clean_ev_campus_dataset.csv', encoding='utf-8', mode='w') as cleaned_file):
            csv_reader = csv.reader(ev_file)
            # csv_writer = csv.writer(cleaned_file)

            headers = next(csv_reader)

            cleaned = clean_data(headers, csv_reader)

            computed_charging_power = compute_charging_power(cleaned)




    except FileNotFoundError:
        print("File not found")
    else:
        print(f'Files successfully read.')
    finally:
        print('[STREAM IO] Data has been cleaned.')


if __name__ == '__main__':
    main()
