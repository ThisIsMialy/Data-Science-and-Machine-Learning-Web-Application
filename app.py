"""
app.py
ITS72604 Assessment 2

TASK 3 and 4:
Deploy Streamlit App

GOAL:
Create a 3-page Streamlit app and deploy it in Streamlit Cloud Community
Page 1: Executive Dashboard (KPI)
- display metrics from the cleaned_ev_dataset

Page 2:  Visual Analytics
- plots relationship of features
- Interactive scatterplot for user

Page 3: Live ML Inference
- Create an interactive way to predict the `Grid_Status_Label` given features
"""

import joblib
import pandas as pd
import streamlit as st
import seaborn as sns
from matplotlib import pyplot as plt

# Cache the CSV file for streamlit to load
@st.cache_data
def load_data():
    return pd.read_csv('assets/cleaned_ev_dataset.csv')

# Load the ML model using joblib
@st.cache_resource
def load_model():
    return joblib.load('assets/ev_grid_label_model.joblib')

st.set_page_config(page_title="Smart EV Charging Dashboard", layout="wide")

# Load data into df
df = load_data()
# Access ML model
model = load_model()

st.sidebar.title("Menu")

# Create a radio button in the sidebar to switch between pages
page = st.sidebar.radio("Select a page:",
                        [
                            "1. Executive Dashboard (KPI)",
                            "2. Visual Analytics",
                            "3. Live Machine Learning Inference"
                        ])


if page == "1. Executive Dashboard (KPI)":
    st.title("🔌 Taylor's Campus EV Charging Dashboard")
    st.subheader("Executive Dashboard")

    # Create 3 columns for metrics
    col1, col2, col3 = st.columns(3)

    total_energy = df['Energy_kWh'].sum()
    avg_duration = df['Duration_Mins'].mean()
    peak_sessions = df[df['Peak_Hour_Flag'] == 1].shape[0]

    # Create metrics cards for 3 columns
    with col1:
        st.metric(
            label="Total Energy Delivered",
            value=f"{total_energy:.2f} kWh",
            icon="🔥",
            border=True
        )

    with col2:
        st.metric(
            label="Average Session Duration",
            value=f"{avg_duration:.2f} min",
            delta_color="inverse",
            border=True
        )

    with col3:
        st.metric(
            label="Peak Sessions",
            value=f"{peak_sessions:,}",
            border=True
        )

    # Interactive Table for user

    selected_user = st.multiselect(
        "Select User Type",
        options=df['User_Type'].unique(),
        default=df['User_Type'].unique()
    )

    selected_charger = st.multiselect(
        "Select Charger Type",
        options=df['Charger_Type'].unique(),
        default=df['Charger_Type'].unique()
    )

    selected_grid_status = st.multiselect(
        "Select Grid Status Label",
        options=df['Grid_Status_Label'].unique(),
        default=df['Grid_Status_Label'].unique()
    )

    selected_efficiency = st.multiselect(
        "Select Efficiency Group",
        options=df['Efficiency_Group'].unique(),
        default=df['Efficiency_Group'].unique()
    )

    min_start, max_start = st.slider(
        "Start Hour Range",
        min_value=int(df['Start_Hour'].min()),
        max_value=int(df['Start_Hour'].max()),
        value=(int(df['Start_Hour'].min()), int(df['Start_Hour'].max()))
    )

    min_duration, max_duration = st.slider(
        "Session Duration Range",
        min_value=int(df['Duration_Mins'].min()),
        max_value=int(df['Duration_Mins'].max()),
        value=(int(df['Duration_Mins'].min()), int(df['Duration_Mins'].max()))
    )

    min_energy, max_energy = st.slider(
        "Energy in kWh Range",
        min_value=int(df['Energy_kWh'].min()),
        max_value=int(df['Energy_kWh'].max()),
        value=(int(df['Energy_kWh'].min()), int(df['Energy_kWh'].max()))
    )

    # Create a filtered Dataframe to select Features
    filtered_df = df[
        (df['User_Type'].isin(selected_user)) &
        (df['Charger_Type'].isin(selected_charger)) &
        (df['Grid_Status_Label'].isin(selected_grid_status)) &
        (df['Efficiency_Group'].isin(selected_efficiency)) &
        (df['Start_Hour'].between(min_start, max_start)) &
        (df['Duration_Mins'].between(min_duration, max_duration)) &
        (df['Energy_kWh'].between(min_energy, max_energy))

        ]

    # Show dataframe for user interactions
    st.dataframe(filtered_df, width='stretch')

# PAGE 2
elif page == "2. Visual Analytics":
    st.title("Visual Analytics")
    # Create 3 columns to place 3 different plots
    col1, col2, col3  = st.columns(3)

    # Create a Seaborn Scatterplot for Battery Temperature and Energy
    with col1:
        st.subheader("Temperature vs. Energy Scatter Plot")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.set_theme(style="whitegrid")
        sns.scatterplot(
            data=df,
            x='Energy_kWh',
            y='Battery_Temp_C',
            hue='Charger_Type',
            palette={'AC': '#2b5c8f', 'DC': '#e05a47'},
            alpha=0.75,
            s=75,
            ax=ax1
        )
        ax1.set_xlabel("Battery Temperature (°C)")
        ax1.set_ylabel("Energy Delivered (kWh)")
        st.pyplot(fig1)

    # Create a Seaborn Barplot for Battery Temperature and Energy
    with col2:
        st.subheader("Average Energy: Peak vs. Non-Peak")
        fig2, ax2 = plt.subplots(figsize=(8, 6))

        sns.barplot(
            data=df,
            x='Peak_Hour_Flag',
            y='Energy_kWh',
            palette='Set2',
            ax=ax2
        )
        ax2.set_xticklabels(['Non-Peak (0)', 'Peak (1)'])
        ax2.set_xlabel("Peak Hour Flags")
        ax2.set_ylabel("Energy (kWh)")
        st.pyplot(fig2)

    # Create Seaborn histplot based on the Charger Type data distributions across User_type
    with col3:
        st.subheader("Charger Type Distribution across users")
        fig3, ax3 = plt.subplots(figsize=(8, 6))

        sns.histplot(data=df, x="User_Type", hue="Charger_Type", kde=True)
        ax3.set_xlabel("UserType")
        st.pyplot(fig3)

    # Create a bigger and interactive scatterplot that user can change X , Y and Hue
    # This allows user to find other relationships between data
    # Drop column Session_ID since it's all unique data
    st.subheader("Interactive ScatterPlot")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    categorical_cols.remove("Session_ID")

    # columns to change x, y and hue
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("X-axis", numeric_cols, index=0)
    with col2:
        y_axis = st.selectbox("Y-axis", numeric_cols, index=1)
    with col3:
        hue_col = st.selectbox("Hue (Color by)", [None] + categorical_cols)

    # Create the Seaborn scatterplot using Matplotlib axes
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue_col, ax=ax)

    # Display the figure in Streamlit
    st.pyplot(fig)

# PAGE 3
elif page == "3. Live Machine Learning Inference":
    st.title("Live Machine Learning Inference")
    st.markdown("Adjust the parameters below to predict the real-time grid status.")

    # Divide the page into two columns: inputs on the left, results on the right
    col_input, col_result = st.columns([1, 1])

    # Column on left side (features)
    with col_input:
        # Features required for the prediction
        # ['Start_Hour', 'Duration_Mins', 'Energy_kWh', 'Overstay_Mins',
        #        'Battery_Temp_C', 'Peak_Hour_Flag', 'Charger_Type_AC',
        #        'Charger_Type_DC']


        st.subheader("⚙️ Input Parameters")

        # Create interactive input widgets
        # Some features have minimum and maximum allowed inputs, this is reflected on each widget
        start_hour = st.slider("Start Hour (0 - 24)", min_value=0, max_value=24, step=1)

        duration_mins = st.number_input("Duration (Minutes) [15.0 - 300.0]",
                                        min_value=15.0, max_value=300.0, value=15.0, step=0.01)

        energy_kwh = st.number_input("Energy Delivered (kWh) [5.0 - 120.0]",
                                     min_value=5.0, max_value=120.0, value=5.0, step=0.01)

        overstay_mins = st.number_input("Overstay (in minutes)[0 - 300]"
                                        , min_value=0, max_value=300, value=0, step=1)

        battery_temp = st.slider("Battery Temperature (°C) [24.0 - 70.0]",
                                 min_value=24.0, max_value=70.0, step=0.1)

        # Store peak_hour as 1 or 0
        peak_hour = st.selectbox("Is Peak Hour? (12PM - 5PM)", [0, 1],
                                 format_func=lambda x: "Yes" if x == 1 else "No")

        charger_type = st.selectbox("Charger Type", ["AC_Fast", "DC_Ultra"])

        # OneHotEnconding means that there are two features for Charger_type
        # Create 2 variables to store them depending on either AC or DC
        if charger_type == "AC_Fast":
            charger_ac = 1
            charger_dc = 0
        elif charger_type == "DC_Ultra":
            charger_ac = 0
            charger_dc = 1

        # Button to call predict()
        predict_btn = st.button("🚀 Predict Grid Status", use_container_width=True)

    # Column on the right side
    with col_result:
        st.subheader("📊 Prediction Result")

        # When clicked
        if predict_btn:
            # Convert all parameters into a pandas DataFrame that match the required features for the prediction
            input_data = pd.DataFrame([{
                'Start_Hour': start_hour,
                'Duration_Mins': duration_mins,
                'Energy_kWh': energy_kwh,
                'Overstay_Mins': overstay_mins,
                'Battery_Temp_C': battery_temp,
                'Peak_Hour_Flag': peak_hour,
                'Charger_Type_AC': charger_ac,
                'Charger_Type_DC': charger_dc
            }])

            # Make prediction using the loaded model
            prediction = model.predict(input_data)[0]

            # Display result with color-coded banners based on the prediction
            power_kw =round( energy_kwh / (duration_mins/ 60 ), 2)
            carbon_savings = round( energy_kwh* 0.475, 2)

            # Calculate power and carbon savings based on formula in the data_pipeline
            st.write(f"Based on the inputs, the calculated **Power** is: `{energy_kwh:.2f} kW`")
            st.write(f"**Carbon Savings** is: `{carbon_savings:.2f} kg C02`")

            st.divider()

            # From the y_pred
            # if model predicts that the inputs are `Normal`, display balloons on screen
            if prediction == 'Normal':
                st.success(
                    f"✅ **Predicted Status: {prediction}**\n\nThe grid is operating normally. "
                    f"No action required.")
                st.balloons()  # Balloon animation
            # if model predicts that the inputs are `Overheated`, create st.warning
            elif prediction == 'Overheated':
                st.warning(
                    f"⚠️ **Predicted Status: {prediction}**\n\nWarning: The battery is overheating. "
                    f"Please monitor the session.")
            # if model predicts that the inputs are `Grid_surge`, create st.error
            elif prediction == 'Grid_Surge':
                st.error(
                    f"🚨 **Predicted Status: {prediction}**\n\nDANGER: Grid surge detected! "
                    f"Immediate load balancing required.")
            # unexpected output here
            else:
                st.info(f"Predicted Status: {prediction}")
        else:
            st.info("👈 Please adjust the parameters on the left and click 'Predict Grid Status'.")


