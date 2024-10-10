# Streamlit App Code
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os  # Import os module to work with file paths

# Set up the Streamlit app
st.title("Capelli Sport Orders Analysis")
st.write("""
This app allows you to analyze and visualize Capelli Sport order data. You can explore trends over time, focus on specific clubs, and understand the dynamics of open orders over five weeks old.
""")

# Set visualization style
sns.set(style='whitegrid')

# Define the data directory
DATA_DIR = 'data'

# Check if data directory exists
if os.path.exists(DATA_DIR):
    # List all CSV files in the data directory
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

    if data_files:
        # Initialize an empty list to store DataFrames
        df_list = []
        report_dates_set = set()  # To collect valid report dates

        for filename in data_files:
            file_path = os.path.join(DATA_DIR, filename)
            # Read each CSV file
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                st.error(f"Error reading {filename}: {e}")
                continue

            # Extract the report date from the filename using regex
            match = re.search(r'- (\d{1,2}_\d{1,2}_\d{2}) Orders\.csv', filename)
            if match:
                date_str = match.group(1)
                # Convert date string to datetime object
                try:
                    report_date = pd.to_datetime(date_str, format='%m_%d_%y')
                    report_dates_set.add(report_date)
                except ValueError:
                    st.warning(f"Filename '{filename}' contains an invalid date format. Please ensure the date is in 'mm_dd_yy' format.")
                    continue
            else:
                # If no date found, handle appropriately
                st.warning(f"Filename '{filename}' does not match expected pattern. Please ensure the filename includes the report date in the format '- mm_dd_yy Orders.csv'.")
                continue  # Skip this file

            # Add the extracted report date to the DataFrame
            df['Report Date'] = report_date

            # Append the DataFrame to the list
            df_list.append(df)

        if not df_list:
            st.error("No valid data loaded. Please check your data files in the 'data' directory.")
            st.stop()

        # Combine all DataFrames into one
        data = pd.concat(df_list, ignore_index=True)

        # Proceed with preprocessing as before

        # Convert date columns to datetime
        data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
        data['Report Date'] = pd.to_datetime(data['Report Date'], errors='coerce')

        # Ensure numeric columns are properly typed
        numeric_columns = ['Shipped Quantity', 'Unshipped Quantity']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Handle missing values
        data.fillna({'Shipped Quantity': 0, 'Unshipped Quantity': 0}, inplace=True)
        data['Combined Order Status'] = data['Combined Order Status'].fillna('Unknown')

        # Standardize text columns
        data['Order Status'] = data['Order Status'].astype(str).str.strip().str.lower()
        data['Combined Order Status'] = data['Combined Order Status'].astype(str).str.strip().str.lower()
        data['Club'] = data['Club'].astype(str).str.strip()

        # Define a function to categorize orders
        def categorize_order(row):
            if row['Combined Order Status'] in ['open', 'partially shipped']:
                if pd.notnull(row['Order Date']) and pd.notnull(row['Report Date']):
                    order_age = (row['Report Date'] - row['Order Date']).days
                    if order_age > 35:  # 5 weeks * 7 days
                        return 'Outstanding Over 5 Weeks'
                    else:
                        return 'Outstanding Under 5 Weeks'
                else:
                    return 'Other'
            else:
                return 'Other'

        # Apply the function to categorize orders
        data['Order Category'] = data.apply(categorize_order, axis=1)

        # Ensure 'Order ID' is of type string to prevent issues during merging
        data['Order ID'] = data['Order ID'].astype(str)

        # Get the list of clubs
        clubs = data['Club'].dropna().unique()
        clubs = sorted(clubs)

        # Get the list of report dates
        report_dates = sorted(report_dates_set)
        report_date_strings = [dt.strftime('%Y-%m-%d') for dt in report_dates]

        # Ensure report_date_strings is not empty
        if not report_date_strings:
            st.error("No valid report dates found after processing files. Please check your filenames and data.")
            st.stop()

        # User selection for club and time period
        st.sidebar.header("Filter Options")

        selected_club = st.sidebar.selectbox("Select Club", options=['All Clubs'] + clubs)
        selected_start_date = st.sidebar.selectbox("Select Start Date", options=report_date_strings, index=0)
        selected_end_date = st.sidebar.selectbox("Select End Date", options=report_date_strings, index=len(report_date_strings)-1)

        # Filter data based on user selection
        if selected_club != 'All Clubs':
            data_filtered = data[data['Club'] == selected_club]
        else:
            data_filtered = data.copy()

        # Convert selected dates back to datetime
        start_date = pd.to_datetime(selected_start_date)
        end_date = pd.to_datetime(selected_end_date)

        # Ensure start_date is before end_date
        if start_date > end_date:
            st.error("Start date must be before end date.")
            st.stop()
        else:
            # Filter data between selected dates
            data_filtered = data_filtered[(data_filtered['Report Date'] >= start_date) & (data_filtered['Report Date'] <= end_date)]

            if data_filtered.empty:
                st.warning("No data available for the selected club and date range.")
                st.stop()

            # (The rest of your analysis code goes here...)

    else:
        st.error("No CSV files found in the 'data' directory. Please add your report CSV files there.")
        st.stop()
else:
    st.error("Data directory not found. Please create a 'data' directory in your repository and add your report CSV files there.")
    st.stop()
