# Streamlit App Code
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os  # Added to handle file operations

# Set up the Streamlit app
st.title("Capelli Sport Orders Analysis")
st.write("""
This app allows you to analyze and visualize Capelli Sport order data. You can explore trends over time, focus on specific clubs, and understand the dynamics of open orders over five weeks old.
""")

# Set visualization style
sns.set(style='whitegrid')

# Define the data directory
DATA_DIR = 'data'  # Ensure this directory exists in your GitHub repository

# Function to load data from the data directory
def load_data_from_directory(data_dir):
    if not os.path.exists(data_dir):
        st.error(f"The data directory '{data_dir}' does not exist. Please ensure it is present in your repository.")
        st.stop()

    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not data_files:
        st.error(f"No CSV files found in the '{data_dir}' directory. Please add the required data files.")
        st.stop()

    # Initialize an empty list to store DataFrames
    df_list = []
    report_dates_set = set()  # To collect valid report dates

    for filename in data_files:
        file_path = os.path.join(data_dir, filename)
        # Read each CSV file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            st.error(f"Error reading {filename}: {e}")
            continue

        # Extract the report date from the filename using regex
        # Adjust the regex to match your filenames
        match = re.search(r'Master Capelli Report Sheet - (\d{1,2}_\d{1,2}_\d{2}) Orders\.csv', filename)
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
            st.warning(f"Filename '{filename}' does not match expected pattern. Please ensure the filename matches 'Master Capelli Report Sheet - mm_dd_yy Orders.csv'.")
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
    return data, report_dates_set

# Load the data
data, report_dates_set = load_data_from_directory(DATA_DIR)

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

selected_club = st.sidebar.selectbox("Select Club", options=['All Clubs'] + list(clubs))
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

    # Analysis and Visualizations

    # Trend of Open Orders Over 5 Weeks Old
    trend_data = []
    for report_date in sorted(data_filtered['Report Date'].dropna().unique()):
        df_report = data_filtered[data_filtered['Report Date'] == report_date]

        # Number of open orders over 5 weeks old
        open_over_5_weeks = df_report[df_report['Order Category'] == 'Outstanding Over 5 Weeks']
        num_open_over_5_weeks = open_over_5_weeks['Order ID'].nunique()

        trend_data.append({
            'Report Date': report_date,
            'Open Orders Over 5 Weeks': num_open_over_5_weeks
        })

    trend_df = pd.DataFrame(trend_data)

    st.subheader("Trend of Open Orders Over 5 Weeks Old")
    st.write("This graph shows the number of open or partially shipped orders over five weeks old for the selected club and time period.")

    # Plot the trend
    plt.figure(figsize=(10, 6))
    plt.plot(trend_df['Report Date'], trend_df['Open Orders Over 5 Weeks'], marker='o')
    plt.title('Number of Open Orders Over 5 Weeks Old Over Time')
    plt.xlabel('Report Date')
    plt.ylabel('Number of Open Orders Over 5 Weeks')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

    # Analysis 1: Open or Partially Shipped Orders Becoming Over 5 Weeks Old Between Report Dates
    st.subheader("Open or Partially Shipped Orders Becoming Over 5 Weeks Old Between Report Dates")
    st.write("""
    This analysis shows the number of **open or partially shipped orders** that became over five weeks old between each report date for the selected club.
    """)

    # Create pivot tables for each report date
    pivot_tables = {}
    for report_date in sorted(data_filtered['Report Date'].dropna().unique()):
        df_report = data_filtered[data_filtered['Report Date'] == report_date]
        pivot_tables[report_date] = df_report[['Order ID', 'Order Category']].drop_duplicates().set_index('Order ID')

    # Sort report dates for chronological order
    sorted_report_dates = sorted(pivot_tables.keys())

    # Initialize DataFrame to store changes
    changes_list = []

    # Loop through the report dates to find orders that became over 5 weeks old
    for i in range(1, len(sorted_report_dates)):
        prev_date = sorted_report_dates[i-1]
        curr_date = sorted_report_dates[i]
        prev_pivot = pivot_tables[prev_date]
        curr_pivot = pivot_tables[curr_date]

        # Find orders that were not over 5 weeks old in prev_date but are over 5 weeks old in curr_date
        merged = prev_pivot.join(curr_pivot, lsuffix='_prev', rsuffix='_curr', how='outer')
        condition = (merged['Order Category_prev'] != 'Outstanding Over 5 Weeks') & \
                    (merged['Order Category_curr'] == 'Outstanding Over 5 Weeks')
        new_over_5_weeks_orders = merged[condition].reset_index()
        num_new_over_5_weeks_orders = new_over_5_weeks_orders['Order ID'].nunique()

        # Store the result
        changes_list.append({
            'From Date': prev_date,
            'To Date': curr_date,
            'New Orders Over 5 Weeks': num_new_over_5_weeks_orders
        })

        # Convert dates to strings
        prev_date_str = pd.Timestamp(prev_date).strftime('%Y-%m-%d')
        curr_date_str = pd.Timestamp(curr_date).strftime('%Y-%m-%d')

        st.write(f"From {prev_date_str} to {curr_date_str}, **{num_new_over_5_weeks_orders}** orders became over 5 weeks old.")

    # Analysis 2: Orders Shipped and New Orders Added Between Report Dates
    st.subheader("Orders Shipped and New Orders Added Between Report Dates")
    st.write("This analysis shows the number of orders shipped and new orders added between each report date for the selected club.")

    shipment_data = []

    for i in range(1, len(sorted_report_dates)):
        prev_date = sorted_report_dates[i-1]
        curr_date = sorted_report_dates[i]
        prev_df = data_filtered[data_filtered['Report Date'] == prev_date][['Order ID', 'Combined Order Status']].drop_duplicates()
        curr_df = data_filtered[data_filtered['Report Date'] == curr_date][['Order ID', 'Combined Order Status']].drop_duplicates()

        # Merge to compare statuses
        merged_status = prev_df.merge(curr_df, on='Order ID', how='outer', suffixes=('_prev', '_curr'))

        # Identify orders that changed from 'open' or 'partially shipped' to 'shipped'
        condition_shipped = merged_status['Combined Order Status_prev'].isin(['open', 'partially shipped']) & \
                            (merged_status['Combined Order Status_curr'] == 'shipped')
        orders_shipped = merged_status[condition_shipped]['Order ID'].nunique()

        # Identify new orders added in the current period
        new_orders = curr_df[~curr_df['Order ID'].isin(prev_df['Order ID'])]['Order ID'].nunique()

        shipment_data.append({
            'From Date': prev_date,
            'To Date': curr_date,
            'Orders Shipped': orders_shipped,
            'New Orders': new_orders
        })

        # Convert dates to strings
        prev_date_str = pd.Timestamp(prev_date).strftime('%Y-%m-%d')
        curr_date_str = pd.Timestamp(curr_date).strftime('%Y-%m-%d')

        st.write(f"From {prev_date_str} to {curr_date_str}, **{orders_shipped}** orders were shipped, and **{new_orders}** new orders were added.")

    # Combine shipment data into DataFrame
    shipment_df = pd.DataFrame(shipment_data)

    # Plot Orders Shipped and New Orders Over Time
    if not shipment_df.empty:
        st.subheader("Orders Shipped vs. New Orders Over Time")
        st.write("This graph compares the number of orders shipped and new orders added over the selected time period.")

        plt.figure(figsize=(10, 6))
        plt.plot(shipment_df['To Date'], shipment_df['Orders Shipped'], marker='o', label='Orders Shipped')
        plt.plot(shipment_df['To Date'], shipment_df['New Orders'], marker='o', label='New Orders')
        plt.title('Orders Shipped vs. New Orders Over Time')
        plt.xlabel('To Date')
        plt.ylabel('Number of Orders')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.write("Not enough data points to display Orders Shipped vs. New Orders.")

    # Top Clubs Contributing to New Over 5 Weeks Orders
    if selected_club == 'All Clubs':
        st.subheader("Top Clubs Contributing to New Over 5 Weeks Orders")
        st.write("This analysis shows the clubs contributing most to the number of new open or partially shipped orders over five weeks old.")

        club_contributions = {}
        for i in range(1, len(sorted_report_dates)):
            prev_date = sorted_report_dates[i-1]
            curr_date = sorted_report_dates[i]
            prev_df = data_filtered[data_filtered['Report Date'] == prev_date]
            curr_df = data_filtered[data_filtered['Report Date'] == curr_date]

            # Merge data on 'Order ID'
            merged = prev_df[['Order ID', 'Order Category', 'Club']].merge(
                curr_df[['Order ID', 'Order Category', 'Club']],
                on='Order ID', how='outer', suffixes=('_prev', '_curr')
            )

            # Find orders that became over 5 weeks old
            condition = (merged['Order Category_prev'] != 'Outstanding Over 5 Weeks') & \
                        (merged['Order Category_curr'] == 'Outstanding Over 5 Weeks')
            new_over_5_weeks_orders = merged[condition]

            # Count by club
            club_counts = new_over_5_weeks_orders['Club_curr'].value_counts()
            club_contributions[(prev_date, curr_date)] = club_counts

            prev_date_str = pd.Timestamp(prev_date).strftime('%Y-%m-%d')
            curr_date_str = pd.Timestamp(curr_date).strftime('%Y-%m-%d')

            st.write(f"\n**Top clubs contributing to new over 5 weeks old orders from {prev_date_str} to {curr_date_str}:**")
            st.write(club_counts.head())

        # Visualize top clubs in the last period
        if club_contributions:
            last_period = list(club_contributions.keys())[-1]
            club_counts_last_period = club_contributions[last_period]

            if not club_counts_last_period.empty:
                start_date_str = pd.Timestamp(last_period[0]).strftime('%Y-%m-%d')
                end_date_str = pd.Timestamp(last_period[1]).strftime('%Y-%m-%d')

                plt.figure(figsize=(12, 6))
                sns.barplot(x=club_counts_last_period.index[:10], y=club_counts_last_period.values[:10])
                plt.title(f'Top 10 Clubs Contributing to New Over 5 Weeks Orders from {start_date_str} to {end_date_str}')
                plt.xlabel('Club')
                plt.ylabel('Number of Orders')
                plt.xticks(rotation=90)
                plt.tight_layout()
                st.pyplot(plt)
    else:
        st.subheader(f"Detailed Analysis for {selected_club}")
        st.write(f"Here is a detailed analysis for **{selected_club}** based on the selected date range.")

        # Provide paragraph explanation for the selected club
        # We'll use the data from changes_list and shipment_data for this club

        # Summarize the findings
        explanation = ""

        # Current outstanding orders over 5 weeks old
        latest_date = data_filtered['Report Date'].max()
        latest_data = data_filtered[data_filtered['Report Date'] == latest_date]
        current_outstanding = latest_data[latest_data['Order Category'] == 'Outstanding Over 5 Weeks']['Order ID'].nunique()
        explanation += f"As of {latest_date.strftime('%Y-%m-%d')}, **{selected_club}** has **{current_outstanding}** outstanding orders over 5 weeks old.\n\n"

        # Trend over time
        trend_over_time = []
        for report_date in sorted_report_dates:
            df_report = data_filtered[data_filtered['Report Date'] == report_date]
            count = df_report[df_report['Order Category'] == 'Outstanding Over 5 Weeks']['Order ID'].nunique()
            trend_over_time.append({'Report Date': report_date, 'Outstanding Orders Over 5 Weeks': count})

        trend_over_time_df = pd.DataFrame(trend_over_time)

        # Interpret whether the situation is getting better or worse
        if trend_over_time_df['Outstanding Orders Over 5 Weeks'].is_monotonic_decreasing:
            explanation += f"The number of outstanding orders over 5 weeks old for **{selected_club}** is decreasing over time, indicating an improvement.\n\n"
        elif trend_over_time_df['Outstanding Orders Over 5 Weeks'].is_monotonic_increasing:
            explanation += f"The number of outstanding orders over 5 weeks old for **{selected_club}** is increasing over time, indicating a worsening situation.\n\n"
        else:
            explanation += f"The number of outstanding orders over 5 weeks old for **{selected_club}** fluctuates over time.\n\n"

        # Orders becoming over 5 weeks old
        explanation += f"Between {selected_start_date} and {selected_end_date}, {selected_club} had the following number of orders becoming over 5 weeks old:\n"
        for change in changes_list:
            from_date = pd.Timestamp(change['From Date']).strftime('%Y-%m-%d')
            to_date = pd.Timestamp(change['To Date']).strftime('%Y-%m-%d')
            num_orders = change['New Orders Over 5 Weeks']
            explanation += f"- From {from_date} to {to_date}: **{num_orders}** orders\n"

        # Orders shipped and new orders added
        explanation += f"\nDuring the same periods, the number of orders shipped and new orders added were as follows:\n"
        for shipment in shipment_data:
            from_date = pd.Timestamp(shipment['From Date']).strftime('%Y-%m-%d')
            to_date = pd.Timestamp(shipment['To Date']).strftime('%Y-%m-%d')
            shipped = shipment['Orders Shipped']
            new_orders = shipment['New Orders']
            explanation += f"- From {from_date} to {to_date}: **{shipped}** orders shipped, **{new_orders}** new orders added\n"

        st.write(explanation)

        # Plot the trend for the selected club
        st.subheader(f"Trend of Outstanding Orders Over 5 Weeks Old for {selected_club}")
        st.write("This graph shows how the number of outstanding orders over 5 weeks old has changed over time for the selected club.")

        plt.figure(figsize=(10, 6))
        plt.plot(trend_over_time_df['Report Date'], trend_over_time_df['Outstanding Orders Over 5 Weeks'], marker='o')
        plt.title(f'Outstanding Orders Over 5 Weeks Old Over Time for {selected_club}')
        plt.xlabel('Report Date')
        plt.ylabel('Number of Outstanding Orders')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)

        # Provide an interpretation paragraph
        st.write("**Interpretation:**")
        st.write(f"""
        The data indicates changes in the number of open or partially shipped orders over five weeks old for **{selected_club}**. By analyzing the trend and comparing the number of new overdue orders with the orders shipped and new orders added, we can assess the club's order processing efficiency. An increasing trend suggests a backlog forming, while a decreasing trend indicates progress in reducing overdue orders.
        """)

    # Age Distribution of Open Orders Over 5 Weeks
    st.subheader("Age Distribution of Open Orders Over 5 Weeks")
    st.write("This histogram displays the distribution of ages (in days) of open or partially shipped orders over five weeks old.")

    all_over_5_weeks = data_filtered[data_filtered['Order Category'] == 'Outstanding Over 5 Weeks'].copy()
    if not all_over_5_weeks.empty:
        all_over_5_weeks['Order Age (Days)'] = (all_over_5_weeks['Report Date'] - all_over_5_weeks['Order Date']).dt.days

        plt.figure(figsize=(10, 6))
        sns.histplot(all_over_5_weeks['Order Age (Days)'], bins=30, kde=True)
        plt.title('Age Distribution of Open Orders Over 5 Weeks')
        plt.xlabel('Order Age (Days)')
        plt.ylabel('Number of Orders')
        plt.tight_layout()
        st.pyplot(plt)
    else:
        st.write("No open orders over 5 weeks to display age distribution.")
