# overallanalysis.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import logging
from logging.handlers import RotatingFileHandler
import faulthandler
import tempfile
import plotly.express as px
import plotly.graph_objects as go

# Enable faulthandler to get tracebacks on segmentation faults
faulthandler.enable()

# -------------------------- Logging Setup -------------------------- #

# Define the path for the log file using a temporary directory for cross-platform compatibility
log_file_path = os.path.join(tempfile.gettempdir(), 'app_debug.log')

# Create a rotating file handler
handler = RotatingFileHandler(
    log_file_path,
    maxBytes=5*1024*1024,  # 5 MB
    backupCount=3,         # Keep up to 3 backup logs
    encoding='utf-8'
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Configure the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Optional: Add a console handler for real-time logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set to INFO or DEBUG as needed
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# -------------------------- Streamlit App Setup -------------------------- #

st.set_page_config(page_title="Capelli Sport Orders Analysis", layout="wide")

st.title("Capelli Sport Orders Analysis")
st.write("""
This app analyzes and visualizes Capelli Sport order data. Explore trends over time, focus on specific clubs, and understand the dynamics of open orders over five weeks old.
""")

# -------------------------- Column Mapping -------------------------- #

# Define a mapping from possible column names to standard names
COLUMN_MAPPING = {
    'Shipped Qty': 'Shipped Quantity',
    'Unshipped Qty': 'Unshipped Quantity',
    'Club Name': 'Club',
    # Add more mappings if there are other inconsistencies
}

# -------------------------- Text Cleaning Function -------------------------- #

def clean_text(text):
    """
    Cleans text by removing all types of whitespace and converting to lowercase.

    Parameters:
    - text (str): The text to clean.

    Returns:
    - str: The cleaned text.
    """
    if pd.isna(text):
        return ''
    # Remove all types of whitespace characters and lowercase the text
    return re.sub(r'\s+', ' ', str(text)).strip().lower()

# -------------------------- Data Loading and Processing Function -------------------------- #

@st.cache_data(show_spinner=False)
def load_data_from_directory(data_dir):
    """
    Loads and processes all CSV files from the specified directory.

    Parameters:
    - data_dir (str): Path to the directory containing CSV files.

    Returns:
    - pd.DataFrame: Combined and processed DataFrame.
    - set: Set of unique report dates.
    """
    logger.info(f"Loading data from directory: {data_dir}")
    if not os.path.exists(data_dir):
        st.error(f"The data directory '{data_dir}' does not exist. Please ensure it is present in your repository.")
        logger.error(f"Data directory '{data_dir}' does not exist.")
        st.stop()

    # Exclude hidden files and temporary .icloud files
    data_files = [f for f in os.listdir(data_dir)
                 if f.endswith('.csv') and not f.startswith('.') and not f.endswith('.icloud')]

    if not data_files:
        st.error(f"No valid CSV files found in the '{data_dir}' directory. Please add the required data files.")
        logger.error(f"No valid CSV files found in '{data_dir}'.")
        st.stop()

    # Initialize an empty list to store DataFrames
    df_list = []
    report_dates_set = set()  # To collect valid report dates
    skipped_files = []        # To track skipped files

    for filename in data_files:
        file_path = os.path.join(data_dir, filename)
        # Read each CSV file
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully read file: {filename}")
        except Exception as e:
            st.warning(f"Error reading {filename}: {e}")
            logger.error(f"Error reading {filename}: {e}")
            skipped_files.append(filename)
            continue

        # Extract the report date from the filename using regex
        match = re.search(r'Master\s+Capelli\s+Report\s+Sheet\s+-\s+(\d{1,2}_\d{1,2}_\d{2})\s+Orders\.csv', filename, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            logger.info(f"Extracted date string: {date_str} from {filename}")
            # Convert date string to datetime object
            try:
                report_date = pd.to_datetime(date_str, format='%m_%d_%y')
                report_dates_set.add(report_date)
                logger.info(f"Converted report date: {report_date.strftime('%Y-%m-%d')}")
            except ValueError as ve:
                st.warning(f"Filename '{filename}' contains an invalid date format. Please ensure the date is in 'mm_dd_yy' format.")
                logger.warning(f"Invalid date format in filename: {filename}")
                skipped_files.append(filename)
                continue
        else:
            # If no date found, handle appropriately
            st.warning(f"Filename '{filename}' does not match expected pattern. Please ensure the filename matches 'Master Capelli Report Sheet - mm_dd_yy Orders.csv'.")
            logger.warning(f"Filename pattern mismatch: {filename}")
            skipped_files.append(filename)
            continue  # Skip this file

        # Standardize column names
        df.rename(columns=COLUMN_MAPPING, inplace=True)
        logger.debug(f"Renamed columns for {filename}: {df.columns.tolist()}")

        # Ensure all column names are strings
        df.columns = df.columns.map(str)
        logger.debug(f"Converted column names to strings for {filename}: {df.columns.tolist()}")

        # Check for missing essential columns after renaming
        essential_columns = ['Order ID', 'Order Status', 'Order Date', 'Order Range', 'Club', 'Shipped Quantity', 'Unshipped Quantity', 'Combined Order Status']
        missing_columns = [col for col in essential_columns if col not in df.columns]
        if missing_columns:
            st.warning(f"Filename '{filename}' is missing columns after renaming: {missing_columns}. Please check the file structure.")
            logger.warning(f"Missing columns in {filename}: {missing_columns}")
            skipped_files.append(filename)
            continue  # Skip this file

        # Apply text cleaning to relevant columns
        text_columns = ['Order Status', 'Combined Order Status', 'Order Range']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)
                logger.debug(f"Standardized column '{col}' in {filename}: {df[col].unique()}")
            else:
                logger.warning(f"Column '{col}' not found in the data of {filename}.")

        # Add the extracted report date to the DataFrame
        df['Report Date'] = report_date

        # Append the DataFrame to the list
        df_list.append(df)
        logger.info(f"Appended data from {filename}")

    if skipped_files:
        st.warning(f"The following files were skipped due to errors or mismatched patterns: {', '.join(skipped_files)}")
        logger.warning(f"Skipped files: {skipped_files}")

    if not df_list:
        st.error("No valid data loaded. Please check your data files in the 'reports' directory.")
        logger.error("No valid data loaded after processing all files.")
        st.stop()

    # Combine all DataFrames into one
    data = pd.concat(df_list, ignore_index=True)
    logger.info("Successfully combined all data into a single DataFrame.")

    # Ensure all column names are strings to prevent mixed-type warnings
    data.columns = data.columns.map(str)
    logger.debug(f"All column names converted to strings: {data.columns.tolist()}")

    return data, report_dates_set

# -------------------------- Data Loading -------------------------- #

DATA_DIR = 'reports'  # Ensure this directory exists in your repository
data, report_dates_set = load_data_from_directory(DATA_DIR)

# -------------------------- Shipping Data Loading and Processing -------------------------- #

@st.cache_data(show_spinner=False)
def load_shipping_data(shipping_dir):
    """
    Loads and processes the shipping data from the specified directory.

    Parameters:
    - shipping_dir (str): Path to the directory containing shipping data files.

    Returns:
    - pd.DataFrame: Processed shipping data.
    """
    logger.info(f"Loading shipping data from directory: {shipping_dir}")
    if not os.path.exists(shipping_dir):
        st.error(f"The shipping data directory '{shipping_dir}' does not exist. Please ensure it is present.")
        logger.error(f"Shipping data directory '{shipping_dir}' does not exist.")
        return pd.DataFrame()

    # Exclude hidden files and temporary .icloud files
    shipping_files = [f for f in os.listdir(shipping_dir)
                      if f.endswith('.csv') and not f.startswith('.') and not f.endswith('.icloud')]

    if not shipping_files:
        st.error(f"No valid CSV files found in the '{shipping_dir}' directory. Please add the required data files.")
        logger.error(f"No valid CSV files found in '{shipping_dir}'.")
        return pd.DataFrame()

    # For simplicity, assuming there is only one relevant file
    shipping_file = shipping_files[0]
    file_path = os.path.join(shipping_dir, shipping_file)
    try:
        shipping_data = pd.read_csv(file_path)
        logger.info(f"Successfully read shipping data file: {shipping_file}")
    except Exception as e:
        st.warning(f"Error reading {shipping_file}: {e}")
        logger.error(f"Error reading {shipping_file}: {e}")
        return pd.DataFrame()

    return shipping_data

# Load the shipping data
SHIPPING_DIR = 'shippingdates'  # Ensure this directory exists
shipping_data = load_shipping_data(SHIPPING_DIR)

# -------------------------- Data Preprocessing -------------------------- #

# Identify and display rows with missing Report Date
missing_report_date = data[data['Report Date'].isna()]
num_missing = missing_report_date.shape[0]
st.write(f"Total rows with missing Report Date: {num_missing}")

if num_missing > 0:
    st.dataframe(missing_report_date)
    logger.warning(f"Found {num_missing} rows with missing Report Date.")

# Convert date columns to datetime
data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
data['Report Date'] = pd.to_datetime(data['Report Date'], errors='coerce')

# Check for any parsing errors
if data['Order Date'].isnull().any():
    st.warning("Some 'Order Date' entries could not be parsed and are set to NaT.")
    logger.warning("Some 'Order Date' entries could not be parsed.")

# Ensure numeric columns are properly typed
numeric_columns = ['Shipped Quantity', 'Unshipped Quantity']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Handle missing values
data.fillna({'Shipped Quantity': 0, 'Unshipped Quantity': 0}, inplace=True)
data['Combined Order Status'] = data['Combined Order Status'].fillna('unknown')

# Define a function to categorize orders based on 'Order Range' and 'Combined Order Status'
def categorize_order(row):
    order_range = row['Order Range'].strip().lower()
    combined_status = row['Combined Order Status'].strip().lower()
    if order_range == 'older than 5 weeks':
        if combined_status in ['open', 'partially shipped']:
            return 'Outstanding Over 5 Weeks'
    return 'Other'

# Apply the function to categorize orders
data['Order Category'] = data.apply(categorize_order, axis=1)

# Correct any remaining 'Oth' entries to 'Other'
oth_entries_final = data[data['Order Category'].str.lower() == 'oth']
if not oth_entries_final.empty:
    st.warning(f"Found {len(oth_entries_final)} entries with 'Order Category' as 'Oth'. Correcting them to 'Other'.")
    logger.warning(f"Found 'Oth' entries in data. Correcting to 'Other'.")
    data['Order Category'] = data['Order Category'].replace('oth', 'Other')

# Verify that 'Order Category' exists
if 'Order Category' not in data.columns:
    st.error("'Order Category' column is missing after preprocessing.")
    logger.error("'Order Category' column is missing after preprocessing.")
    st.stop()

# Optional: Display unique values in 'Order Category' for verification
unique_order_categories = data['Order Category'].unique()
logger.info(f"Unique Order Categories: {unique_order_categories}")
# st.write(f"Unique Order Categories: {unique_order_categories}")  # Uncomment if you want to display in the app

# Ensure 'Order ID' is of type string to prevent issues during merging
data['Order ID'] = data['Order ID'].astype(str)

# -------------------------- Merge Shipping Data into Orders Data -------------------------- #

if not shipping_data.empty:
    # Ensure necessary columns are present in shipping_data
    required_columns_shipping = ['Customer Reference', 'Club Name', 'Shipping Date', 'Date Created']
    if not all(col in shipping_data.columns for col in required_columns_shipping):
        st.error(f"The shipping data file must contain the following columns: {', '.join(required_columns_shipping)}")
        logger.error(f"Shipping data missing required columns: {required_columns_shipping}")
    else:
        # Convert 'Shipping Date' and 'Date Created' to datetime
        shipping_data['Shipping Date'] = pd.to_datetime(shipping_data['Shipping Date'], errors='coerce')
        shipping_data['Date Created'] = pd.to_datetime(shipping_data['Date Created'], errors='coerce')
        # Remove entries with invalid dates
        shipping_data = shipping_data.dropna(subset=['Shipping Date', 'Date Created'])

        # Remove duplicates of 'Customer Reference' by aggregating
        order_data = shipping_data.groupby('Customer Reference', as_index=False).agg({
            'Club Name': 'first',         # Assuming 'Club Name' is consistent within an order
            'Date Created': 'min',        # Earliest creation date among items
            'Shipping Date': 'max'        # Latest shipping date among items
        })

        # Compute time difference in days for each order
        order_data['Time Difference'] = (order_data['Shipping Date'] - order_data['Date Created']).dt.days

        # Assuming 'Customer Reference' corresponds to 'Order ID' in 'data'
        # Rename 'Customer Reference' to 'Order ID' for merging
        order_data.rename(columns={'Customer Reference': 'Order ID'}, inplace=True)

        # **Convert 'Order ID' in 'order_data' to string**
        order_data['Order ID'] = order_data['Order ID'].astype(str)

        # Merge 'order_data' into 'data' on 'Order ID'
        try:
            data = pd.merge(data, order_data[['Order ID', 'Time Difference']], on='Order ID', how='left')
            logger.info("Merged 'Time Difference' into main orders data.")
        except ValueError as ve:
            st.error(f"Merge Error: {ve}")
            logger.error(f"Merge Error: {ve}")
            st.stop()

        # Fill missing 'Time Difference' with 0 or appropriate value
        data['Time Difference'] = data['Time Difference'].fillna(0).astype(int)
else:
    st.warning("No shipping data available to merge with orders data.")
    logger.warning("Shipping data is empty or not loaded.")

# -------------------------- Get List of Clubs and Report Dates -------------------------- #

# Get the list of clubs
clubs = data['Club'].dropna().unique()
clubs = sorted(clubs)

# Get the list of report dates
report_dates = sorted(report_dates_set)
report_date_strings = [dt.strftime('%Y-%m-%d') for dt in report_dates]

# Ensure report_date_strings is not empty
if not report_date_strings:
    st.error("No valid report dates found after processing files. Please check your filenames and data.")
    logger.error("No valid report dates extracted.")
    st.stop()

# -------------------------- Define `sorted_report_dates` -------------------------- #

# **Critical Fix: Define `sorted_report_dates` BEFORE any usage**
sorted_report_dates = sorted(report_dates_set)
logger.info(f"Sorted report dates: {sorted_report_dates}")

# -------------------------- Sidebar Filters -------------------------- #

st.sidebar.header("Filter Options")

# Selection box for club
selected_club = st.sidebar.selectbox("Select Club", options=['All Clubs'] + list(clubs))

# -------------------------- Define Most Recent Report Date and Denominator Window -------------------------- #

# Identify the most recent report date
most_recent_report_date = data['Report Date'].max()
if pd.isna(most_recent_report_date):
    st.error("Could not determine the most recent report date. Please check your data.")
    logger.error("Most recent report date is NaT.")
    st.stop()
logger.info(f"Most recent report date identified: {most_recent_report_date.strftime('%Y-%m-%d')}")

# Define the start date for the denominator: from May 1 to 5 weeks ago based on 'Order Date'
start_date_denominator = pd.to_datetime('2024-05-01')  # Fixed start date as per user request
logger.info(f"Start date for denominator: {start_date_denominator.strftime('%Y-%m-%d')}")

# -------------------------- Aggregate Data -------------------------- #

# -------------------------- Compute Shipped Orders and Open Orders Percentage -------------------------- #

# Define May 1, 2024 as the start date
start_date_shipped = pd.to_datetime('2024-05-01')

# Filter shipped orders since May 1
# An order is considered shipped if 'Shipped Quantity' >0 and 'Order Date' <= 'Report Date'
shipped_data = data[(data['Order Date'] >= start_date_shipped) &
                    (data['Shipped Quantity'] > 0)]

# Group shipped orders by 'Order Date' and count unique 'Order ID's
shipped_per_day = shipped_data.groupby('Order Date')['Order ID'].nunique().reset_index(name='Shipped Orders')

# Sort shipped_per_day by 'Order Date' ascending
shipped_per_day = shipped_per_day.sort_values('Order Date')

# Compute cumulative shipped orders
shipped_per_day['Cumulative Shipped Orders'] = shipped_per_day['Shipped Orders'].cumsum()

# Create a DataFrame with all report dates sorted ascending
report_dates_sorted = sorted(report_dates_set)

# Create a DataFrame for report dates
report_dates_df = pd.DataFrame({'Report Date': report_dates_sorted})

# Merge shipped_per_day with report_dates_df using 'Order Date' and 'Report Date'
# For each report date, find the latest 'Order Date' <= 'Report Date' and get the corresponding 'Cumulative Shipped Orders'
ship_and_open_df = pd.merge_asof(report_dates_df.sort_values('Report Date'),
                                 shipped_per_day[['Order Date', 'Cumulative Shipped Orders']].sort_values('Order Date'),
                                 left_on='Report Date',
                                 right_on='Order Date',
                                 direction='backward')

# Fill NaN with 0 (if no shipments up to that report date)
ship_and_open_df['Cumulative Shipped Orders'] = ship_and_open_df['Cumulative Shipped Orders'].fillna(0).astype(int)

# Merge with total_aggregation
total_aggregation = data[data['Order Category'] == 'Outstanding Over 5 Weeks'] \
    .dropna(subset=['Report Date']) \
    .groupby('Report Date') \
    .size() \
    .reset_index(name='Total Open Orders Over 5 Weeks')

ship_and_open_df = pd.merge(total_aggregation, ship_and_open_df[['Report Date', 'Cumulative Shipped Orders']], on='Report Date', how='left')

# Calculate the percentage
ship_and_open_df['% Open Orders Over 5 Weeks'] = (ship_and_open_df['Total Open Orders Over 5 Weeks'] / ship_and_open_df['Cumulative Shipped Orders']) * 100

# Handle division by zero
ship_and_open_df['% Open Orders Over 5 Weeks'] = ship_and_open_df['% Open Orders Over 5 Weeks'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Sort by Report Date ascending
ship_and_open_df = ship_and_open_df.sort_values('Report Date')

# Format the table for display
display_ship_and_open = ship_and_open_df.copy()
display_ship_and_open['Report Date'] = display_ship_and_open['Report Date'].dt.strftime('%Y-%m-%d')
display_ship_and_open['Total Open Orders Over 5 Weeks'] = display_ship_and_open['Total Open Orders Over 5 Weeks'].astype(int)
display_ship_and_open['% Open Orders Over 5 Weeks'] = display_ship_and_open['% Open Orders Over 5 Weeks'].round(2)

# **Remove 'Cumulative Shipped Orders' column as per user request**
display_ship_and_open = display_ship_and_open[['Report Date', 'Total Open Orders Over 5 Weeks', '% Open Orders Over 5 Weeks']]

# Apply formatting
formatted_ship_and_open = display_ship_and_open.style.format({
    'Total Open Orders Over 5 Weeks': "{:,}",
    '% Open Orders Over 5 Weeks': "{:.2f}%"
})

# -------------------------- Display Shipped Orders and Open Orders Over 5 Weeks Old by Report Date -------------------------- #

st.subheader("Shipped Orders and Open Orders Over 5 Weeks Old by Report Date")
st.write("""
This table shows, for each report date, the **cumulative number of shipped orders since May 1**, the **total number of open orders over five weeks old**, and the **percentage of open orders over five weeks old**.
""")
st.dataframe(formatted_ship_and_open)

# -------------------------- Display Percentage of Open Orders Over 5 Weeks Old by Club and Report Date -------------------------- #

st.subheader("Percentage of Open Orders Over 5 Weeks Old by Club and Report Date")
st.write("""
This table shows the percentage of open orders over five weeks old for each club over time, along with the total number of such orders and the number of orders shipped.
""")

# Calculate total orders per club per report date
total_orders_per_club_date = data.groupby(['Club', 'Report Date'])['Order ID'].nunique().reset_index(name='Total Orders')

# Calculate 'Outstanding Over 5 Weeks' orders per club per report date
outstanding_orders_per_club_date = data[data['Order Category'] == 'Outstanding Over 5 Weeks'].groupby(['Club', 'Report Date'])['Order ID'].nunique().reset_index(name='Outstanding Over 5 Weeks')

# Merge total orders with outstanding orders
percentage_per_club_date = pd.merge(total_orders_per_club_date, outstanding_orders_per_club_date, on=['Club', 'Report Date'], how='left')

# Fill NaN values with 0
percentage_per_club_date['Outstanding Over 5 Weeks'] = percentage_per_club_date['Outstanding Over 5 Weeks'].fillna(0)

# Calculate percentage
percentage_per_club_date['Percentage Over 5 Weeks Old (%)'] = (percentage_per_club_date['Outstanding Over 5 Weeks'] / percentage_per_club_date['Total Orders']) * 100

# Round percentage to two decimal places
percentage_per_club_date['Percentage Over 5 Weeks Old (%)'] = percentage_per_club_date['Percentage Over 5 Weeks Old (%)'].round(2)

# Calculate shipped orders per club per report date
shipped_orders_per_club_date = data[data['Shipped Quantity'] > 0].groupby(['Club', 'Report Date'])['Order ID'].nunique().reset_index(name='Shipped Orders')

# Merge with the existing percentage_per_club_date
percentage_per_club_date = pd.merge(percentage_per_club_date, shipped_orders_per_club_date, on=['Club', 'Report Date'], how='left')

# Fill NaN shipped orders with 0
percentage_per_club_date['Shipped Orders'] = percentage_per_club_date['Shipped Orders'].fillna(0).astype(int)

# Select relevant columns
percentage_per_club_date = percentage_per_club_date[['Club', 'Report Date', 'Percentage Over 5 Weeks Old (%)', 'Outstanding Over 5 Weeks', 'Shipped Orders']]

# Rename columns for clarity
percentage_per_club_date.rename(columns={
    'Outstanding Over 5 Weeks': 'Total Open Orders Over 5 Weeks'
}, inplace=True)

# Pivot the table to have Report Dates as columns with both percentage and total orders
percentage_pivot = percentage_per_club_date.pivot(index='Club', columns='Report Date', values=['Percentage Over 5 Weeks Old (%)', 'Total Open Orders Over 5 Weeks', 'Shipped Orders'])

# Flatten the MultiIndex columns
percentage_pivot.columns = [f"{date.strftime('%Y-%m-%d')} {metric}" for metric, date in percentage_pivot.columns]

# Reset index to turn 'Club' back into a column
percentage_pivot.reset_index(inplace=True)

# Sort the pivot table by Club name
percentage_pivot = percentage_pivot.sort_values('Club')

# Convert report date columns to string format for easier formatting
percentage_pivot.columns = ['Club'] + [col for col in percentage_pivot.columns if col != 'Club']

# Reorder columns to have 'Club' first, followed by sorted report dates with metrics
sorted_report_dates_strings = sorted(report_date_strings)

# Create a list for new column order with alternating Percentage, Total, and Shipped
new_column_order = ['Club']
for date in sorted_report_dates_strings:
    pct_col = f"{date} Percentage Over 5 Weeks Old (%)"
    total_col = f"{date} Total Open Orders Over 5 Weeks"
    shipped_col = f"{date} Shipped Orders"
    # Rename columns to have 'Percentage', 'Total', and 'Shipped'
    if pct_col in percentage_pivot.columns and total_col in percentage_pivot.columns and shipped_col in percentage_pivot.columns:
        percentage_pivot.rename(columns={
            pct_col: f"{date} %",
            total_col: f"{date} Total",
            shipped_col: f"{date} Shipped"
        }, inplace=True)
        new_column_order.extend([f"{date} %", f"{date} Total", f"{date} Shipped"])
    elif pct_col in percentage_pivot.columns and total_col not in percentage_pivot.columns and shipped_col not in percentage_pivot.columns:
        percentage_pivot.rename(columns={
            pct_col: f"{date} %"
        }, inplace=True)
        new_column_order.append(f"{date} %")
    elif pct_col not in percentage_pivot.columns and total_col in percentage_pivot.columns and shipped_col not in percentage_pivot.columns:
        percentage_pivot.rename(columns={
            total_col: f"{date} Total"
        }, inplace=True)
        new_column_order.append(f"{date} Total")
    elif pct_col not in percentage_pivot.columns and total_col not in percentage_pivot.columns and shipped_col in percentage_pivot.columns:
        percentage_pivot.rename(columns={
            shipped_col: f"{date} Shipped"
        }, inplace=True)
        new_column_order.append(f"{date} Shipped")
    # If none exist, skip

# Ensure all columns in new_column_order exist
existing_columns = [col for col in new_column_order if col in percentage_pivot.columns]
percentage_pivot = percentage_pivot[existing_columns]

# **Round all numerical columns to two decimal places as per user request**
# Identify numerical columns (all except 'Club')
numerical_cols_percentage_table = percentage_pivot.columns.tolist()
numerical_cols_percentage_table.remove('Club')

# Apply rounding
percentage_pivot[numerical_cols_percentage_table] = percentage_pivot[numerical_cols_percentage_table].round(2)

# Apply formatting using Pandas Styler
styled_percentage_pivot = percentage_pivot.style.format({
    col: "{:.2f}" for col in numerical_cols_percentage_table
})

# Display the styled percentage pivot table
st.write(styled_percentage_pivot)

# -------------------------- Plot Total Open Orders Over 5 Weeks Old -------------------------- #
st.subheader("Trend of Total Open Orders Over 5 Weeks Old")
st.write("This graph shows the total number of open orders over five weeks old across all clubs for each report date.")

# Plot using Plotly
fig_total_trend = px.line(
    ship_and_open_df,
    x='Report Date',
    y='Total Open Orders Over 5 Weeks',
    markers=True,
    title='Total Open Orders Over 5 Weeks Old Over Time',
    labels={'Report Date': 'Report Date', 'Total Open Orders Over 5 Weeks': 'Total Open Orders Over 5 Weeks'}
)

fig_total_trend.update_layout(
    xaxis=dict(tickangle=45),
    yaxis=dict(title='Total Open Orders Over 5 Weeks'),
    template='plotly_white'
)

st.plotly_chart(fig_total_trend, use_container_width=True)
logger.info("Plotted Trend of Total Open Orders Over 5 Weeks Old using Plotly")

# -------------------------- Orders Shipped Over Time -------------------------- #

st.subheader("Orders Shipped Over Time")
st.write("""
This graph shows the number of orders shipped over time based on shipping dates, grouped by week.
""")

if not shipping_data.empty and all(col in shipping_data.columns for col in ['Customer Reference', 'Club Name', 'Shipping Date', 'Date Created']):
    # Convert 'Shipping Date' and 'Date Created' to datetime
    shipping_data['Shipping Date'] = pd.to_datetime(shipping_data['Shipping Date'], errors='coerce')
    shipping_data['Date Created'] = pd.to_datetime(shipping_data['Date Created'], errors='coerce')
    # Remove entries with invalid dates
    shipping_data = shipping_data.dropna(subset=['Shipping Date', 'Date Created'])

    # Remove duplicates of 'Customer Reference' by aggregating
    order_data_shipping = shipping_data.groupby('Customer Reference', as_index=False).agg({
        'Club Name': 'first',         # Assuming 'Club Name' is consistent within an order
        'Date Created': 'min',        # Earliest creation date among items
        'Shipping Date': 'max'        # Latest shipping date among items
    })

    # Compute time difference in days for each order
    order_data_shipping['Time Difference'] = (order_data_shipping['Shipping Date'] - order_data_shipping['Date Created']).dt.days

    # Set 'Shipping Date' as index
    order_data_shipping.set_index('Shipping Date', inplace=True)

    # Group by week and count unique orders
    orders_over_time_shipping = order_data_shipping.resample('W').size().reset_index(name='Unique Orders')

    # Sort by 'Shipping Date'
    orders_over_time_shipping = orders_over_time_shipping.sort_values('Shipping Date')

    # Plot the number of orders over time
    fig_orders_over_time = px.line(
        orders_over_time_shipping,
        x='Shipping Date',
        y='Unique Orders',
        markers=True,
        title='Orders Shipped Over Time (Weekly)',
        labels={'Shipping Date': 'Week Starting', 'Unique Orders': 'Number of Orders'}
    )
    fig_orders_over_time.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(title='Number of Orders'),
        template='plotly_white'
    )
    st.plotly_chart(fig_orders_over_time, use_container_width=True)
    logger.info("Plotted Orders Shipped Over Time (Weekly) using Plotly")
else:
    st.write("No shipping data available to display orders shipped over time.")
    logger.warning("Shipping data is empty or missing required columns.")

# -------------------------- Shipping Time for Each Order Over Time -------------------------- #

st.subheader("Shipping Time for Each Order Over Time")
st.write("""
This graph shows how long it took for each order to ship, based on the shipping date over time.
""")

if not shipping_data.empty and all(col in shipping_data.columns for col in ['Customer Reference', 'Club Name', 'Shipping Date', 'Date Created']):
    # Reset index to access 'Shipping Date' as a column
    order_data_shipping.reset_index(inplace=True)

    # Compute shipping times
    shipping_times = order_data_shipping[['Shipping Date', 'Time Difference', 'Customer Reference']]  # Added 'Customer Reference'

    # Plot using Plotly
    fig_shipping_time = px.scatter(
        shipping_times,
        x='Shipping Date',
        y='Time Difference',
        title='Shipping Time for Each Order Over Time',
        labels={'Shipping Date': 'Shipping Date', 'Time Difference': 'Shipping Time (days)'},
        opacity=0.6,
        hover_data=['Time Difference', 'Customer Reference']  # Included 'Customer Reference' in hover data
    )

    # Update layout for better readability
    fig_shipping_time.update_layout(
        template='plotly_white',
        xaxis=dict(tickangle=45),
        yaxis=dict(title='Shipping Time (days)')
    )

    st.plotly_chart(fig_shipping_time, use_container_width=True)
    logger.info("Plotted Shipping Time for Each Order Over Time using Plotly")
else:
    st.write("No shipping data available to display shipping time graph.")
    logger.warning("Shipping data is empty or missing required columns for shipping time graph.")

# -------------------------- Average Shipping Time per Month -------------------------- #

st.subheader("Average Shipping Time per Month")
st.write("""
This bar chart displays the **average shipping time (in days)** for each month.
""")

if not shipping_data.empty and all(col in shipping_data.columns for col in ['Customer Reference', 'Club Name', 'Shipping Date', 'Date Created']):
    # Reset index to access 'Shipping Date' as a column
    order_data_shipping.reset_index(inplace=True)

    # Extract month and year for grouping
    order_data_shipping['Shipping Month'] = order_data_shipping['Shipping Date'].dt.to_period('M').dt.to_timestamp()

    # Compute average shipping time per month
    avg_shipping_time_per_month = order_data_shipping.groupby('Shipping Month')['Time Difference'].mean().reset_index()

    # Sort by Shipping Month
    avg_shipping_time_per_month = avg_shipping_time_per_month.sort_values('Shipping Month')

    # Plot the average shipping time per month as a bar chart
    fig_avg_shipping = px.bar(
        avg_shipping_time_per_month,
        x='Shipping Month',
        y='Time Difference',
        title='Average Shipping Time per Month',
        labels={'Shipping Month': 'Month', 'Time Difference': 'Average Shipping Time (days)'},
        text='Time Difference'
    )
    fig_avg_shipping.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_avg_shipping.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(title='Average Shipping Time (days)'),
        template='plotly_white'
    )
    st.plotly_chart(fig_avg_shipping, use_container_width=True)
    logger.info("Plotted Average Shipping Time per Month as a Bar Chart using Plotly")
else:
    st.write("No shipping data available to display the average shipping time per month.")
    logger.warning("Shipping data is empty or missing required columns for average shipping time bar chart.")

# -------------------------- Average Order Time by Club -------------------------- #

st.subheader("Average Order Time by Club")
st.write("""
This table displays the **average shipping time (in days)** for each club, along with the **number of orders shipped by each club**.
""")

# Compute average shipping time per club and number of orders shipped
average_order_time = data.groupby('Club').agg({
    'Time Difference': 'mean',
    'Order ID': 'count'
}).reset_index()
average_order_time['Time Difference'] = average_order_time['Time Difference'].round(2)
average_order_time.rename(columns={'Order ID': 'Number of Orders Shipped'}, inplace=True)

# **Round all numerical columns to two decimal places as per user request**
average_order_time['Number of Orders Shipped'] = average_order_time['Number of Orders Shipped'].astype(float).round(2)

# Display the table
st.dataframe(average_order_time.style.format({
    'Club': lambda x: x.title(),
    'Time Difference': "{:.2f} days",
    'Number of Orders Shipped': "{:.2f}"
}))

# Plot Average Order Time by Club
fig_avg_order_time_club = px.bar(
    average_order_time,
    x='Club',
    y='Time Difference',
    title='Average Shipping Time by Club',
    labels={'Club': 'Club', 'Time Difference': 'Average Shipping Time (days)'},
    text='Time Difference'
)
fig_avg_order_time_club.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_avg_order_time_club.update_layout(
    xaxis=dict(tickangle=45),
    yaxis=dict(title='Average Shipping Time (days)'),
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    template='plotly_white'
)
st.plotly_chart(fig_avg_order_time_club, use_container_width=True)
logger.info("Plotted Average Order Time by Club using Plotly")

# -------------------------- Final Touches -------------------------- #

# You can add additional sections or features as needed.

# -------------------------- End of File -------------------------- #
# # overallanalysis.py
#
# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import os
# import logging
# from logging.handlers import RotatingFileHandler
# import faulthandler
# import tempfile
# import plotly.express as px
# import plotly.graph_objects as go
#
# # Enable faulthandler to get tracebacks on segmentation faults
# faulthandler.enable()
#
# # -------------------------- Logging Setup -------------------------- #
#
# # Define the path for the log file using a temporary directory for cross-platform compatibility
# log_file_path = os.path.join(tempfile.gettempdir(), 'app_debug.log')
#
# # Create a rotating file handler
# handler = RotatingFileHandler(
#     log_file_path,
#     maxBytes=5*1024*1024,  # 5 MB
#     backupCount=3,         # Keep up to 3 backup logs
#     encoding='utf-8'
# )
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
#
# # Configure the root logger
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logger.addHandler(handler)
#
# # Optional: Add a console handler for real-time logging
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)  # Set to INFO or DEBUG as needed
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)
#
# # -------------------------- Streamlit App Setup -------------------------- #
#
# st.set_page_config(page_title="Capelli Sport Orders Analysis", layout="wide")
#
# st.title("Capelli Sport Orders Analysis")
# st.write("""
# This app analyzes and visualizes Capelli Sport order data. Explore trends over time, focus on specific clubs, and understand the dynamics of open orders over five weeks old.
# """)
#
# # -------------------------- Column Mapping -------------------------- #
#
# # Define a mapping from possible column names to standard names
# COLUMN_MAPPING = {
#     'Shipped Qty': 'Shipped Quantity',
#     'Unshipped Qty': 'Unshipped Quantity',
#     'Club Name': 'Club',
#     # Add more mappings if there are other inconsistencies
# }
#
# # -------------------------- Text Cleaning Function -------------------------- #
#
# def clean_text(text):
#     """
#     Cleans text by removing all types of whitespace and converting to lowercase.
#
#     Parameters:
#     - text (str): The text to clean.
#
#     Returns:
#     - str: The cleaned text.
#     """
#     if pd.isna(text):
#         return ''
#     # Remove all types of whitespace characters and lowercase the text
#     return re.sub(r'\s+', ' ', str(text)).strip().lower()
#
# # -------------------------- Data Loading and Processing Function -------------------------- #
#
# @st.cache_data(show_spinner=False)
# def load_data_from_directory(data_dir):
#     """
#     Loads and processes all CSV files from the specified directory.
#
#     Parameters:
#     - data_dir (str): Path to the directory containing CSV files.
#
#     Returns:
#     - pd.DataFrame: Combined and processed DataFrame.
#     - set: Set of unique report dates.
#     """
#     logger.info(f"Loading data from directory: {data_dir}")
#     if not os.path.exists(data_dir):
#         st.error(f"The data directory '{data_dir}' does not exist. Please ensure it is present in your repository.")
#         logger.error(f"Data directory '{data_dir}' does not exist.")
#         st.stop()
#
#     # Exclude hidden files and temporary .icloud files
#     data_files = [f for f in os.listdir(data_dir)
#                  if f.endswith('.csv') and not f.startswith('.') and not f.endswith('.icloud')]
#
#     if not data_files:
#         st.error(f"No valid CSV files found in the '{data_dir}' directory. Please add the required data files.")
#         logger.error(f"No valid CSV files found in '{data_dir}'.")
#         st.stop()
#
#     # Initialize an empty list to store DataFrames
#     df_list = []
#     report_dates_set = set()  # To collect valid report dates
#     skipped_files = []        # To track skipped files
#
#     for filename in data_files:
#         file_path = os.path.join(data_dir, filename)
#         # Read each CSV file
#         try:
#             df = pd.read_csv(file_path)
#             logger.info(f"Successfully read file: {filename}")
#         except Exception as e:
#             st.warning(f"Error reading {filename}: {e}")
#             logger.error(f"Error reading {filename}: {e}")
#             skipped_files.append(filename)
#             continue
#
#         # Extract the report date from the filename using regex
#         match = re.search(r'Master\s+Capelli\s+Report\s+Sheet\s+-\s+(\d{1,2}_\d{1,2}_\d{2})\s+Orders\.csv', filename, re.IGNORECASE)
#         if match:
#             date_str = match.group(1)
#             logger.info(f"Extracted date string: {date_str} from {filename}")
#             # Convert date string to datetime object
#             try:
#                 report_date = pd.to_datetime(date_str, format='%m_%d_%y')
#                 report_dates_set.add(report_date)
#                 logger.info(f"Converted report date: {report_date.strftime('%Y-%m-%d')}")
#             except ValueError as ve:
#                 st.warning(f"Filename '{filename}' contains an invalid date format. Please ensure the date is in 'mm_dd_yy' format.")
#                 logger.warning(f"Invalid date format in filename: {filename}")
#                 skipped_files.append(filename)
#                 continue
#         else:
#             # If no date found, handle appropriately
#             st.warning(f"Filename '{filename}' does not match expected pattern. Please ensure the filename matches 'Master Capelli Report Sheet - mm_dd_yy Orders.csv'.")
#             logger.warning(f"Filename pattern mismatch: {filename}")
#             skipped_files.append(filename)
#             continue  # Skip this file
#
#         # Standardize column names
#         df.rename(columns=COLUMN_MAPPING, inplace=True)
#         logger.debug(f"Renamed columns for {filename}: {df.columns.tolist()}")
#
#         # Ensure all column names are strings
#         df.columns = df.columns.map(str)
#         logger.debug(f"Converted column names to strings for {filename}: {df.columns.tolist()}")
#
#         # Check for missing essential columns after renaming
#         essential_columns = ['Order ID', 'Order Status', 'Order Date', 'Order Range', 'Club', 'Shipped Quantity', 'Unshipped Quantity', 'Combined Order Status']
#         missing_columns = [col for col in essential_columns if col not in df.columns]
#         if missing_columns:
#             st.warning(f"Filename '{filename}' is missing columns after renaming: {missing_columns}. Please check the file structure.")
#             logger.warning(f"Missing columns in {filename}: {missing_columns}")
#             skipped_files.append(filename)
#             continue  # Skip this file
#
#         # Apply text cleaning to relevant columns
#         text_columns = ['Order Status', 'Combined Order Status', 'Order Range']
#         for col in text_columns:
#             if col in df.columns:
#                 df[col] = df[col].apply(clean_text)
#                 logger.debug(f"Standardized column '{col}' in {filename}: {df[col].unique()}")
#             else:
#                 logger.warning(f"Column '{col}' not found in the data of {filename}.")
#
#         # Add the extracted report date to the DataFrame
#         df['Report Date'] = report_date
#
#         # Append the DataFrame to the list
#         df_list.append(df)
#         logger.info(f"Appended data from {filename}")
#
#     if skipped_files:
#         st.warning(f"The following files were skipped due to errors or mismatched patterns: {', '.join(skipped_files)}")
#         logger.warning(f"Skipped files: {skipped_files}")
#
#     if not df_list:
#         st.error("No valid data loaded. Please check your data files in the 'reports' directory.")
#         logger.error("No valid data loaded after processing all files.")
#         st.stop()
#
#     # Combine all DataFrames into one
#     data = pd.concat(df_list, ignore_index=True)
#     logger.info("Successfully combined all data into a single DataFrame.")
#
#     # Ensure all column names are strings to prevent mixed-type warnings
#     data.columns = data.columns.map(str)
#     logger.debug(f"All column names converted to strings: {data.columns.tolist()}")
#
#     return data, report_dates_set
#
# # -------------------------- Data Loading -------------------------- #
#
# DATA_DIR = 'reports'  # Ensure this directory exists in your repository
# data, report_dates_set = load_data_from_directory(DATA_DIR)
#
# # -------------------------- Shipping Data Loading and Processing -------------------------- #
#
# @st.cache_data(show_spinner=False)
# def load_shipping_data(shipping_dir):
#     """
#     Loads and processes the shipping data from the specified directory.
#
#     Parameters:
#     - shipping_dir (str): Path to the directory containing shipping data files.
#
#     Returns:
#     - pd.DataFrame: Processed shipping data.
#     """
#     logger.info(f"Loading shipping data from directory: {shipping_dir}")
#     if not os.path.exists(shipping_dir):
#         st.error(f"The shipping data directory '{shipping_dir}' does not exist. Please ensure it is present.")
#         logger.error(f"Shipping data directory '{shipping_dir}' does not exist.")
#         return pd.DataFrame()
#
#     # Exclude hidden files and temporary .icloud files
#     shipping_files = [f for f in os.listdir(shipping_dir)
#                       if f.endswith('.csv') and not f.startswith('.') and not f.endswith('.icloud')]
#
#     if not shipping_files:
#         st.error(f"No valid CSV files found in the '{shipping_dir}' directory. Please add the required data files.")
#         logger.error(f"No valid CSV files found in '{shipping_dir}'.")
#         return pd.DataFrame()
#
#     # For simplicity, assuming there is only one relevant file
#     shipping_file = shipping_files[0]
#     file_path = os.path.join(shipping_dir, shipping_file)
#     try:
#         shipping_data = pd.read_csv(file_path)
#         logger.info(f"Successfully read shipping data file: {shipping_file}")
#     except Exception as e:
#         st.warning(f"Error reading {shipping_file}: {e}")
#         logger.error(f"Error reading {shipping_file}: {e}")
#         return pd.DataFrame()
#
#     return shipping_data
#
# # Load the shipping data
# SHIPPING_DIR = 'shippingdates'  # Ensure this directory exists
# shipping_data = load_shipping_data(SHIPPING_DIR)
#
# # -------------------------- Data Preprocessing -------------------------- #
#
# # Identify and display rows with missing Report Date
# missing_report_date = data[data['Report Date'].isna()]
# num_missing = missing_report_date.shape[0]
# st.write(f"Total rows with missing Report Date: {num_missing}")
#
# if num_missing > 0:
#     st.dataframe(missing_report_date)
#     logger.warning(f"Found {num_missing} rows with missing Report Date.")
#
# # Convert date columns to datetime
# data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
# data['Report Date'] = pd.to_datetime(data['Report Date'], errors='coerce')
#
# # Check for any parsing errors
# if data['Order Date'].isnull().any():
#     st.warning("Some 'Order Date' entries could not be parsed and are set to NaT.")
#     logger.warning("Some 'Order Date' entries could not be parsed.")
#
# # Ensure numeric columns are properly typed
# numeric_columns = ['Shipped Quantity', 'Unshipped Quantity']
# for col in numeric_columns:
#     data[col] = pd.to_numeric(data[col], errors='coerce')
#
# # Handle missing values
# data.fillna({'Shipped Quantity': 0, 'Unshipped Quantity': 0}, inplace=True)
# data['Combined Order Status'] = data['Combined Order Status'].fillna('unknown')
#
# # Define a function to categorize orders based on 'Order Range' and 'Combined Order Status'
# def categorize_order(row):
#     order_range = row['Order Range'].strip().lower()
#     combined_status = row['Combined Order Status'].strip().lower()
#     if order_range == 'older than 5 weeks':
#         if combined_status in ['open', 'partially shipped']:
#             return 'Outstanding Over 5 Weeks'
#     return 'Other'
#
# # Apply the function to categorize orders
# data['Order Category'] = data.apply(categorize_order, axis=1)
#
# # Correct any remaining 'Oth' entries to 'Other'
# oth_entries_final = data[data['Order Category'].str.lower() == 'oth']
# if not oth_entries_final.empty:
#     st.warning(f"Found {len(oth_entries_final)} entries with 'Order Category' as 'Oth'. Correcting them to 'Other'.")
#     logger.warning(f"Found 'Oth' entries in data. Correcting to 'Other'.")
#     data['Order Category'] = data['Order Category'].replace('oth', 'Other')
#
# # Verify that 'Order Category' exists
# if 'Order Category' not in data.columns:
#     st.error("'Order Category' column is missing after preprocessing.")
#     logger.error("'Order Category' column is missing after preprocessing.")
#     st.stop()
#
# # Optional: Display unique values in 'Order Category' for verification
# unique_order_categories = data['Order Category'].unique()
# logger.info(f"Unique Order Categories: {unique_order_categories}")
# # st.write(f"Unique Order Categories: {unique_order_categories}")  # Uncomment if you want to display in the app
#
# # Ensure 'Order ID' is of type string to prevent issues during merging
# data['Order ID'] = data['Order ID'].astype(str)
#
# # -------------------------- Merge Shipping Data into Orders Data -------------------------- #
#
# if not shipping_data.empty:
#     # Ensure necessary columns are present in shipping_data
#     required_columns_shipping = ['Customer Reference', 'Club Name', 'Shipping Date', 'Date Created']
#     if not all(col in shipping_data.columns for col in required_columns_shipping):
#         st.error(f"The shipping data file must contain the following columns: {', '.join(required_columns_shipping)}")
#         logger.error(f"Shipping data missing required columns: {required_columns_shipping}")
#     else:
#         # Convert 'Shipping Date' and 'Date Created' to datetime
#         shipping_data['Shipping Date'] = pd.to_datetime(shipping_data['Shipping Date'], errors='coerce')
#         shipping_data['Date Created'] = pd.to_datetime(shipping_data['Date Created'], errors='coerce')
#         # Remove entries with invalid dates
#         shipping_data = shipping_data.dropna(subset=['Shipping Date', 'Date Created'])
#
#         # Remove duplicates of 'Customer Reference' by aggregating
#         order_data = shipping_data.groupby('Customer Reference', as_index=False).agg({
#             'Club Name': 'first',         # Assuming 'Club Name' is consistent within an order
#             'Date Created': 'min',        # Earliest creation date among items
#             'Shipping Date': 'max'        # Latest shipping date among items
#         })
#
#         # Compute time difference in days for each order
#         order_data['Time Difference'] = (order_data['Shipping Date'] - order_data['Date Created']).dt.days
#
#         # Assuming 'Customer Reference' corresponds to 'Order ID' in 'data'
#         # Rename 'Customer Reference' to 'Order ID' for merging
#         order_data.rename(columns={'Customer Reference': 'Order ID'}, inplace=True)
#
#         # **Convert 'Order ID' in 'order_data' to string**
#         order_data['Order ID'] = order_data['Order ID'].astype(str)
#
#         # Merge 'order_data' into 'data' on 'Order ID'
#         try:
#             data = pd.merge(data, order_data[['Order ID', 'Time Difference']], on='Order ID', how='left')
#             logger.info("Merged 'Time Difference' into main orders data.")
#         except ValueError as ve:
#             st.error(f"Merge Error: {ve}")
#             logger.error(f"Merge Error: {ve}")
#             st.stop()
#
#         # Fill missing 'Time Difference' with 0 or appropriate value
#         data['Time Difference'] = data['Time Difference'].fillna(0).astype(int)
# else:
#     st.warning("No shipping data available to merge with orders data.")
#     logger.warning("Shipping data is empty or not loaded.")
#
# # -------------------------- Get List of Clubs and Report Dates -------------------------- #
#
# # Get the list of clubs
# clubs = data['Club'].dropna().unique()
# clubs = sorted(clubs)
#
# # Get the list of report dates
# report_dates = sorted(report_dates_set)
# report_date_strings = [dt.strftime('%Y-%m-%d') for dt in report_dates]
#
# # Ensure report_date_strings is not empty
# if not report_date_strings:
#     st.error("No valid report dates found after processing files. Please check your filenames and data.")
#     logger.error("No valid report dates extracted.")
#     st.stop()
#
# # -------------------------- Define `sorted_report_dates` -------------------------- #
#
# # **Critical Fix: Define `sorted_report_dates` BEFORE any usage**
# sorted_report_dates = sorted(report_dates_set)
# logger.info(f"Sorted report dates: {sorted_report_dates}")
#
# # -------------------------- Sidebar Filters -------------------------- #
#
# st.sidebar.header("Filter Options")
#
# # Selection box for club
# selected_club = st.sidebar.selectbox("Select Club", options=['All Clubs'] + list(clubs))
#
# # -------------------------- Define Most Recent Report Date and Denominator Window -------------------------- #
#
# # Identify the most recent report date
# most_recent_report_date = data['Report Date'].max()
# if pd.isna(most_recent_report_date):
#     st.error("Could not determine the most recent report date. Please check your data.")
#     logger.error("Most recent report date is NaT.")
#     st.stop()
# logger.info(f"Most recent report date identified: {most_recent_report_date.strftime('%Y-%m-%d')}")
#
# # Define the start date for the denominator: from May 1 to 5 weeks ago based on 'Order Date'
# start_date_denominator = pd.to_datetime('2024-05-01')  # Fixed start date as per user request
# logger.info(f"Start date for denominator: {start_date_denominator.strftime('%Y-%m-%d')}")
#
# # -------------------------- Aggregate Data -------------------------- #
#
# # -------------------------- Compute Shipped Orders and Open Orders Percentage -------------------------- #
#
# # Define May 1, 2024 as the start date
# start_date_shipped = pd.to_datetime('2024-05-01')
#
# # Filter shipped orders since May 1
# # An order is considered shipped if 'Shipped Quantity' >0 and 'Order Date' <= 'Report Date'
# shipped_data = data[(data['Order Date'] >= start_date_shipped) &
#                     (data['Shipped Quantity'] > 0)]
#
# # Group shipped orders by 'Order Date' and count unique 'Order ID's
# shipped_per_day = shipped_data.groupby('Order Date')['Order ID'].nunique().reset_index(name='Shipped Orders')
#
# # Sort shipped_per_day by 'Order Date' ascending
# shipped_per_day = shipped_per_day.sort_values('Order Date')
#
# # Compute cumulative shipped orders
# shipped_per_day['Cumulative Shipped Orders'] = shipped_per_day['Shipped Orders'].cumsum()
#
# # Create a DataFrame with all report dates sorted ascending
# report_dates_sorted = sorted(report_dates_set)
#
# # Create a DataFrame for report dates
# report_dates_df = pd.DataFrame({'Report Date': report_dates_sorted})
#
# # Merge shipped_per_day with report_dates_df using 'Order Date' and 'Report Date'
# # For each report date, find the latest 'Order Date' <= 'Report Date' and get the corresponding 'Cumulative Shipped Orders'
# ship_and_open_df = pd.merge_asof(report_dates_df.sort_values('Report Date'),
#                                  shipped_per_day[['Order Date', 'Cumulative Shipped Orders']].sort_values('Order Date'),
#                                  left_on='Report Date',
#                                  right_on='Order Date',
#                                  direction='backward')
#
# # Fill NaN with 0 (if no shipments up to that report date)
# ship_and_open_df['Cumulative Shipped Orders'] = ship_and_open_df['Cumulative Shipped Orders'].fillna(0).astype(int)
#
# # Merge with total_aggregation
# total_aggregation = data[data['Order Category'] == 'Outstanding Over 5 Weeks'] \
#     .dropna(subset=['Report Date']) \
#     .groupby('Report Date') \
#     .size() \
#     .reset_index(name='Total Open Orders Over 5 Weeks')
#
# ship_and_open_df = pd.merge(total_aggregation, ship_and_open_df[['Report Date', 'Cumulative Shipped Orders']], on='Report Date', how='left')
#
# # Calculate the percentage
# ship_and_open_df['% Open Orders Over 5 Weeks'] = (ship_and_open_df['Total Open Orders Over 5 Weeks'] / ship_and_open_df['Cumulative Shipped Orders']) * 100
#
# # Handle division by zero
# ship_and_open_df['% Open Orders Over 5 Weeks'] = ship_and_open_df['% Open Orders Over 5 Weeks'].replace([np.inf, -np.inf], np.nan).fillna(0)
#
# # Sort by Report Date ascending
# ship_and_open_df = ship_and_open_df.sort_values('Report Date')
#
# # Format the table for display
# display_ship_and_open = ship_and_open_df.copy()
# display_ship_and_open['Report Date'] = display_ship_and_open['Report Date'].dt.strftime('%Y-%m-%d')
# display_ship_and_open['Total Open Orders Over 5 Weeks'] = display_ship_and_open['Total Open Orders Over 5 Weeks'].astype(int)
# display_ship_and_open['Cumulative Shipped Orders'] = display_ship_and_open['Cumulative Shipped Orders'].astype(int)
# display_ship_and_open['% Open Orders Over 5 Weeks'] = display_ship_and_open['% Open Orders Over 5 Weeks'].round(2)
#
# # Apply formatting
# formatted_ship_and_open = display_ship_and_open.style.format({
#     # 'Cumulative Shipped Orders': "{:,}",
#     'Total Open Orders Over 5 Weeks': "{:,}",
#     '% Open Orders Over 5 Weeks': "{:.2f}%"
# })
#
# # -------------------------- Display Shipped Orders and Open Orders Over 5 Weeks Old by Report Date -------------------------- #
#
# st.subheader("Shipped Orders and Open Orders Over 5 Weeks Old by Report Date")
# st.write("""
# This table shows, for each report date, the **cumulative number of shipped orders since May 1**, the **total number of open orders over five weeks old**, and the **percentage of open orders over five weeks old**.
# """)
# st.dataframe(formatted_ship_and_open)
#
# # -------------------------- Display Percentage of Open Orders Over 5 Weeks Old by Club and Report Date -------------------------- #
#
# st.subheader("Percentage of Open Orders Over 5 Weeks Old by Club and Report Date")
# st.write("""
# This table shows the percentage of open orders over five weeks old for each club over time, along with the total number of such orders and the number of orders shipped.
# """)
#
# # Calculate total orders per club per report date
# total_orders_per_club_date = data.groupby(['Club', 'Report Date'])['Order ID'].nunique().reset_index(name='Total Orders')
#
# # Calculate 'Outstanding Over 5 Weeks' orders per club per report date
# outstanding_orders_per_club_date = data[data['Order Category'] == 'Outstanding Over 5 Weeks'].groupby(['Club', 'Report Date'])['Order ID'].nunique().reset_index(name='Outstanding Over 5 Weeks')
#
# # Merge total orders with outstanding orders
# percentage_per_club_date = pd.merge(total_orders_per_club_date, outstanding_orders_per_club_date, on=['Club', 'Report Date'], how='left')
#
# # Fill NaN values with 0
# percentage_per_club_date['Outstanding Over 5 Weeks'] = percentage_per_club_date['Outstanding Over 5 Weeks'].fillna(0)
#
# # Calculate percentage
# percentage_per_club_date['Percentage Over 5 Weeks Old (%)'] = (percentage_per_club_date['Outstanding Over 5 Weeks'] / percentage_per_club_date['Total Orders']) * 100
#
# # Round percentage to two decimal places
# percentage_per_club_date['Percentage Over 5 Weeks Old (%)'] = percentage_per_club_date['Percentage Over 5 Weeks Old (%)'].round(2)
#
# # Calculate shipped orders per club per report date
# shipped_orders_per_club_date = data[data['Shipped Quantity'] > 0].groupby(['Club', 'Report Date'])['Order ID'].nunique().reset_index(name='Shipped Orders')
#
# # Merge with the existing percentage_per_club_date
# percentage_per_club_date = pd.merge(percentage_per_club_date, shipped_orders_per_club_date, on=['Club', 'Report Date'], how='left')
#
# # Fill NaN shipped orders with 0
# percentage_per_club_date['Shipped Orders'] = percentage_per_club_date['Shipped Orders'].fillna(0).astype(int)
#
# # Select relevant columns
# percentage_per_club_date = percentage_per_club_date[['Club', 'Report Date', 'Percentage Over 5 Weeks Old (%)', 'Outstanding Over 5 Weeks', 'Shipped Orders']]
#
# # Rename columns for clarity
# percentage_per_club_date.rename(columns={
#     'Outstanding Over 5 Weeks': 'Total Open Orders Over 5 Weeks'
# }, inplace=True)
#
# # Pivot the table to have Report Dates as columns with both percentage and total orders
# percentage_pivot = percentage_per_club_date.pivot(index='Club', columns='Report Date', values=['Percentage Over 5 Weeks Old (%)', 'Total Open Orders Over 5 Weeks', 'Shipped Orders'])
#
# # Flatten the MultiIndex columns
# percentage_pivot.columns = [f"{date.strftime('%Y-%m-%d')} {metric}" for metric, date in percentage_pivot.columns]
#
# # Reset index to turn 'Club' back into a column
# percentage_pivot.reset_index(inplace=True)
#
# # Sort the pivot table by Club name
# percentage_pivot = percentage_pivot.sort_values('Club')
#
# # Convert report date columns to string format for easier formatting
# percentage_pivot.columns = ['Club'] + [col for col in percentage_pivot.columns if col != 'Club']
#
# # Reorder columns to have 'Club' first, followed by sorted report dates with metrics
# sorted_report_dates_strings = sorted(report_date_strings)
#
# # Create a list for new column order with alternating Percentage, Total, and Shipped
# new_column_order = ['Club']
# for date in sorted_report_dates_strings:
#     pct_col = f"{date} Percentage Over 5 Weeks Old (%)"
#     total_col = f"{date} Total Open Orders Over 5 Weeks"
#     shipped_col = f"{date} Shipped Orders"
#     # Rename columns to have 'Percentage', 'Total', and 'Shipped'
#     if pct_col in percentage_pivot.columns and total_col in percentage_pivot.columns and shipped_col in percentage_pivot.columns:
#         percentage_pivot.rename(columns={
#             pct_col: f"{date} %",
#             total_col: f"{date} Total",
#             shipped_col: f"{date} Shipped"
#         }, inplace=True)
#         new_column_order.extend([f"{date} %", f"{date} Total", f"{date} Shipped"])
#     elif pct_col in percentage_pivot.columns and total_col not in percentage_pivot.columns and shipped_col not in percentage_pivot.columns:
#         percentage_pivot.rename(columns={
#             pct_col: f"{date} %"
#         }, inplace=True)
#         new_column_order.append(f"{date} %")
#     elif pct_col not in percentage_pivot.columns and total_col in percentage_pivot.columns and shipped_col not in percentage_pivot.columns:
#         percentage_pivot.rename(columns={
#             total_col: f"{date} Total"
#         }, inplace=True)
#         new_column_order.append(f"{date} Total")
#     elif pct_col not in percentage_pivot.columns and total_col not in percentage_pivot.columns and shipped_col in percentage_pivot.columns:
#         percentage_pivot.rename(columns={
#             shipped_col: f"{date} Shipped"
#         }, inplace=True)
#         new_column_order.append(f"{date} Shipped")
#     # If none exist, skip
#
# # Ensure all columns in new_column_order exist
# existing_columns = [col for col in new_column_order if col in percentage_pivot.columns]
# percentage_pivot = percentage_pivot[existing_columns]
#
# # Apply formatting to percentage columns
# percentage_cols = [col for col in percentage_pivot.columns if '%' in col]
#
# # Apply formatting using Pandas Styler
# styled_percentage_pivot = percentage_pivot.style.format({
#     col: "{:.2f}%" for col in percentage_cols
# })
#
# # Display the styled percentage pivot table
# st.write(styled_percentage_pivot)
#
# # -------------------------- Plot Total Open Orders Over 5 Weeks Old -------------------------- #
# st.subheader("Trend of Total Open Orders Over 5 Weeks Old")
# st.write("This graph shows the total number of open orders over five weeks old across all clubs for each report date.")
#
# # Plot using Plotly
# fig_total_trend = px.line(
#     ship_and_open_df,
#     x='Report Date',
#     y='Total Open Orders Over 5 Weeks',
#     markers=True,
#     title='Total Open Orders Over 5 Weeks Old Over Time',
#     labels={'Report Date': 'Report Date', 'Total Open Orders Over 5 Weeks': 'Total Open Orders Over 5 Weeks'}
# )
#
# fig_total_trend.update_layout(
#     xaxis=dict(tickangle=45),
#     yaxis=dict(title='Total Open Orders Over 5 Weeks'),
#     template='plotly_white'
# )
#
# st.plotly_chart(fig_total_trend, use_container_width=True)
# logger.info("Plotted Trend of Total Open Orders Over 5 Weeks Old using Plotly")
#
# # -------------------------- Orders Shipped Over Time -------------------------- #
#
# st.subheader("Orders Shipped Over Time")
# st.write("""
# This graph shows the number of orders shipped over time based on shipping dates, grouped by week.
# """)
#
# if not shipping_data.empty and all(col in shipping_data.columns for col in ['Customer Reference', 'Club Name', 'Shipping Date', 'Date Created']):
#     # Convert 'Shipping Date' and 'Date Created' to datetime
#     shipping_data['Shipping Date'] = pd.to_datetime(shipping_data['Shipping Date'], errors='coerce')
#     shipping_data['Date Created'] = pd.to_datetime(shipping_data['Date Created'], errors='coerce')
#     # Remove entries with invalid dates
#     shipping_data = shipping_data.dropna(subset=['Shipping Date', 'Date Created'])
#
#     # Remove duplicates of 'Customer Reference' by aggregating
#     order_data_shipping = shipping_data.groupby('Customer Reference', as_index=False).agg({
#         'Club Name': 'first',         # Assuming 'Club Name' is consistent within an order
#         'Date Created': 'min',        # Earliest creation date among items
#         'Shipping Date': 'max'        # Latest shipping date among items
#     })
#
#     # Compute time difference in days for each order
#     order_data_shipping['Time Difference'] = (order_data_shipping['Shipping Date'] - order_data_shipping['Date Created']).dt.days
#
#     # Set 'Shipping Date' as index
#     order_data_shipping.set_index('Shipping Date', inplace=True)
#
#     # Group by week and count unique orders
#     orders_over_time_shipping = order_data_shipping.resample('W').size().reset_index(name='Unique Orders')
#
#     # Sort by 'Shipping Date'
#     orders_over_time_shipping = orders_over_time_shipping.sort_values('Shipping Date')
#
#     # Plot the number of orders over time
#     fig_orders_over_time = px.line(
#         orders_over_time_shipping,
#         x='Shipping Date',
#         y='Unique Orders',
#         markers=True,
#         title='Orders Shipped Over Time (Weekly)',
#         labels={'Shipping Date': 'Week Starting', 'Unique Orders': 'Number of Orders'}
#     )
#     fig_orders_over_time.update_layout(
#         xaxis=dict(tickangle=45),
#         yaxis=dict(title='Number of Orders'),
#         template='plotly_white'
#     )
#     st.plotly_chart(fig_orders_over_time, use_container_width=True)
#     logger.info("Plotted Orders Shipped Over Time (Weekly) using Plotly")
# else:
#     st.write("No shipping data available to display orders shipped over time.")
#     logger.warning("Shipping data is empty or missing required columns.")
#
# # -------------------------- Shipping Time for Each Order Over Time -------------------------- #
#
# st.subheader("Shipping Time for Each Order Over Time")
# st.write("""
# This graph shows how long it took for each order to ship, based on the shipping date over time.
# """)
#
# if not shipping_data.empty and all(col in shipping_data.columns for col in ['Customer Reference', 'Club Name', 'Shipping Date', 'Date Created']):
#     # Reset index to access 'Shipping Date' as a column
#     order_data_shipping.reset_index(inplace=True)
#
#     # Compute shipping times
#     shipping_times = order_data_shipping[['Shipping Date', 'Time Difference', 'Customer Reference']]  # Added 'Customer Reference'
#
#     # Plot using Plotly
#     fig_shipping_time = px.scatter(
#         shipping_times,
#         x='Shipping Date',
#         y='Time Difference',
#         title='Shipping Time for Each Order Over Time',
#         labels={'Shipping Date': 'Shipping Date', 'Time Difference': 'Shipping Time (days)'},
#         opacity=0.6,
#         hover_data=['Time Difference', 'Customer Reference']  # Included 'Customer Reference' in hover data
#     )
#
#     # Update layout for better readability
#     fig_shipping_time.update_layout(
#         template='plotly_white',
#         xaxis=dict(tickangle=45),
#         yaxis=dict(title='Shipping Time (days)')
#     )
#
#     st.plotly_chart(fig_shipping_time, use_container_width=True)
#     logger.info("Plotted Shipping Time for Each Order Over Time using Plotly")
# else:
#     st.write("No shipping data available to display shipping time graph.")
#     logger.warning("Shipping data is empty or missing required columns for shipping time graph.")
#
# # -------------------------- Average Shipping Time per Month -------------------------- #
#
# st.subheader("Average Shipping Time per Month")
# st.write("""
# This bar chart displays the **average shipping time (in days)** for each month.
# """)
#
# if not shipping_data.empty and all(col in shipping_data.columns for col in ['Customer Reference', 'Club Name', 'Shipping Date', 'Date Created']):
#     # Reset index to access 'Shipping Date' as a column
#     order_data_shipping.reset_index(inplace=True)
#
#     # Extract month and year for grouping
#     order_data_shipping['Shipping Month'] = order_data_shipping['Shipping Date'].dt.to_period('M').dt.to_timestamp()
#
#     # Compute average shipping time per month
#     avg_shipping_time_per_month = order_data_shipping.groupby('Shipping Month')['Time Difference'].mean().reset_index()
#
#     # Sort by Shipping Month
#     avg_shipping_time_per_month = avg_shipping_time_per_month.sort_values('Shipping Month')
#
#     # Plot the average shipping time per month as a bar chart
#     fig_avg_shipping = px.bar(
#         avg_shipping_time_per_month,
#         x='Shipping Month',
#         y='Time Difference',
#         title='Average Shipping Time per Month',
#         labels={'Shipping Month': 'Month', 'Time Difference': 'Average Shipping Time (days)'},
#         text='Time Difference'
#     )
#     fig_avg_shipping.update_traces(texttemplate='%{text:.2f}', textposition='outside')
#     fig_avg_shipping.update_layout(
#         xaxis=dict(tickangle=45),
#         yaxis=dict(title='Average Shipping Time (days)'),
#         template='plotly_white'
#     )
#     st.plotly_chart(fig_avg_shipping, use_container_width=True)
#     logger.info("Plotted Average Shipping Time per Month as a Bar Chart using Plotly")
# else:
#     st.write("No shipping data available to display the average shipping time per month.")
#     logger.warning("Shipping data is empty or missing required columns for average shipping time bar chart.")
#
# # -------------------------- Average Order Time by Club -------------------------- #
#
# st.subheader("Average Order Time by Club")
# st.write("""
# This table displays the **average shipping time (in days)** for each club, along with the **number of orders shipped by each club**.
# """)
#
# # Compute average shipping time per club and number of orders shipped
# average_order_time = data.groupby('Club').agg({
#     'Time Difference': 'mean',
#     'Order ID': 'count'
# }).reset_index()
# average_order_time['Time Difference'] = average_order_time['Time Difference'].round(2)
# average_order_time.rename(columns={'Order ID': 'Number of Orders Shipped'}, inplace=True)
#
# # Display the table
# st.dataframe(average_order_time.style.format({
#     'Club': lambda x: x.title(),
#     'Time Difference': "{:.2f} days",
#     'Number of Orders Shipped': "{:,}"
# }))
#
# # Plot Average Order Time by Club
# fig_avg_order_time_club = px.bar(
#     average_order_time,
#     x='Club',
#     y='Time Difference',
#     title='Average Shipping Time by Club',
#     labels={'Club': 'Club', 'Time Difference': 'Average Shipping Time (days)'},
#     text='Time Difference'
# )
# fig_avg_order_time_club.update_traces(texttemplate='%{text:.2f}', textposition='outside')
# fig_avg_order_time_club.update_layout(
#     xaxis=dict(tickangle=45),
#     yaxis=dict(title='Average Shipping Time (days)'),
#     uniformtext_minsize=8,
#     uniformtext_mode='hide',
#     template='plotly_white'
# )
# st.plotly_chart(fig_avg_order_time_club, use_container_width=True)
# logger.info("Plotted Average Order Time by Club using Plotly")
#
# # -------------------------- Final Touches -------------------------- #
#
# # You can add additional sections or features as needed.
#
# # -------------------------- End of File -------------------------- #
