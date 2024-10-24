# capelli_orders_app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import os  # To handle local file operations
import logging
from logging.handlers import RotatingFileHandler
import faulthandler
import tempfile  # For cross-platform temporary file handling
import plotly.express as px

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

# Selection boxes for start and end dates
selected_start_date = st.sidebar.selectbox("Select Start Date", options=report_date_strings, index=0, key='start_date')
selected_end_date = st.sidebar.selectbox("Select End Date", options=report_date_strings, index=len(report_date_strings)-1, key='end_date')

# Convert selected dates back to datetime
start_date = pd.to_datetime(selected_start_date)
end_date = pd.to_datetime(selected_end_date)

# Ensure start_date is before end_date
if start_date > end_date:
    st.error("Start date must be before end date.")
    logger.error("Start date is after end date.")
    st.stop()

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

# Aggregate data: Count of 'Outstanding Over 5 Weeks' per Club per Report Date
aggregation = data[data['Order Category'] == 'Outstanding Over 5 Weeks'].groupby(['Club', 'Report Date']).size().reset_index(name='Open Orders Over 5 Weeks')

# Pivot the table to have Report Dates as columns and Clubs as rows
pivot_table = aggregation.pivot(index='Club', columns='Report Date', values='Open Orders Over 5 Weeks').fillna(0).astype(int)

# Reset index to turn 'Club' back into a column
pivot_table.reset_index(inplace=True)

# Sort the pivot table by Club name
pivot_table = pivot_table.sort_values('Club')

# -------------------------- Compute Total Open Orders Over 5 Weeks Old -------------------------- #

# Aggregate total open orders over 5 weeks old across all clubs per report date
total_aggregation = data[data['Order Category'] == 'Outstanding Over 5 Weeks'] \
    .dropna(subset=['Report Date']) \
    .groupby('Report Date') \
    .size() \
    .reset_index(name='Total Open Orders Over 5 Weeks')

# Sort the aggregation by report date
total_aggregation = total_aggregation.sort_values('Report Date')

# Create a separate DataFrame for display with formatted dates
display_total_aggregation = total_aggregation.copy()
display_total_aggregation['Report Date'] = display_total_aggregation['Report Date'].dt.strftime('%Y-%m-%d')

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
report_dates_sorted = total_aggregation['Report Date'].sort_values().unique()

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
ship_and_open_df = pd.merge(total_aggregation, ship_and_open_df[['Report Date', 'Cumulative Shipped Orders']], on='Report Date', how='left')

# Calculate the percentage
ship_and_open_df['Percentage Open Over 5 Weeks'] = (ship_and_open_df['Total Open Orders Over 5 Weeks'] / ship_and_open_df['Cumulative Shipped Orders']) * 100

# Handle division by zero
ship_and_open_df['Percentage Open Over 5 Weeks'] = ship_and_open_df['Percentage Open Over 5 Weeks'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Sort by Report Date ascending
ship_and_open_df = ship_and_open_df.sort_values('Report Date')

# Format the table for display
display_ship_and_open = ship_and_open_df.copy()
display_ship_and_open['Report Date'] = display_ship_and_open['Report Date'].dt.strftime('%Y-%m-%d')
display_ship_and_open['Total Open Orders Over 5 Weeks'] = display_ship_and_open['Total Open Orders Over 5 Weeks'].astype(int)
display_ship_and_open['Cumulative Shipped Orders'] = display_ship_and_open['Cumulative Shipped Orders'].astype(int)
display_ship_and_open['Percentage Open Over 5 Weeks'] = display_ship_and_open['Percentage Open Over 5 Weeks'].round(2)

# Apply formatting
formatted_ship_and_open = display_ship_and_open.style.format({
    'Cumulative Shipped Orders': "{:,}",
    'Total Open Orders Over 5 Weeks': "{:,}",
    'Percentage Open Over 5 Weeks': "{:.2f}%"
})

# -------------------------- Display Shipped and Open Orders Percentage Table -------------------------- #

st.subheader("Shipped Orders and Open Orders Over 5 Weeks Old by Report Date")
st.write("""
This table shows, for each report date, the **cumulative number of shipped orders since May 1**, the **total number of open orders over five weeks old**, and the **percentage of open orders over five weeks old**.
""")
st.dataframe(formatted_ship_and_open)

# -------------------------- Display Total Summary Table (Modified) -------------------------- #

st.subheader("Total Open Orders Over 5 Weeks Old by Report Date and Club")
st.write("""
This table shows the total number of open orders over five weeks old for each **club** and **report date**.
""")

# Create a copy of the pivot_table to avoid modifying the original DataFrame
formatted_pivot = pivot_table.copy()

# Convert report date columns to string format for easier formatting
formatted_pivot.columns = ['Club'] + [col.strftime('%Y-%m-%d') for col in formatted_pivot.columns[1:]]

# Convert sorted_report_dates (Timestamps) to strings matching the column names
sorted_report_dates_strings = [col.strftime('%Y-%m-%d') for col in sorted_report_dates]

# Reorder columns to have 'Club' first, followed by sorted report dates
formatted_pivot = formatted_pivot[['Club'] + sorted_report_dates_strings]

# Identify numeric columns (all columns except 'Club')
numeric_columns = sorted_report_dates_strings

# Apply formatting only to numeric columns using Pandas Styler
styled_pivot = formatted_pivot.style.format("{:,}", subset=numeric_columns)

# Display the styled pivot table using st.write()
st.write(styled_pivot)

# -------------------------- Plot Total Open Orders Over 5 Weeks Old by Report Date -------------------------- #

st.subheader("Trend of Total Open Orders Over 5 Weeks Old")
st.write("This graph shows the total number of open orders over five weeks old across all clubs for each report date.")

# Plot using Plotly
fig_total_trend = px.line(
    total_aggregation,
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

# -------------------------- Display Percentage Summary Table (Reintroduced) -------------------------- #

st.subheader("Percentage of Open Orders Over 5 Weeks Old by Club and Report Date")
st.write("""
This table shows the **percentage of open orders over five weeks old for each club** over time.
""")

# Create a DataFrame of all combinations of clubs and report dates
from itertools import product  # Import here as it's needed
clubs_report_dates = pd.DataFrame(list(product(clubs, report_dates)), columns=['Club', 'Report Date'])

# Compute the numerator: Number of 'Outstanding Over 5 Weeks' orders for each club at each report date
numerator_df = data[data['Order Category'] == 'Outstanding Over 5 Weeks'].groupby(['Club', 'Report Date'])['Order ID'].nunique().reset_index(name='Outstanding_Orders')

# Merge with clubs_report_dates to ensure all combinations are present
clubs_report_dates = pd.merge(clubs_report_dates, numerator_df, on=['Club', 'Report Date'], how='left')
clubs_report_dates['Outstanding_Orders'] = clubs_report_dates['Outstanding_Orders'].fillna(0)

# Compute the denominator for each club and report date
denominator_list = []

for report_date in report_dates:
    end_date = report_date - pd.DateOffset(weeks=5)
    temp = data[
        (data['Order Date'] >= start_date_denominator) &
        (data['Order Date'] <= end_date)
    ]
    temp_grouped = temp.groupby('Club')['Order ID'].nunique().reset_index()
    temp_grouped['Report Date'] = report_date
    temp_grouped['Total_Orders'] = temp_grouped['Order ID']
    temp_grouped.drop(columns=['Order ID'], inplace=True)
    denominator_list.append(temp_grouped)

# Concatenate all temp_grouped DataFrames
denominator_df = pd.concat(denominator_list, ignore_index=True)

# Merge denominator with clubs_report_dates
clubs_report_dates = pd.merge(clubs_report_dates, denominator_df[['Club', 'Report Date', 'Total_Orders']], on=['Club', 'Report Date'], how='left')
clubs_report_dates['Total_Orders'] = clubs_report_dates['Total_Orders'].fillna(0)

# Compute the percentage
clubs_report_dates['Percentage_Open'] = (clubs_report_dates['Outstanding_Orders'] / clubs_report_dates['Total_Orders']) * 100

# Handle division by zero
clubs_report_dates['Percentage_Open'] = clubs_report_dates['Percentage_Open'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Round the percentage to two decimal places
clubs_report_dates['Percentage_Open'] = clubs_report_dates['Percentage_Open'].round(2)

# Create a copy for display purposes and convert 'Report Date' to string
clubs_report_dates_display = clubs_report_dates.copy()
clubs_report_dates_display['Report Date'] = clubs_report_dates_display['Report Date'].dt.strftime('%Y-%m-%d')

# Pivot the table to have Report Dates as columns and Clubs as rows
percentage_pivot = clubs_report_dates_display.pivot(index='Club', columns='Report Date', values='Percentage_Open')

# Sort the pivot table by Club name
percentage_pivot = percentage_pivot.sort_values('Club')

# Apply formatting to the pivot table
styled_percentage_pivot = percentage_pivot.style.format("{:.2f}%")

# Display the styled pivot table
st.write(styled_percentage_pivot)

# -------------------------- Data Filtering -------------------------- #

# Filter data based on user selection
if selected_club != 'All Clubs':
    data_filtered = data[data['Club'] == selected_club]
    logger.info(f"Filtered data for club: {selected_club}")
else:
    data_filtered = data.copy()
    logger.info("Filtered data for All Clubs")

# Modify the filtering to use 'Report Date' instead of 'Order Date'
data_filtered = data_filtered[(data_filtered['Report Date'] >= start_date) & (data_filtered['Report Date'] <= end_date)]
logger.info(f"Filtered data between {selected_start_date} and {selected_end_date} based on Report Date")

# -------------------------- Detailed Analysis for Selected Club -------------------------- #

if selected_club != 'All Clubs' and not data_filtered.empty:
    st.subheader(f"Detailed Analysis for {selected_club}")
    st.write(f"Here is a detailed analysis for **{selected_club}** based on the selected date range.")

    # Summarize the findings
    explanation = ""

    # Current outstanding orders over 5 weeks old
    latest_date = data_filtered['Report Date'].max()
    if pd.isna(latest_date) or data_filtered.empty:
        # Do not display "No data available" message
        pass
    else:
        latest_data = data_filtered[data_filtered['Report Date'] == latest_date]
        current_outstanding = latest_data[latest_data['Order Category'] == 'Outstanding Over 5 Weeks']['Order ID'].nunique()
        explanation += f"As of {latest_date.strftime('%Y-%m-%d')}, **{selected_club}** has **{current_outstanding}** outstanding orders over 5 weeks old.\n\n"

        # Trend over time
        trend_over_time = data_filtered[data_filtered['Order Category'] == 'Outstanding Over 5 Weeks'].groupby('Report Date').size().reset_index(name='Open Orders Over 5 Weeks')
        trend_over_time = trend_over_time.sort_values('Report Date')

        # Interpret whether the situation is getting better or worse
        if not trend_over_time.empty:
            if trend_over_time['Open Orders Over 5 Weeks'].is_monotonic_decreasing:
                explanation += f"The number of outstanding orders over 5 weeks old for **{selected_club}** is decreasing over time, indicating an improvement.\n\n"
            elif trend_over_time['Open Orders Over 5 Weeks'].is_monotonic_increasing:
                explanation += f"The number of outstanding orders over 5 weeks old for **{selected_club}** is increasing over time, indicating a worsening situation.\n\n"
            else:
                explanation += f"The number of outstanding orders over 5 weeks old for **{selected_club}** fluctuates over time.\n\n"
        else:
            # Do not display "No data available" message
            pass

        st.write(explanation)

        # Plot the trend for the selected club using Plotly
        st.subheader(f"Trend of Outstanding Orders Over 5 Weeks Old for {selected_club}")
        st.write("This graph shows how the number of outstanding orders over 5 weeks old has changed over time for the selected club.")

        if not trend_over_time.empty:
            fig4 = px.line(
                trend_over_time,
                x='Report Date',
                y='Open Orders Over 5 Weeks',
                markers=True,
                title=f'Outstanding Orders Over 5 Weeks Old Over Time for {selected_club}',
                labels={'Report Date': 'Report Date', 'Open Orders Over 5 Weeks': 'Number of Outstanding Orders'}
            )
            fig4.update_layout(
                xaxis=dict(tickangle=45),
                yaxis=dict(title='Number of Outstanding Orders'),
                template='plotly_white'
            )
            st.plotly_chart(fig4, use_container_width=True)
            logger.info(f"Plotted Trend for {selected_club} using Plotly")
        else:
            # Do not display "No data available" message
            pass

        # Provide an interpretation paragraph
        st.write("**Interpretation:**")
        st.write(f"""
        The data indicates changes in the number of open or partially shipped orders over five weeks old for **{selected_club}**. By analyzing the trend, we can assess the club's order processing efficiency. An increasing trend suggests a backlog forming, while a decreasing trend indicates progress in reducing overdue orders.
        """)
else:
    # Do not display "No data available" message
    pass

# -------------------------- New Graphs: Orders Happening Over Time and Shipping Times -------------------------- #

st.subheader("Orders Happening Over Time")

st.write("""
This graph shows the number of orders happening over time based on shipping dates, grouped by week.
""")

# Load the new data from 'shippingdates' subdirectory
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

if not shipping_data.empty:
    # Ensure necessary columns are present
    required_columns = ['Customer Reference', 'Club Name', 'Shipping Date', 'Date Created']
    if not all(col in shipping_data.columns for col in required_columns):
        st.error(f"The shipping data file must contain the following columns: {', '.join(required_columns)}")
        logger.error(f"Shipping data missing required columns: {required_columns}")
    else:
        # Convert 'Shipping Date' and 'Date Created' to datetime
        shipping_data['Shipping Date'] = pd.to_datetime(shipping_data['Shipping Date'], errors='coerce')
        shipping_data['Date Created'] = pd.to_datetime(shipping_data['Date Created'], errors='coerce')
        # Remove entries with invalid dates
        shipping_data = shipping_data.dropna(subset=['Shipping Date', 'Date Created'])
        # Compute time difference in days
        shipping_data['Time Difference'] = (shipping_data['Shipping Date'] - shipping_data['Date Created']).dt.days
        # Exclude orders with missing shipping dates (already done with dropna)

        # Set 'Shipping Date' as index
        shipping_data.set_index('Shipping Date', inplace=True)

        # Group by week and count unique 'Customer Reference'
        orders_over_time = shipping_data['Customer Reference'].resample('W').nunique().reset_index(name='Unique Orders')
        # Sort by 'Shipping Date'
        orders_over_time = orders_over_time.sort_values('Shipping Date')

        # Plot the number of orders over time
        fig_orders_over_time = px.line(
            orders_over_time,
            x='Shipping Date',
            y='Unique Orders',
            markers=True,
            title='Orders Happening Over Time (Weekly)',
            labels={'Shipping Date': 'Week Starting', 'Unique Orders': 'Number of Orders'}
        )
        fig_orders_over_time.update_layout(
            xaxis=dict(tickangle=45),
            yaxis=dict(title='Number of Orders'),
            template='plotly_white'
        )
        st.plotly_chart(fig_orders_over_time, use_container_width=True)
        logger.info("Plotted Orders Happening Over Time (Weekly) using Plotly")

        # Plot shipping time for each order over time
        st.subheader("Shipping Time for Each Order Over Time")
        st.write("""
        This graph shows how long it took for each order to ship, based on the shipping date over time.
        """)

        # Reset index to access 'Shipping Date' as a column
        shipping_data.reset_index(inplace=True)

        # Compute the overall average shipping time
        overall_avg_shipping_time = shipping_data['Time Difference'].mean().round(2)

        # Display the overall average shipping time
        st.markdown(f"### **Average Shipping Time Across All Orders: {overall_avg_shipping_time} days**")

        # Create a scatter plot of shipping time per order over time
        fig_shipping_times = px.scatter(
            shipping_data,
            x='Shipping Date',
            y='Time Difference',
            title='Shipping Time for Each Order Over Time',
            labels={'Shipping Date': 'Shipping Date', 'Time Difference': 'Shipping Time (days)'},
            hover_data=['Customer Reference', 'Club Name']
        )
        fig_shipping_times.update_layout(
            xaxis=dict(tickangle=45),
            yaxis=dict(title='Shipping Time (days)'),
            template='plotly_white'
        )
        st.plotly_chart(fig_shipping_times, use_container_width=True)
        logger.info("Plotted Shipping Time for Each Order Over Time using Plotly")
else:
    st.write("No shipping data available to display orders over time.")
    logger.warning("Shipping data is empty or not loaded.")
