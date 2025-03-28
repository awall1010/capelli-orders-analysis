# overallanalysis.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os  # To handle local file operations
import logging
import faulthandler

# -------------------------- Enable Fault Handler -------------------------- #

# Enable faulthandler to get tracebacks on segmentation faults
faulthandler.enable()

# -------------------------- Logging Setup -------------------------- #

# Configure logging to display messages in the console and save to a file
logging.basicConfig(
    level=logging.DEBUG,
    filename='app_debug.log',
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# -------------------------- Streamlit App Setup -------------------------- #

st.set_page_config(page_title="Capelli Sport Orders Analysis", layout="wide")

st.title("Capelli Sport Orders Analysis")
st.write("""
This app analyzes and visualizes Capelli Sport order data. Explore trends over time, focus on specific clubs, and understand the dynamics of open orders over five weeks old.
""")

# Set visualization style
sns.set(style='whitegrid')

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

def load_data_from_directory(data_dir):
    """
    Loads and processes all CSV files from the specified directory.

    Parameters:
    - data_dir (str): Path to the directory containing CSV files.

    Returns:
    - pd.DataFrame: Combined and processed DataFrame.
    - set: Set of unique report dates.
    """
    if not os.path.exists(data_dir):
        st.error(f"The data directory '{data_dir}' does not exist. Please ensure it is present in your repository.")
        logger.error(f"Data directory '{data_dir}' does not exist.")
        st.stop()

    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not data_files:
        st.error(f"No CSV files found in the '{data_dir}' directory. Please add the required data files.")
        logger.error(f"No CSV files found in '{data_dir}'.")
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
            st.error(f"Error reading {filename}: {e}")
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

        # Categorize orders based on 'Order Range' and 'Combined Order Status'
        def categorize_order(row):
            if row['Order Range'] == 'older than 5 weeks':
                if row['Combined Order Status'] in ['open', 'partially shipped']:
                    return 'Outstanding Over 5 Weeks'
            return 'Other'

        df['Order Category'] = df.apply(categorize_order, axis=1)
        logger.debug(f"Order categories in {filename}: {df['Order Category'].unique()}")

        # Correct any 'Oth' entries to 'Other'
        oth_entries = df[df['Order Category'].str.lower() == 'oth']
        if not oth_entries.empty:
            st.warning(f"Found {len(oth_entries)} entries with 'Order Category' as 'Oth' in {filename}. Correcting them to 'Other'.")
            logger.warning(f"Found 'Oth' entries in {filename}. Correcting to 'Other'.")
            df['Order Category'] = df['Order Category'].replace('oth', 'Other')

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
    return data, report_dates_set

# -------------------------- Data Loading -------------------------- #

DATA_DIR = 'reports'  # Ensure this directory exists in your repository
data, report_dates_set = load_data_from_directory(DATA_DIR)

# -------------------------- Data Preprocessing -------------------------- #

# Convert date columns to datetime
data['Order Date'] = pd.to_datetime(data['Order Date'], errors='coerce')
data['Report Date'] = pd.to_datetime(data['Report Date'], errors='coerce')

# Ensure numeric columns are properly typed
numeric_columns = ['Shipped Quantity', 'Unshipped Quantity']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Handle missing values
data.fillna({'Shipped Quantity': 0, 'Unshipped Quantity': 0}, inplace=True)
data['Combined Order Status'] = data['Combined Order Status'].fillna('unknown')

# Define a function to categorize orders based on 'Order Range'
def categorize_order(row):
    if row['Order Range'] == 'older than 5 weeks':
        if row['Combined Order Status'] in ['open', 'partially shipped']:
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

# -------------------------- Check for Missing Report Dates -------------------------- #

st.subheader("Debugging: Rows with Missing Report Date")
missing_report_date = data[data['Report Date'].isna()]
num_missing = missing_report_date.shape[0]
st.write(f"Total rows with missing Report Date: {num_missing}")

if num_missing > 0:
    st.dataframe(missing_report_date)

# -------------------------- Aggregate Data -------------------------- #

# Aggregate data: Count of 'Outstanding Over 5 Weeks' per Club per Report Date
aggregation = data[data['Order Category'] == 'Outstanding Over 5 Weeks'].groupby(['Club', 'Report Date']).size().reset_index(name='Open Orders Over 5 Weeks')

# Pivot the table to have Report Dates as columns and Clubs as rows
pivot_table = aggregation.pivot(index='Club', columns='Report Date', values='Open Orders Over 5 Weeks').fillna(0).astype(int)

# Reset index to turn 'Club' back into a column
pivot_table.reset_index(inplace=True)

# Sort the pivot table by Club name
pivot_table = pivot_table.sort_values('Club')

# Define sorted_report_dates after pivot_table is created
sorted_report_dates = sorted(pivot_table.columns[1:])  # Exclude 'Club' column

# -------------------------- Compute Total Open Orders Over 5 Weeks Old -------------------------- #

# Aggregate total open orders over 5 weeks old across all clubs per report date
total_aggregation = data[data['Order Category'] == 'Outstanding Over 5 Weeks'].dropna(subset=['Report Date']).groupby('Report Date').size().reset_index(name='Total Open Orders Over 5 Weeks')

# Sort the aggregation by report date
total_aggregation = total_aggregation.sort_values('Report Date')

# Convert report dates to string for display
total_aggregation['Report Date'] = total_aggregation['Report Date'].dt.strftime('%Y-%m-%d')

# -------------------------- Display Total Summary Table -------------------------- #

st.subheader("Total Open Orders Over 5 Weeks Old by Report Date")
st.write("This table shows the total number of open orders over five weeks old across all clubs for each report date.")

# Format the total_aggregation table for better readability
formatted_total = total_aggregation.copy()
# Apply formatting to the 'Total Open Orders Over 5 Weeks' column
formatted_total = formatted_total.style.format("{:,}", subset=['Total Open Orders Over 5 Weeks'])

st.dataframe(formatted_total)

# -------------------------- Display Summary Table -------------------------- #

st.subheader("Summary of Open Orders Over 5 Weeks Old by Club")
st.write("This table shows the number of open orders over five weeks old for each club across different report dates.")

# Format the pivot_table for better readability
formatted_pivot = pivot_table.copy()
formatted_pivot.columns = ['Club'] + [date.strftime('%Y-%m-%d') for date in formatted_pivot.columns[1:]]

# Apply formatting only to the numerical columns (all columns except 'Club')
numerical_columns = formatted_pivot.columns[1:]
st.dataframe(formatted_pivot.style.format("{:,}", subset=numerical_columns))

# -------------------------- Sidebar Filters -------------------------- #

st.sidebar.header("Filter Options")

# Selection box for club
selected_club = st.sidebar.selectbox("Select Club", options=['All Clubs'] + list(clubs))

# Selection boxes for start and end dates
selected_start_date = st.sidebar.selectbox("Select Start Date", options=report_date_strings, index=0)
selected_end_date = st.sidebar.selectbox("Select End Date", options=report_date_strings, index=len(report_date_strings)-1)

# -------------------------- Data Filtering -------------------------- #

# Convert selected dates back to datetime
start_date = pd.to_datetime(selected_start_date)
end_date = pd.to_datetime(selected_end_date)

# Ensure start_date is before end_date
if start_date > end_date:
    st.error("Start date must be before end date.")
    logger.error("Start date is after end date.")
    st.stop()

# Filter data based on user selection
if selected_club != 'All Clubs':
    data_filtered = data[data['Club'] == selected_club]
    logger.info(f"Filtered data for club: {selected_club}")
else:
    data_filtered = data.copy()
    logger.info("Filtered data for All Clubs")

# Further filter data between selected dates
data_filtered = data_filtered[(data_filtered['Report Date'] >= start_date) & (data_filtered['Report Date'] <= end_date)]
logger.info(f"Filtered data between {selected_start_date} and {selected_end_date}")

if data_filtered.empty:
    st.warning("No data available for the selected club and date range.")
    logger.warning("No data after applying filters.")
    st.stop()

# -------------------------- Analysis and Visualizations -------------------------- #

# 1. Trend of Open Orders Over 5 Weeks Old
trend_data = []
for report_date in sorted(data_filtered['Report Date'].dropna().unique()):
    df_report = data_filtered[data_filtered['Report Date'] == report_date]
    # Number of open orders over 5 weeks old
    num_open_over_5_weeks = df_report[df_report['Order Category'] == 'Outstanding Over 5 Weeks']['Order ID'].nunique()
    trend_data.append({
        'Report Date': report_date,
        'Open Orders Over 5 Weeks': num_open_over_5_weeks
    })

trend_df = pd.DataFrame(trend_data)

st.subheader("Trend of Open Orders Over 5 Weeks Old")
st.write("This graph shows the number of open or partially shipped orders over five weeks old for the selected club and time period.")

# Plot the trend using Object-Oriented Interface
try:
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(trend_df['Report Date'], trend_df['Open Orders Over 5 Weeks'], marker='o', linestyle='-')
    ax1.set_title('Number of Open Orders Over 5 Weeks Old Over Time')
    ax1.set_xlabel('Report Date')
    ax1.set_ylabel('Number of Open Orders Over 5 Weeks')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True)
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)  # Close the figure to free memory
    logger.info("Plotted Trend of Open Orders Over 5 Weeks Old")
except Exception as e:
    st.error(f"An error occurred while plotting the trend: {e}")
    logger.error(f"Error plotting trend: {e}")

# 2. Open or Partially Shipped Orders Becoming Over 5 Weeks Old Between Report Dates
st.subheader("Open or Partially Shipped Orders Becoming Over 5 Weeks Old Between Report Dates")
st.write("""
This analysis shows the number of **open or partially shipped orders** that became over five weeks old between each report date for the selected club.
""")

# Create pivot tables for each report date
pivot_tables = {}
for report_date in sorted_report_dates:
    df_report = data_filtered[data_filtered['Report Date'] == report_date]
    pivot_tables[report_date] = df_report[['Order ID', 'Order Category']].drop_duplicates().set_index('Order ID')

# Sort report dates for chronological order
sorted_report_dates = sorted(pivot_tables.keys())

# Initialize list to store changes
changes_list = []

# Loop through the report dates to find orders that became over 5 weeks old
for i in range(1, len(sorted_report_dates)):
    prev_date = sorted_report_dates[i-1]
    curr_date = sorted_report_dates[i]
    prev_pivot = pivot_tables[prev_date]
    curr_pivot = pivot_tables[curr_date]

    # Find orders that were not over 5 weeks old in prev_date but are over 5 weeks old in curr_date
    merged = prev_pivot.join(curr_pivot, lsuffix='_prev', rsuffix='_curr', how='outer')
    condition = (merged['Order Category_prev'] != 'Outstanding Over 5 Weeks') & (merged['Order Category_curr'] == 'Outstanding Over 5 Weeks')
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

# 3. Orders Shipped and New Orders Added Between Report Dates
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

    try:
        # Plot using Object-Oriented Interface
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(shipment_df['To Date'], shipment_df['Orders Shipped'], marker='o', label='Orders Shipped', linestyle='-')
        ax2.plot(shipment_df['To Date'], shipment_df['New Orders'], marker='o', label='New Orders', linestyle='-')
        ax2.set_title('Orders Shipped vs. New Orders Over Time')
        ax2.set_xlabel('To Date')
        ax2.set_ylabel('Number of Orders')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)  # Close the figure to free memory
        logger.info("Plotted Orders Shipped vs. New Orders Over Time")
    except Exception as e:
        st.error(f"An error occurred while plotting Orders Shipped vs. New Orders: {e}")
        logger.error(f"Error plotting Orders Shipped vs. New Orders: {e}")
else:
    st.write("Not enough data points to display Orders Shipped vs. New Orders.")

# 4. Top Clubs Contributing to New Over 5 Weeks Orders
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

            try:
                # Plot using Object-Oriented Interface
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                sns.barplot(x=club_counts_last_period.index[:10], y=club_counts_last_period.values[:10], ax=ax3)
                ax3.set_title(f'Top 10 Clubs Contributing to New Over 5 Weeks Orders from {start_date_str} to {end_date_str}')
                ax3.set_xlabel('Club')
                ax3.set_ylabel('Number of Orders')
                ax3.tick_params(axis='x', rotation=90)
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)  # Close the figure to free memory
                logger.info("Plotted Top Clubs Contributing to New Over 5 Weeks Orders")
            except Exception as e:
                st.error(f"An error occurred while plotting Top Clubs: {e}")
                logger.error(f"Error plotting Top Clubs: {e}")
else:
    st.subheader(f"Detailed Analysis for {selected_club}")
    st.write(f"Here is a detailed analysis for **{selected_club}** based on the selected date range.")

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

    try:
        # Plot using Object-Oriented Interface
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.plot(trend_over_time_df['Report Date'], trend_over_time_df['Outstanding Orders Over 5 Weeks'], marker='o', linestyle='-')
        ax4.set_title(f'Outstanding Orders Over 5 Weeks Old Over Time for {selected_club}')
        ax4.set_xlabel('Report Date')
        ax4.set_ylabel('Number of Outstanding Orders')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True)
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)  # Close the figure to free memory
        logger.info(f"Plotted Trend for {selected_club}")
    except Exception as e:
        st.error(f"An error occurred while plotting the trend for {selected_club}: {e}")
        logger.error(f"Error plotting trend for {selected_club}: {e}")

    # Provide an interpretation paragraph
    st.write("**Interpretation:**")
    st.write(f"""
    The data indicates changes in the number of open or partially shipped orders over five weeks old for **{selected_club}**. By analyzing the trend and comparing the number of new overdue orders with the orders shipped and new orders added, we can assess the club's order processing efficiency. An increasing trend suggests a backlog forming, while a decreasing trend indicates progress in reducing overdue orders.
    """)

# 5. Age Distribution of Open Orders Over 5 Weeks
st.subheader("Age Distribution of Open Orders Over 5 Weeks")
st.write("This histogram displays the distribution of ages (in days) of open or partially shipped orders over five weeks old.")

all_over_5_weeks = data_filtered[data_filtered['Order Category'] == 'Outstanding Over 5 Weeks'].copy()
if not all_over_5_weeks.empty:
    # Since we're not using 'Order Date' anymore, we can't calculate age in days. However, if 'Order Range' provides categorical information,
    # we can represent it differently. Alternatively, if you have another way to calculate age, implement it here.

    # If 'Order Range' doesn't provide specific age, consider removing this histogram or replacing it with another relevant visualization.

    st.write("**Note:** Since order categorization is based on 'Order Range', which is categorical, the age distribution cannot be represented numerically.")

    # Alternatively, count the number of orders per 'Order Range' category
    order_range_counts = all_over_5_weeks['Order Range'].value_counts().reset_index()
    order_range_counts.columns = ['Order Range', 'Count']

    try:
        # Plot using Object-Oriented Interface
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Order Range', y='Count', data=order_range_counts, ax=ax5)
        ax5.set_title('Count of Open Orders by Order Range')
        ax5.set_xlabel('Order Range')
        ax5.set_ylabel('Number of Orders')
        ax5.tick_params(axis='x', rotation=45)
        fig5.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)  # Close the figure to free memory
        logger.info("Plotted Order Range Distribution")
    except Exception as e:
        st.error(f"An error occurred while plotting the Order Range distribution: {e}")
        logger.error(f"Error plotting Order Range distribution: {e}")
else:
    st.write("No open orders over 5 weeks to display order range distribution.")
