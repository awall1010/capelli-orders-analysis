# # combined_order_analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from itertools import cycle
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
import logging

# -------------------------- Streamlit App Setup -------------------------- #

st.set_page_config(page_title="Order Analysis Comparison", layout="wide")

st.title("Order Analysis Comparison: 2023 vs 2025")
st.write("""
This application analyzes and compares aggregated order data from the `Capelli2023_aggregated_orders.csv` and `aggregated_orders3.23.csv` files. Explore delivery times and shipping performance across different clubs based on the order creation dates.
""")

# -------------------------- Logging Configuration -------------------------- #

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------- Data Loading Functions -------------------------- #

@st.cache_data
def load_2023_data(filepath):
    """
    Loads and preprocesses the 2023 aggregated orders data.

    Parameters:
    - filepath (str): Path to the Capelli2023_aggregated_orders.csv file.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame with unified columns.
    """
    if not os.path.exists(filepath):
        st.error(f"The file '{filepath}' does not exist in the specified directory.")
        st.stop()

    try:
        df_2023 = pd.read_csv(filepath, sep=None, engine='python')  # Auto-detect separator
        logger.info(f"Successfully loaded 2023 data from {filepath}")
    except Exception as e:
        st.error(f"Error reading the 2023 file: {e}")
        logger.error(f"Error reading the 2023 file: {e}")
        st.stop()

    # Verify required columns exist
    required_columns_2023 = [
        'order_id', 'Club Name', 'First Order Date', 'Latest Ship Date', 'Total Products Ordered'
    ]

    missing_columns_2023 = [col for col in required_columns_2023 if col not in df_2023.columns]
    if missing_columns_2023:
        st.error(f"The following required columns are missing in the 2023 CSV file: {', '.join(missing_columns_2023)}")
        logger.error(f"Missing columns in 2023 data: {missing_columns_2023}")
        st.stop()

    # Rename columns to match recent data
    df_2023 = df_2023.rename(columns={
        'order_id': 'Order ID',
        'First Order Date': 'Date Created',
        'Latest Ship Date': 'Shipping Date',
        'Total Products Ordered': 'Order Quantity'
    })

    # Parse date columns
    df_2023['Date Created'] = pd.to_datetime(df_2023['Date Created'], errors='coerce')
    df_2023['Shipping Date'] = pd.to_datetime(df_2023['Shipping Date'], errors='coerce')

    # Add Shipped Quantity and Unshipped Quantity
    df_2023['Shipped Quantity'] = df_2023['Order Quantity']
    df_2023['Unshipped Quantity'] = 0.0  # Assuming all orders are shipped in 2023 data
    df_2023['Sales Order Header Status'] = 'SHIPPED'  # Assuming all orders are shipped

    # Add Year column
    df_2023['Year'] = 2023

    # Reorder columns to match recent data
    df_2023 = df_2023[[
        'Order ID', 'Club Name', 'Date Created', 'Order Quantity',
        'Shipped Quantity', 'Unshipped Quantity', 'Shipping Date',
        'Sales Order Header Status', 'Year'
    ]]

    return df_2023

@st.cache_data
def load_recent_data(filepath):
    """
    Loads and preprocesses the recent aggregated orders data.

    Parameters:
    - filepath (str): Path to the aggregated_orders3.23.csv file.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame with unified columns.
    """
    if not os.path.exists(filepath):
        st.error(f"The file '{filepath}' does not exist in the specified directory.")
        st.stop()

    try:
        df_recent = pd.read_csv(filepath, sep=None, engine='python')  # Auto-detect separator
        logger.info(f"Successfully loaded recent data from {filepath}")
    except Exception as e:
        st.error(f"Error reading the recent file: {e}")
        logger.error(f"Error reading the recent file: {e}")
        st.stop()

    # Verify required columns exist
    required_columns_recent = [
        'Customer Reference', 'Club Name', 'Date Created', 'Order Quantity',
        'Shipped Quantity', 'Unshipped Quantity', 'Shipping Date',
        'Sales Order Header Status'
    ]

    missing_columns_recent = [col for col in required_columns_recent if col not in df_recent.columns]
    if missing_columns_recent:
        st.error(f"The following required columns are missing in the recent CSV file: {', '.join(missing_columns_recent)}")
        logger.error(f"Missing columns in recent data: {missing_columns_recent}")
        st.stop()

    # Rename columns to match 2023 data
    df_recent = df_recent.rename(columns={
        'Customer Reference': 'Order ID'
    })

    # Parse date columns
    df_recent['Date Created'] = pd.to_datetime(df_recent['Date Created'], errors='coerce')
    df_recent['Shipping Date'] = pd.to_datetime(df_recent['Shipping Date'], errors='coerce')

    # Add Year column extracted from 'Date Created'
    df_recent['Year'] = df_recent['Date Created'].dt.year

    # Reorder columns to match 2023 data
    df_recent = df_recent[[
        'Order ID', 'Club Name', 'Date Created', 'Order Quantity',
        'Shipped Quantity', 'Unshipped Quantity', 'Shipping Date',
        'Sales Order Header Status', 'Year'
    ]]

    return df_recent

@st.cache_data
def combine_data(df_2023, df_recent):
    """
    Combines 2023 and recent data into a single DataFrame.

    Parameters:
    - df_2023 (pd.DataFrame): 2023 orders DataFrame.
    - df_recent (pd.DataFrame): Recent orders DataFrame.

    Returns:
    - pd.DataFrame: Combined DataFrame.
    """
    combined_df = pd.concat([df_2023, df_recent], ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")
    return combined_df

# -------------------------- Data Loading -------------------------- #

# Load the data
data_file_2023 = os.path.join('shippingdates', 'Capelli2023_aggregated_orders.csv')
data_file_recent = os.path.join('shippingdates', 'aggregated_order3.23.csv')

df_2023 = load_2023_data(data_file_2023)
df_recent = load_recent_data(data_file_recent)

# Combine datasets
df_combined = combine_data(df_2023, df_recent)

# -------------------------- Data Preprocessing -------------------------- #

# Drop records with Order ID "1320579" if present in recent data
initial_count = df_combined.shape[0]
df_combined = df_combined[df_combined['Order ID'] != "1320579"]
logger.info(f"Dropped {initial_count - df_combined.shape[0]} records with Order ID '1320579'")

# Ensure numerical columns are correctly typed
numeric_cols = ['Order Quantity', 'Shipped Quantity', 'Unshipped Quantity']
for col in numeric_cols:
    df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

# Handle missing values
# For 'Sales Order Header Status', fill NaN with 'UNKNOWN'
df_combined['Sales Order Header Status'] = df_combined['Sales Order Header Status'].fillna('UNKNOWN')

# Fill NaN in numerical columns with 0
df_combined[numeric_cols] = df_combined[numeric_cols].fillna(0)

# Define all statuses that indicate a closed/shipped order
closed_statuses = ['CLOSED', 'COMPLETED', 'SHIPPED', 'CLOSE']

# Standardize 'Sales Order Header Status' to uppercase and strip whitespace
df_combined['Sales Order Header Status'] = df_combined['Sales Order Header Status'].str.upper().str.strip()

# Create a boolean column to indicate if an order is closed
df_combined['Is Closed'] = df_combined['Sales Order Header Status'].isin(closed_statuses)

# Calculate 'Time to Ship' only for closed orders
df_combined['Time to Ship'] = np.where(
    df_combined['Is Closed'],
    (df_combined['Shipping Date'] - df_combined['Date Created']).dt.days,
    np.nan  # Not applicable for OPEN orders
)

# Calculate 'Order Age' as the number of days since 'Date Created' up to today
df_combined['Order Age'] = (pd.to_datetime(datetime.today().date()) - df_combined['Date Created']).dt.days

# Determine 'Over 5 weeks?' based on:
# - For CLOSED orders: 'Time to Ship' > 35 days
# - For OPEN orders: 'Order Age' > 35 days
df_combined['Over 5 weeks?'] = np.where(
    (df_combined['Is Closed'] & (df_combined['Time to Ship'] > 35)) |
    (~df_combined['Is Closed'] & (df_combined['Order Age'] > 35)),
    'Over 5 weeks',
    'Under 5 weeks'
)

# Drop records with negative 'Time to Ship' or 'Order Age' (if any)
df_combined = df_combined[
    ((df_combined['Is Closed'] & (df_combined['Time to Ship'] >= 0)) |
     (~df_combined['Is Closed'] & (df_combined['Order Age'] >= 0)))
]

logger.info(f"After preprocessing, total records: {df_combined.shape[0]}")

# Display data types after preprocessing
logger.info("Data Types After Preprocessing:\n" + str(df_combined.dtypes))
logger.info("First 5 Rows After Preprocessing:\n" + str(df_combined.head()))

# -------------------------- Sidebar Filters -------------------------- #

st.sidebar.header("Filter Options")

# Dropdown for selecting year(s)
years_available = sorted(df_combined['Year'].dropna().unique())
selected_years = st.sidebar.multiselect("Select Year(s)", options=years_available, default=years_available)

# Dropdown for selecting a club
clubs = df_combined['Club Name'].unique()
clubs_sorted = sorted(clubs)
selected_club = st.sidebar.selectbox("Select Club", options=['All Clubs'] + list(clubs_sorted))

# If a specific club is selected, filter the DataFrame
if selected_club != 'All Clubs':
    filtered_df = df_combined[(df_combined['Club Name'] == selected_club) & (df_combined['Year'].isin(selected_years))]
    logger.info(f"Selected Club: {selected_club} | Years: {selected_years} | Records: {filtered_df.shape[0]}")
else:
    filtered_df = df_combined[df_combined['Year'].isin(selected_years)]
    logger.info(f"Selected Club: All Clubs | Years: {selected_years} | Records: {filtered_df.shape[0]}")

# -------------------------- Metrics and Visualizations -------------------------- #

st.subheader("Outstanding Orders Metrics")
st.write("""
This section displays the total number of outstanding orders and the percentage of these orders that are over 35 days old.
""")

# Filter for outstanding orders (OPEN)
outstanding_df = filtered_df[filtered_df['Sales Order Header Status'] == 'OPEN'].copy()
logger.info(f"Outstanding Orders: {outstanding_df.shape[0]}")

# Calculate Total Outstanding Orders
total_outstanding = outstanding_df.shape[0]

# Calculate Number of Outstanding Orders Over 35 Days
outstanding_over_35_days = outstanding_df[outstanding_df['Order Age'] > 35].shape[0]

# Calculate Percentage of Outstanding Orders Over 35 Days
percent_over_35_days = (outstanding_over_35_days / total_outstanding * 100) if total_outstanding > 0 else 0.0

# Display Metrics in Three Columns with Toggle Buttons
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Outstanding Orders",
        value=f"{total_outstanding:,}"
    )
    if st.button("Show Details - Total Outstanding", key="toggle_total"):
        st.write("### Details: Total Outstanding Orders")
        st.dataframe(outstanding_df.reset_index(drop=True))
        if st.button("Hide Details - Total Outstanding", key="hide_total"):
            st.experimental_rerun()

with col2:
    st.metric(
        label="Outstanding Orders Over 5 Weeks",
        value=f"{outstanding_over_35_days:,}"
    )
    if st.button("Show Details - Over 5 Weeks", key="toggle_over_5_weeks"):
        st.write("### Details: Outstanding Orders Over 5 Weeks")
        st.dataframe(outstanding_df[outstanding_df['Order Age'] > 35].reset_index(drop=True))
        if st.button("Hide Details - Over 5 Weeks", key="hide_over_5_weeks"):
            st.experimental_rerun()

with col3:
    st.metric(
        label="% of Outstanding Orders Over 5 Weeks",
        value=f"{percent_over_35_days:.2f}%"
    )
    if st.button("Show Details - Percentage Over 5 Weeks", key="toggle_percent_over_5_weeks"):
        st.write("### Details: Percentage of Outstanding Orders Over 5 Weeks")
        st.write(f"**Total Outstanding Orders:** {total_outstanding:,}")
        st.write(f"**Outstanding Orders Over 5 Weeks:** {outstanding_over_35_days:,}")
        st.write(f"**Percentage:** {percent_over_35_days:.2f}%")
        st.write("#### List of Orders Over 5 Weeks:")
        st.dataframe(outstanding_df[outstanding_df['Order Age'] > 35].reset_index(drop=True))
        if st.button("Hide Details - Percentage Over 5 Weeks", key="hide_percent_over_5_weeks"):
            st.experimental_rerun()

# -------------------------- Define 'closed_orders_df' Outside Conditional -------------------------- #

# Ensure 'closed_orders_df' is defined before any sections that use it
closed_orders_df = filtered_df[filtered_df['Is Closed']].copy()
logger.info(f"Closed Orders: {closed_orders_df.shape[0]}")

# -------------------------- Shipping Time for Each Order Over Time -------------------------- #

st.subheader("Shipping Time for Each Order Over Time")
st.write("""
This graph shows how long it took for each order to ship, based on the order creation date over time.
""")

# Ensure there are shipped orders to plot
if not closed_orders_df['Time to Ship'].isna().all():
    shipping_time_df = closed_orders_df.dropna(subset=['Date Created', 'Time to Ship'])

    if shipping_time_df.empty:
        st.write("No shipping time data available to display.")
    else:
        # Plotting
        fig_shipping_time = px.scatter(
            shipping_time_df,
            x='Date Created',
            y='Time to Ship',
            color='Club Name',
            title=f'Shipping Time Over Time for {selected_club}' if selected_club != 'All Clubs' else 'Shipping Time Over Time for All Clubs',
            labels={
                'Date Created': 'Order Creation Date',
                'Time to Ship': 'Shipping Time (days)',
                'Club Name': 'Club'
            },
            hover_data=['Order ID', 'Order Quantity', 'Shipped Quantity']
        )

        # Add a dashed line at 35 days to indicate the 5-week cutoff
        fig_shipping_time.add_shape(
            dict(
                type="line",
                x0=shipping_time_df['Date Created'].min(),
                y0=35,
                x1=shipping_time_df['Date Created'].max(),
                y1=35,
                line=dict(color="Red", width=2, dash="dash"),
            )
        )

        fig_shipping_time.update_layout(
            xaxis=dict(tickangle=45),
            yaxis=dict(title='Shipping Time (days)'),
            template='plotly_white',
            legend_title_text='Club'
        )

        st.plotly_chart(fig_shipping_time, use_container_width=True)
else:
    st.write("No shipped orders available to display the shipping time graph.")

# -------------------------- Count of Orders Over 5 Weeks (Month Over Month) -------------------------- #

st.subheader("Count of Orders Over 5 Weeks (Month Over Month)")
st.write("""
This table shows the count of orders that took over 5 weeks to ship for each month based on the order creation date.
""")

# Ensure there are shipped orders to perform count analysis
if not closed_orders_df['Time to Ship'].isna().all():
    count_df = closed_orders_df.dropna(subset=['Date Created', 'Time to Ship']).copy()

    if count_df.empty:
        st.write("No shipping time data available to display count analysis.")
    else:
        # Extract Month-Year from 'Date Created' in 'YYYY-MM' format
        count_df['Month'] = count_df['Date Created'].dt.to_period('M').astype(str)  # '2023-01'

        # Group by Month-Year
        grouped_counts = count_df.groupby('Month')

        # Calculate orders shipped under 5 weeks
        under_5_weeks_counts = grouped_counts.apply(lambda x: (x['Over 5 weeks?'] == 'Under 5 weeks').sum()).reset_index(name='Orders Shipped Under 5 Weeks')

        # Calculate orders shipped over 5 weeks
        over_5_weeks_counts = grouped_counts.apply(lambda x: (x['Over 5 weeks?'] == 'Over 5 weeks').sum()).reset_index(name='Orders Shipped Over 5 Weeks')

        # Merge the counts
        counts_summary = pd.merge(under_5_weeks_counts, over_5_weeks_counts, on='Month', how='outer')

        # Fill NaN values with 0 for numeric columns
        counts_summary[['Orders Shipped Under 5 Weeks', 'Orders Shipped Over 5 Weeks']] = counts_summary[['Orders Shipped Under 5 Weeks', 'Orders Shipped Over 5 Weeks']].fillna(0)

        # Ensure correct data types
        counts_summary['Orders Shipped Under 5 Weeks'] = counts_summary['Orders Shipped Under 5 Weeks'].astype(int)
        counts_summary['Orders Shipped Over 5 Weeks'] = counts_summary['Orders Shipped Over 5 Weeks'].astype(int)
        counts_summary['Total Orders Shipped'] = counts_summary['Orders Shipped Under 5 Weeks'] + counts_summary['Orders Shipped Over 5 Weeks']

        # Sort by Month-Year
        counts_summary['Month'] = pd.to_datetime(counts_summary['Month'], format='%Y-%m')
        counts_summary = counts_summary.sort_values('Month')
        counts_summary['Month'] = counts_summary['Month'].dt.strftime('%Y-%m')

        # Select and reorder columns
        counts_table = counts_summary[['Month', 'Orders Shipped Under 5 Weeks', 'Orders Shipped Over 5 Weeks']]

        # Calculate grand total
        total_under_counts = counts_summary['Orders Shipped Under 5 Weeks'].sum()
        total_over_counts = counts_summary['Orders Shipped Over 5 Weeks'].sum()

        grand_total_counts = pd.DataFrame([{
            'Month': 'Grand Total',
            'Orders Shipped Under 5 Weeks': total_under_counts,
            'Orders Shipped Over 5 Weeks': total_over_counts
        }])

        counts_table = pd.concat([counts_table, grand_total_counts], ignore_index=True)

        # Define a function to highlight the grand total row
        def highlight_grand_total_counts(row):
            if row['Month'] == 'Grand Total':
                return ['background-color: lightblue'] * len(row)
            else:
                return [''] * len(row)

        # Apply the styling
        styled_counts_table = counts_table.style.format({
            'Orders Shipped Under 5 Weeks': "{:,}",
            'Orders Shipped Over 5 Weeks': "{:,}"
        }).apply(highlight_grand_total_counts, axis=1).set_properties(**{
            'text-align': 'center'
        }).set_table_styles([
            dict(selector='th', props=[('text-align', 'center')]),
            dict(selector='td', props=[('text-align', 'center')])
        ])

        # -------------------------- Display the Counts Table -------------------------- #

        st.dataframe(styled_counts_table, use_container_width=True)

        # -------------------------- Download Button for Count Table -------------------------- #

        # Convert the DataFrame to CSV
        csv_counts = counts_table.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Count Table as CSV",
            data=csv_counts,
            file_name='count_orders_over_5_weeks.csv',
            mime='text/csv',
        )
else:
    st.write("No shipped orders available to display the count of orders over 5 weeks.")

# -------------------------- Percentage of Orders Over 5 Weeks (Month Over Month) -------------------------- #

st.subheader("Percentage of Orders Over 5 Weeks (Month Over Month)")
st.write("""
This table shows the percentage of orders that took over 5 weeks to ship for each month based on the order creation date.
""")

# Ensure there are shipped orders to calculate percentages
if not closed_orders_df['Time to Ship'].isna().all():
    percentage_df = closed_orders_df.dropna(subset=['Date Created', 'Time to Ship']).copy()

    if percentage_df.empty:
        st.write("No shipping time data available to display percentage analysis.")
    else:
        # Extract Creation Month as 'YYYY-MM' string
        percentage_df['Creation Month'] = percentage_df['Date Created'].dt.to_period('M').dt.strftime('%Y-%m')

        # Group by Creation Month
        grouped = percentage_df.groupby('Creation Month')

        # Calculate orders shipped under 5 weeks
        under_5_weeks = grouped.apply(lambda x: (x['Over 5 weeks?'] == 'Under 5 weeks').sum()).reset_index(name='Orders Shipped Under 5 Weeks')

        # Calculate orders shipped over 5 weeks
        over_5_weeks = grouped.apply(lambda x: (x['Over 5 weeks?'] == 'Over 5 weeks').sum()).reset_index(name='Orders Shipped Over 5 Weeks')

        # Merge the summaries
        percentage_summary = pd.merge(under_5_weeks, over_5_weeks, on='Creation Month', how='outer')
        percentage_summary['Total Orders Shipped'] = percentage_summary['Orders Shipped Under 5 Weeks'] + percentage_summary['Orders Shipped Over 5 Weeks']

        # Calculate percentages
        percentage_summary['% Of Orders Shipped under 5 weeks'] = (
            percentage_summary['Orders Shipped Under 5 Weeks'] / percentage_summary['Total Orders Shipped'] * 100
        ).round(2)

        percentage_summary['% Of Orders Shipped over 5 weeks'] = (
            percentage_summary['Orders Shipped Over 5 Weeks'] / percentage_summary['Total Orders Shipped'] * 100
        ).round(2)

        # Format Creation Month as string for better display
        percentage_summary['Creation Month'] = percentage_summary['Creation Month'].astype(str)

        # Rename columns for clarity
        percentage_summary.rename(columns={'Creation Month': 'Month'}, inplace=True)

        # -------------------------- Ensure All Months are Present -------------------------- #

        # Determine the range of months from '2023-01' to the latest month in the data
        min_month = '2023-01'
        max_month = percentage_summary['Month'].max()
        all_months = pd.period_range(start=min_month, end=max_month, freq='M').astype(str)

        # Create a DataFrame with all months
        all_months_df = pd.DataFrame({'Month': all_months})

        # Merge with the percentage_summary to include all months
        percentage_summary_complete = pd.merge(all_months_df, percentage_summary, on='Month', how='left')

        # Fill NaN values with 0 for months with no data
        percentage_summary_complete[['Orders Shipped Under 5 Weeks', 'Orders Shipped Over 5 Weeks', 'Total Orders Shipped']] = percentage_summary_complete[['Orders Shipped Under 5 Weeks', 'Orders Shipped Over 5 Weeks', 'Total Orders Shipped']].fillna(0)

        # Recalculate percentages to handle division by zero
        percentage_summary_complete['% Of Orders Shipped under 5 weeks'] = np.where(
            percentage_summary_complete['Total Orders Shipped'] > 0,
            (percentage_summary_complete['Orders Shipped Under 5 Weeks'] / percentage_summary_complete['Total Orders Shipped'] * 100).round(2),
            0.0
        )

        percentage_summary_complete['% Of Orders Shipped over 5 weeks'] = np.where(
            percentage_summary_complete['Total Orders Shipped'] > 0,
            (percentage_summary_complete['Orders Shipped Over 5 Weeks'] / percentage_summary_complete['Total Orders Shipped'] * 100).round(2),
            0.0
        )

        # Select and reorder columns
        percentage_table = percentage_summary_complete[['Month', '% Of Orders Shipped under 5 weeks', '% Of Orders Shipped over 5 weeks']]

        # Calculate grand total
        total_under = percentage_summary_complete['Orders Shipped Under 5 Weeks'].sum()
        total_over = percentage_summary_complete['Orders Shipped Over 5 Weeks'].sum()
        total = percentage_summary_complete['Total Orders Shipped'].sum()

        if total > 0:
            total_under_pct = (total_under / total * 100).round(2)
            total_over_pct = (total_over / total * 100).round(2)
        else:
            total_under_pct = 0.0
            total_over_pct = 0.0

        grand_total = pd.DataFrame([{
            'Month': 'Grand Total',
            '% Of Orders Shipped under 5 weeks': total_under_pct,
            '% Of Orders Shipped over 5 weeks': total_over_pct
        }])

        percentage_table = pd.concat([percentage_table, grand_total], ignore_index=True)

        # Define a function to highlight the grand total row
        def highlight_grand_total(row):
            if row['Month'] == 'Grand Total':
                return ['background-color: lightblue'] * len(row)
            else:
                return [''] * len(row)

        # Apply the styling
        styled_percentage_table = percentage_table.style.format({
            '% Of Orders Shipped under 5 weeks': "{:.2f}%",
            '% Of Orders Shipped over 5 weeks': "{:.2f}%"
        }).apply(highlight_grand_total, axis=1).set_properties(**{
            'text-align': 'center'
        }).set_table_styles([
            dict(selector='th', props=[('text-align', 'center')]),
            dict(selector='td', props=[('text-align', 'center')])
        ])

        # -------------------------- Display the Percentage Table -------------------------- #

        st.dataframe(styled_percentage_table, use_container_width=True)

        # -------------------------- Download Button for Percentage Table -------------------------- #

        # Convert the DataFrame to CSV
        csv_percentage = percentage_table.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Percentage Table as CSV",
            data=csv_percentage,
            file_name='percentage_orders_over_5_weeks.csv',
            mime='text/csv',
        )
else:
    st.write("No shipped orders available to display the percentage of orders over 5 weeks.")

# -------------------------- Percentage and Count of Orders Shipped Within 5 Weeks per Club -------------------------- #

if 2023 in selected_years or 2025 in selected_years:
    st.subheader("Percentage and Count of Orders Shipped Within 5 Weeks per Club (Month Over Month)")
    st.write("""
    This table displays each club and, for each month based on the order creation date, the percentage and count of orders shipped within and over 5 weeks.
    """)

    # Ensure there are shipped orders to calculate percentages
    if not closed_orders_df['Time to Ship'].isna().all():
        within_5_weeks_df = closed_orders_df.dropna(subset=['Date Created', 'Time to Ship']).copy()

        if within_5_weeks_df.empty:
            st.write("No shipping time data available to display percentage and count analysis.")
        else:
            # Extract Creation Month as 'YYYY-MM' string
            within_5_weeks_df['Creation Month'] = within_5_weeks_df['Date Created'].dt.to_period('M').dt.strftime('%Y-%m')

            # Group by Club and Creation Month
            grouped = within_5_weeks_df.groupby(['Club Name', 'Creation Month'])

            # Calculate total orders per club per month
            total_orders = grouped.size().reset_index(name='Total Orders Shipped')

            # Calculate orders shipped under 5 weeks
            under_5_weeks = grouped.apply(lambda x: (x['Over 5 weeks?'] == 'Under 5 weeks').sum()).reset_index(name='Orders Shipped Under 5 Weeks')

            # Calculate orders shipped over 5 weeks
            over_5_weeks = grouped.apply(lambda x: (x['Over 5 weeks?'] == 'Over 5 weeks').sum()).reset_index(name='Orders Shipped Over 5 Weeks')

            # Merge the summaries
            within_summary = pd.merge(total_orders, under_5_weeks, on=['Club Name', 'Creation Month'])
            within_summary = pd.merge(within_summary, over_5_weeks, on=['Club Name', 'Creation Month'])

            # Calculate percentages
            within_summary['% Shipped Within 5 Weeks'] = (
                within_summary['Orders Shipped Under 5 Weeks'] / within_summary['Total Orders Shipped'] * 100
            ).round(2)

            within_summary['% Shipped Over 5 Weeks'] = (
                within_summary['Orders Shipped Over 5 Weeks'] / within_summary['Total Orders Shipped'] * 100
            ).round(2)

            # Pivot the table to have months as columns
            pivot_within = within_summary.pivot(index='Club Name', columns='Creation Month', values=[
                '% Shipped Within 5 Weeks',
                '% Shipped Over 5 Weeks',
                'Orders Shipped Under 5 Weeks',
                'Orders Shipped Over 5 Weeks'
            ])

            # Flatten MultiIndex columns
            pivot_within.columns = [f"{month} {metric}" for metric, month in pivot_within.columns]
            pivot_within.reset_index(inplace=True)

            # Fill NaN with 0 (if any)
            pivot_within = pivot_within.fillna(0)

            # Define a list of subdued colors for the months (Optional)
            color_list = [
                '#D3D3D3',  # LightGray
                '#B0C4DE',  # LightSteelBlue
                '#98FB98',  # PaleGreen
                '#FFFACD',  # LemonChiffon
                '#E6E6FA',  # Lavender
                '#FFDAB9',  # PeachPuff
                '#E0E68C',  # Khaki
                '#AFEEEE',  # PaleTurquoise
                '#FFDEAD',  # NavajoWhite
                '#E0FFFF',  # LightCyan
                '#F5DEB3',  # Wheat
                '#FFF9C4',  # LightGoldenrodYellow
                '#FFE4E1',  # MistyRose
                '#F0FFF0',  # Honeydew
                '#FFF0F5',  # LavenderBlush
                '#F8F8FF',  # GhostWhite
                '#FFEBCD',  # BlanchedAlmond
                '#F5F5DC',  # Beige
                '#FFEFD5',  # PapayaWhip
                '#F0FFF0',  # Honeydew
                '#FAFAD2',  # LightGoldenrodYellow
                '#FFF5EE',  # Seashell
                '#FDF5E6',  # OldLace
                '#FFF8DC',  # Cornsilk
                '#F0FFFF',  # Azure
                '#FFF0F5'   # LavenderBlush
            ]

            # Assign colors to each month
            months = sorted(within_5_weeks_df['Creation Month'].unique())
            color_cycle = cycle(color_list)
            month_colors = {month: next(color_cycle) for month in months}

            # Create a dictionary to map each column to its corresponding color based on month
            column_color_mapping = {}
            for month in months:
                column_color_mapping[f"{month} % Shipped Within 5 Weeks"] = month_colors[month]
                column_color_mapping[f"{month} % Shipped Over 5 Weeks"] = month_colors[month]
                column_color_mapping[f"{month} Orders Shipped Under 5 Weeks"] = month_colors[month]
                column_color_mapping[f"{month} Orders Shipped Over 5 Weeks"] = month_colors[month]

            # Define a function to apply background color based on column
            def highlight_columns(row):
                styles = []
                for col in row.index:
                    if col == 'Club Name':
                        # Apply distinct style for 'Club Name'
                        styles.append('background-color: #f2f2f2; color: black; text-align: center; font-weight: bold;')
                    else:
                        # Apply color based on month
                        bg_color = column_color_mapping.get(col, '')
                        if 'Shipped Within' in col or 'Shipped Over' in col:
                            styles.append(f'background-color: {bg_color}; color: black; text-align: center;')
                        elif 'Orders Shipped' in col:
                            styles.append(f'background-color: {bg_color}; color: black; text-align: center;')
                        else:
                            styles.append('')
                return styles

            # Apply the styling
            styled_pivot_within = pivot_within.style.format({
                **{col: "{:.2f}%" for col in pivot_within.columns if 'Shipped Within' in col or 'Shipped Over' in col},
                **{col: "{:,}" for col in pivot_within.columns if 'Orders Shipped' in col}
            }).apply(highlight_columns, axis=1)

            # -------------------------- Display the Table -------------------------- #

            st.dataframe(styled_pivot_within, use_container_width=True)

            # -------------------------- Download Button -------------------------- #

            # Convert the DataFrame to CSV
            csv = pivot_within.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Table as CSV",
                data=csv,
                file_name='percentage_count_orders_shipped_within_5_weeks_per_club.csv',
                mime='text/csv',
            )
    else:
        st.write("Please select at least one year (2023 or 2025) to view the percentage and count of orders shipped within 5 weeks.")
else:
    st.write("Please select at least one year (2023 or 2025) to view the percentage and count of orders shipped within 5 weeks.")

# -------------------------- Shipping Time Over Time Comparison -------------------------- #

st.subheader("Shipping Time Over Time Comparison")
st.write("""
This graph overlays the shipping times over time for the selected year(s), allowing you to compare performance across different years.
""")

# Ensure there are shipped orders to plot
if not closed_orders_df['Time to Ship'].isna().all():
    shipping_time_df = closed_orders_df.dropna(subset=['Date Created', 'Time to Ship']).copy()

    if shipping_time_df.empty:
        st.write("No shipping time data available to display.")
    else:
        # Extract Month from 'Date Created'
        shipping_time_df['Month'] = shipping_time_df['Date Created'].dt.month_name()

        # Ensure months are ordered correctly
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        shipping_time_df['Month'] = pd.Categorical(shipping_time_df['Month'], categories=month_order, ordered=True)

        # Group by 'Month' and 'Year' to get average shipping time
        avg_shipping_time = shipping_time_df.groupby(['Month', 'Year'])['Time to Ship'].mean().reset_index()

        # Sort by Month order
        avg_shipping_time['Month'] = pd.Categorical(avg_shipping_time['Month'], categories=month_order, ordered=True)
        avg_shipping_time = avg_shipping_time.sort_values('Month')

        fig_shipping_time_overlay = px.line(
            avg_shipping_time,
            x='Month',
            y='Time to Ship',
            color='Year',
            markers=True,
            title='Average Shipping Time by Month and Year',
            labels={
                'Month': 'Month',
                'Time to Ship': 'Average Shipping Time (days)'
            }
        )

        # Add a dashed line at 35 days to indicate the 5-week cutoff
        fig_shipping_time_overlay.add_shape(
            dict(
                type="line",
                x0=-0.5,  # Starting before the first month
                y0=35,
                x1=11.5,  # Ending after the last month
                y1=35,
                xref='x',
                yref='y',
                line=dict(color="Red", width=2, dash="dash"),
            )
        )

        fig_shipping_time_overlay.update_layout(
            xaxis=dict(tickangle=45, categoryorder='array', categoryarray=month_order),
            yaxis=dict(title='Average Shipping Time (days)'),
            template='plotly_white',
            legend_title_text='Year'
        )

        st.plotly_chart(fig_shipping_time_overlay, use_container_width=True)

        # -------------------------- Download Button for Shipping Time Overlay -------------------------- #

        # Convert the DataFrame to CSV
        csv_shipping_time_overlay = avg_shipping_time.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Shipping Time Overlay Data as CSV",
            data=csv_shipping_time_overlay,
            file_name='shipping_time_overlay.csv',
            mime='text/csv',
        )
else:
    st.write("No shipped orders available to display the shipping time graph.")

# -------------------------- Number of Orders Shipped Over Time -------------------------- #

st.subheader("Number of Orders Shipped Over Time")
st.write("""
This graph shows the number of orders shipped each month based on the order creation date.
""")

# Ensure there are shipped orders to plot
if not closed_orders_df['Time to Ship'].isna().all():
    orders_count_df = closed_orders_df.dropna(subset=['Date Created']).copy()

    if orders_count_df.empty:
        st.write("No shipping data available to display orders count over time.")
    else:
        # Extract Month as month name
        orders_count_df['Month'] = orders_count_df['Date Created'].dt.month_name()

        # Extract Year
        orders_count_df['Year'] = orders_count_df['Date Created'].dt.year

        # Group by Month and Year and count orders
        orders_per_month_year = orders_count_df.groupby(['Month', 'Year']).size().reset_index(name='Number of Orders Shipped')

        # Ensure months are ordered correctly
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        orders_per_month_year['Month'] = pd.Categorical(orders_per_month_year['Month'], categories=month_order, ordered=True)

        # Sort by Year and Month
        orders_per_month_year = orders_per_month_year.sort_values(['Year', 'Month'])

        # Plotting
        fig_orders_over_time = px.line(
            orders_per_month_year,
            x='Month',
            y='Number of Orders Shipped',
            color='Year',
            title='Number of Orders Shipped Over Time',
            labels={
                'Month': 'Month',
                'Number of Orders Shipped': 'Number of Orders Shipped'
            },
            markers=True
        )

        fig_orders_over_time.update_layout(
            xaxis=dict(tickangle=45),
            yaxis=dict(title='Number of Orders Shipped'),
            template='plotly_white',
            legend_title_text='Year'
        )

        st.plotly_chart(fig_orders_over_time, use_container_width=True)

        # -------------------------- Download Button for Orders Over Time -------------------------- #

        # Convert the DataFrame to CSV
        csv_orders_over_time = orders_per_month_year.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Orders Over Time Data as CSV",
            data=csv_orders_over_time,
            file_name='orders_shipped_over_time.csv',
            mime='text/csv',
        )
else:
    st.write("No shipped orders available to display orders count over time.")

# -------------------------- Average Shipping Time per Month -------------------------- #

st.subheader("Average Shipping Time per Month")
st.write("""
This section provides both a box plot and a bar chart showing the distribution and average of shipping times per month based on the order creation date.
""")

# Calculate average shipping time per month
if not closed_orders_df['Time to Ship'].isna().all():
    avg_shipping_time_month = closed_orders_df.copy()
    avg_shipping_time_month['Creation Month'] = avg_shipping_time_month['Date Created'].dt.to_period('M').dt.strftime('%Y-%m')

    # Box Plot
    fig_box = px.box(
        avg_shipping_time_month,
        x='Creation Month',
        y='Time to Ship',
        title='Distribution of Shipping Times per Month',
        labels={
            'Creation Month': 'Month',
            'Time to Ship': 'Shipping Time (days)'
        },
        points='all'
    )

    fig_box.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(title='Shipping Time (days)'),
        template='plotly_white'
    )

    # Bar Chart
    avg_shipping_time = avg_shipping_time_month.groupby('Creation Month')['Time to Ship'].mean().reset_index()

    fig_bar = px.bar(
        avg_shipping_time,
        x='Creation Month',
        y='Time to Ship',
        title='Average Shipping Time per Month',
        labels={
            'Creation Month': 'Month',
            'Time to Ship': 'Average Shipping Time (days)'
        },
        text='Time to Ship'
    )

    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar.update_layout(
        xaxis=dict(tickangle=45),
        yaxis=dict(title='Average Shipping Time (days)'),
        template='plotly_white'
    )

    # Display the plots side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_box, use_container_width=True)
    with col2:
        st.plotly_chart(fig_bar, use_container_width=True)

    # -------------------------- Download Button for Average Shipping Time -------------------------- #

    # Convert the DataFrame to CSV
    csv_avg_shipping_time = avg_shipping_time.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Average Shipping Time Data as CSV",
        data=csv_avg_shipping_time,
        file_name='average_shipping_time_per_month.csv',
        mime='text/csv',
    )
else:
    st.write("No shipped orders available to display average shipping time per month.")

# -------------------------- Comparison of Orders and Average Shipping Time Over Time -------------------------- #

st.subheader("Comparison of Orders and Average Shipping Time Over Time")
st.write("""
This visualization compares the total number of orders placed each month with the average shipping time for those orders. It helps in understanding how order volumes influence shipping performance.
""")

# Ensure there are shipped orders to perform comparison analysis
if not closed_orders_df['Time to Ship'].isna().all():
    comparison_df = closed_orders_df.copy()
    comparison_df['Creation Month'] = comparison_df['Date Created'].dt.to_period('M').dt.strftime('%Y-%m')

    # Group by Creation Month to get total orders and average shipping time
    aggregated_comparison = comparison_df.groupby('Creation Month').agg(
        Total_Orders_Placed=('Order ID', 'count'),
        Average_Shipping_Time=('Time to Ship', 'mean')
    ).reset_index()

    # Create Plotly figure with dual y-axes
    fig_comparison = make_subplots(specs=[[{"secondary_y": True}]])

    fig_comparison.add_trace(
        go.Bar(
            x=aggregated_comparison['Creation Month'],
            y=aggregated_comparison['Total_Orders_Placed'],
            name='Total Orders Placed',
            marker_color='indianred'
        ),
        secondary_y=False,
    )

    fig_comparison.add_trace(
        go.Scatter(
            x=aggregated_comparison['Creation Month'],
            y=aggregated_comparison['Average_Shipping_Time'],
            name='Average Shipping Time (days)',
            mode='lines+markers',
            marker_color='royalblue'
        ),
        secondary_y=True,
    )

    # Add figure title
    fig_comparison.update_layout(
        title_text="Total Orders Placed and Average Shipping Time Over Time"
    )

    # Set x-axis title
    fig_comparison.update_xaxes(title_text="Month")

    # Set y-axes titles
    fig_comparison.update_yaxes(title_text="Total Orders Placed", secondary_y=False)
    fig_comparison.update_yaxes(title_text="Average Shipping Time (days)", secondary_y=True)

    # Display the figure
    st.plotly_chart(fig_comparison, use_container_width=True)

    # -------------------------- Download Button for Comparison Data -------------------------- #

    # Convert the aggregated comparison DataFrame to CSV
    csv_comparison = aggregated_comparison.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Comparison Data as CSV",
        data=csv_comparison,
        file_name='orders_and_shipping_time_comparison.csv',
        mime='text/csv',
    )
else:
    st.write("No shipped orders available to perform comparison analysis.")

# -------------------------- Correlation Between Number of Orders Shipped and Shipping Time -------------------------- #

st.subheader("Correlation Between Number of Orders Shipped and Shipping Time")
st.write("""
This analysis examines whether there's a relationship between the number of orders shipped in a month and the average shipping time for those orders. Understanding this correlation can help in forecasting shipping performance based on order volumes.
""")

# Ensure there are shipped orders to perform correlation analysis
if not closed_orders_df['Time to Ship'].isna().all():
    correlation_df = closed_orders_df.dropna(subset=['Date Created', 'Time to Ship']).copy()

    if correlation_df.empty:
        st.write("No shipping time data available to perform correlation analysis.")
    else:
        # Extract Creation Month as 'YYYY-MM' string
        correlation_df['Creation Month'] = correlation_df['Date Created'].dt.to_period('M').dt.strftime('%Y-%m')

        # Group by Creation Month to get total orders and average shipping time
        correlation_summary = correlation_df.groupby('Creation Month').agg(
            Total_Orders_Shipped=('Order ID', 'count'),
            Average_Shipping_Time=('Time to Ship', 'mean')
        ).reset_index()

        # Calculate Pearson correlation coefficient
        correlation_coefficient = correlation_summary['Total_Orders_Shipped'].corr(correlation_summary['Average_Shipping_Time'])

        # Display the correlation coefficient
        st.markdown(f"**Pearson Correlation Coefficient:** {correlation_coefficient:.2f}")

        # Plotting the correlation
        fig_correlation = px.scatter(
            correlation_summary,
            x='Total_Orders_Shipped',
            y='Average_Shipping_Time',
            trendline="ols",
            title="Correlation Between Total Orders Shipped and Average Shipping Time",
            labels={
                'Total_Orders_Shipped': 'Total Orders Shipped per Month',
                'Average_Shipping_Time': 'Average Shipping Time (days)'
            },
            hover_data=['Creation Month']
        )

        fig_correlation.update_layout(
            template='plotly_white'
        )

        st.plotly_chart(fig_correlation, use_container_width=True)

        # -------------------------- Interpretation -------------------------- #

        if correlation_coefficient > 0:
            interpretation = "a positive correlation", "as the number of orders shipped increases, the average shipping time also tends to increase."
        elif correlation_coefficient < 0:
            interpretation = "a negative correlation", "as the number of orders shipped increases, the average shipping time tends to decrease."
        else:
            interpretation = "no correlation", "there is no apparent relationship between the number of orders shipped and the average shipping time."

        st.write(f"**Interpretation:** There is {interpretation[0]} between the number of orders shipped and the average shipping time. Specifically, {interpretation[1]}")

        # -------------------------- Download Button for Correlation Data -------------------------- #

        # Optional: Add a download button for the correlation summary
        csv_correlation = correlation_summary.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Correlation Data as CSV",
            data=csv_correlation,
            file_name='correlation_orders_shipping_time.csv',
            mime='text/csv',
        )
else:
    st.write("No shipped orders available to perform correlation analysis.")

# -------------------------- Average Order Time per Club -------------------------- #

st.subheader("Average Order Time per Club")
st.write("""
This table displays the average shipping time (in days) for each club, along with the number of orders shipped by each club.
""")

# Calculate average shipping time and number of orders shipped per club
average_order_time = closed_orders_df.groupby('Club Name').agg(
    Average_Shipping_Time_Days=('Time to Ship', 'mean'),
    Number_of_Orders_Shipped=('Order ID', 'count')
).reset_index()

# Round the average shipping time
average_order_time['Average_Shipping_Time_Days'] = average_order_time['Average_Shipping_Time_Days'].round(2)

# -------------------------- Display the Table -------------------------- #

styled_average_order_time = average_order_time.style.format({
    'Average_Shipping_Time_Days': "{:.2f} days",
    'Number_of_Orders_Shipped': "{:,}"
}).set_properties(**{
    'text-align': 'center'
}).set_table_styles([
    dict(selector='th', props=[('text-align', 'center')]),
    dict(selector='td', props=[('text-align', 'center')])
])

st.dataframe(styled_average_order_time, use_container_width=True)

# -------------------------- Download Button for Average Order Time Table -------------------------- #

# Convert the DataFrame to CSV
csv_average_order_time = average_order_time[['Club Name', 'Average_Shipping_Time_Days', 'Number_of_Orders_Shipped']].to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Average Order Time Table as CSV",
    data=csv_average_order_time,
    file_name='average_order_time_per_club.csv',
    mime='text/csv',
)

# -------------------------- Final Touches -------------------------- #

st.markdown("---")
st.write("**Note:** This analysis is based on the data available in the `Capelli2023_aggregated_orders.csv` and `aaggregated_orders3.23.csv` files. Please ensure the data is up-to-date for accurate insights.")
