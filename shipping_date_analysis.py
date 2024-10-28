# shipping_date_analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

# -------------------------- Streamlit App Setup -------------------------- #

st.set_page_config(page_title="Aggregated Orders Analysis", layout="wide")

st.title("Aggregated Orders Analysis")
st.write("""
This application analyzes aggregated order data from the `aggregated_orders.csv` file. Explore delivery times and shipping performance across different clubs.
""")

# -------------------------- Data Loading -------------------------- #

@st.cache_data
def load_data(filepath):
    """
    Loads and preprocesses the aggregated orders data.

    Parameters:
    - filepath (str): Path to the aggregated_orders.csv file.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """
    if not os.path.exists(filepath):
        st.error(f"The file '{filepath}' does not exist in the specified directory.")
        st.stop()

    try:
        df = pd.read_csv(filepath, sep=None, engine='python')  # Auto-detect separator
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

    # Verify required columns exist
    required_columns = [
        'Customer Reference', 'Club Name', 'Date Created', 'Order Quantity',
        'Shipped Quantity', 'Unshipped Quantity', 'Shipping Date',
        'Sales Order Header Status'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"The following required columns are missing in the CSV file: {', '.join(missing_columns)}")
        st.stop()

    # Parse date columns
    df['Date Created'] = pd.to_datetime(df['Date Created'], errors='coerce')
    df['Shipping Date'] = pd.to_datetime(df['Shipping Date'], errors='coerce')

    # Ensure numerical columns are correctly typed
    numeric_cols = ['Order Quantity', 'Shipped Quantity', 'Unshipped Quantity']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values
    df['Sales Order Header Status'] = df['Sales Order Header Status'].fillna('UNKNOWN')

    # Fill NaN in numerical columns with 0
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Standardize 'Sales Order Header Status' to uppercase for consistency
    df['Sales Order Header Status'] = df['Sales Order Header Status'].str.upper()

    # Calculate 'Time to Ship' as the difference in days between 'Shipping Date' and 'Date Created'
    df['Time to Ship'] = (df['Shipping Date'] - df['Date Created']).dt.days

    # Determine 'Over 5 weeks?' based on 'Time to Ship' > 35 days
    df['Over 5 weeks?'] = np.where(df['Time to Ship'] > 35, 'Over 5 weeks', 'Under 5 weeks')

    return df

# Load the data
data_file = os.path.join('shippingdates', 'aggregated_orders.csv')
df = load_data(data_file)

# -------------------------- Sidebar Filters -------------------------- #

st.sidebar.header("Filter Options")

# Dropdown for selecting a club
clubs = df['Club Name'].unique()
clubs_sorted = sorted(clubs)
selected_club = st.sidebar.selectbox("Select Club", options=['All Clubs'] + list(clubs_sorted))

# If a specific club is selected, filter the DataFrame
if selected_club != 'All Clubs':
    filtered_df = df[df['Club Name'] == selected_club]
else:
    filtered_df = df.copy()

# -------------------------- Percentage of Orders Over 5 Weeks per Club -------------------------- #

st.subheader("Percentage of Orders Over 5 Weeks (Month Over Month)")
st.write("""
This table shows the percentage of orders that took over 5 weeks to ship for each month.
""")

# Ensure there are shipping dates to calculate percentages
if not filtered_df['Shipping Date'].isna().all():
    percentage_df = filtered_df.dropna(subset=['Shipping Date', 'Time to Ship']).copy()

    if percentage_df.empty:
        st.write("No shipping time data available to display percentage analysis.")
    else:
        # Extract Shipping Month
        percentage_df['Shipping Month'] = percentage_df['Shipping Date'].dt.to_period('M').dt.to_timestamp()

        # Group by Shipping Month
        grouped = percentage_df.groupby('Shipping Month')

        # Calculate total orders shipped per month
        total_orders = grouped.size().reset_index(name='Total Orders Shipped')

        # Calculate orders shipped under 5 weeks
        under_5_weeks = grouped.apply(lambda x: (x['Over 5 weeks?'] == 'Under 5 weeks').sum()).reset_index(name='Orders Shipped Under 5 Weeks')

        # Calculate orders shipped over 5 weeks
        over_5_weeks = grouped.apply(lambda x: (x['Over 5 weeks?'] == 'Over 5 weeks').sum()).reset_index(name='Orders Shipped Over 5 Weeks')

        # Merge the summaries
        percentage_summary = pd.merge(total_orders, under_5_weeks, on='Shipping Month')
        percentage_summary = pd.merge(percentage_summary, over_5_weeks, on='Shipping Month')

        # Calculate percentages
        percentage_summary['% Of Orders Shipped under 5 weeks'] = (
            percentage_summary['Orders Shipped Under 5 Weeks'] / percentage_summary['Total Orders Shipped'] * 100
        ).round(2)

        percentage_summary['% Of Orders Shipped over 5 weeks'] = (
            percentage_summary['Orders Shipped Over 5 Weeks'] / percentage_summary['Total Orders Shipped'] * 100
        ).round(2)

        # Format Shipping Month as string for better display
        percentage_summary['Shipping Month'] = percentage_summary['Shipping Month'].dt.strftime('%Y-%m')

        # Rename columns for clarity
        percentage_summary.rename(columns={'Shipping Month': 'Month'}, inplace=True)

        # Select and reorder columns
        percentage_table = percentage_summary[['Month', '% Of Orders Shipped under 5 weeks', '% Of Orders Shipped over 5 weeks']]

        # Calculate grand total
        total_under = percentage_summary['Orders Shipped Under 5 Weeks'].sum()
        total_over = percentage_summary['Orders Shipped Over 5 Weeks'].sum()
        total = percentage_summary['Total Orders Shipped'].sum()

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
        })

        # Display the percentage table
        st.dataframe(styled_percentage_table, use_container_width=True)
else:
    st.write("No shipping data available to display the percentage of orders over 5 weeks.")

# -------------------------- Count of Orders Over 5 Weeks per Club -------------------------- #

st.subheader("Count of Orders Over 5 Weeks per Club (Month Over Month)")
st.write("""
This table shows the count of orders that took over 5 weeks to ship for each month.
""")

# Ensure there are shipping dates to calculate counts
if not filtered_df['Shipping Date'].isna().all():
    count_df = filtered_df.dropna(subset=['Shipping Date', 'Time to Ship']).copy()

    if count_df.empty:
        st.write("No shipping time data available to display count analysis.")
    else:
        # Extract Shipping Month
        count_df['Shipping Month'] = count_df['Shipping Date'].dt.to_period('M').dt.to_timestamp()

        # Group by Shipping Month
        grouped_counts = count_df.groupby('Shipping Month')

        # Calculate orders shipped under 5 weeks
        under_5_weeks_counts = grouped_counts.apply(lambda x: (x['Over 5 weeks?'] == 'Under 5 weeks').sum()).reset_index(name='Orders Shipped Under 5 Weeks')

        # Calculate orders shipped over 5 weeks
        over_5_weeks_counts = grouped_counts.apply(lambda x: (x['Over 5 weeks?'] == 'Over 5 weeks').sum()).reset_index(name='Orders Shipped Over 5 Weeks')

        # Merge the counts
        counts_summary = pd.merge(under_5_weeks_counts, over_5_weeks_counts, on='Shipping Month')

        # Format Shipping Month as string for better display
        counts_summary['Shipping Month'] = counts_summary['Shipping Month'].dt.strftime('%Y-%m')

        # Rename columns for clarity
        counts_summary.rename(columns={'Shipping Month': 'Month'}, inplace=True)

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
        })

        # Display the counts table
        st.dataframe(styled_counts_table, use_container_width=True)
else:
    st.write("No shipping data available to display the count of orders over 5 weeks.")

# -------------------------- Shipping Time for Each Order Over Time -------------------------- #

st.subheader("Shipping Time for Each Order Over Time")
st.write("""
This graph shows how long it took for each order to ship, based on the shipping date over time.
""")

# Ensure there are shipping dates to plot
if not filtered_df['Shipping Date'].isna().all():
    shipping_time_df = filtered_df.dropna(subset=['Shipping Date', 'Time to Ship'])

    if shipping_time_df.empty:
        st.write("No shipping time data available to display.")
    else:
        # Plotting
        fig_shipping_time = px.scatter(
            shipping_time_df,
            x='Shipping Date',
            y='Time to Ship',
            color='Club Name',
            title=f'Shipping Time Over Time for {selected_club}' if selected_club != 'All Clubs' else 'Shipping Time Over Time for All Clubs',
            labels={
                'Shipping Date': 'Shipping Date',
                'Time to Ship': 'Shipping Time (days)',
                'Club Name': 'Club'
            },
            hover_data=['Customer Reference', 'Order Quantity', 'Shipped Quantity']
        )

        # Add a dashed line at 35 days to indicate the 5-week cutoff
        fig_shipping_time.add_shape(
            dict(
                type="line",
                x0=shipping_time_df['Shipping Date'].min(),
                y0=35,
                x1=shipping_time_df['Shipping Date'].max(),
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
    st.write("No shipping data available to display the shipping time graph.")

# -------------------------- Number of Orders Shipped Over Time -------------------------- #

st.subheader("Number of Orders Shipped Over Time")
st.write("""
This graph shows the number of orders shipped each month.
""")

# Ensure there are shipping dates to plot
if not filtered_df['Shipping Date'].isna().all():
    orders_count_df = filtered_df.dropna(subset=['Shipping Date']).copy()

    if orders_count_df.empty:
        st.write("No shipping data available to display orders count over time.")
    else:
        # Extract Shipping Month
        orders_count_df['Shipping Month'] = orders_count_df['Shipping Date'].dt.to_period('M').dt.to_timestamp()

        # Group by Shipping Month and count orders
        orders_per_month = orders_count_df.groupby('Shipping Month').size().reset_index(name='Number of Orders Shipped')

        # Sort by Shipping Month
        orders_per_month = orders_per_month.sort_values('Shipping Month')

        # Plotting
        fig_orders_over_time = px.line(
            orders_per_month,
            x='Shipping Month',
            y='Number of Orders Shipped',
            title='Number of Orders Shipped Over Time',
            labels={
                'Shipping Month': 'Month',
                'Number of Orders Shipped': 'Number of Orders Shipped'
            },
            markers=True
        )

        fig_orders_over_time.update_layout(
            xaxis=dict(tickangle=45),
            yaxis=dict(title='Number of Orders Shipped'),
            template='plotly_white'
        )

        st.plotly_chart(fig_orders_over_time, use_container_width=True)
else:
    st.write("No shipping data available to display orders count over time.")

# -------------------------- Average Shipping Time per Month -------------------------- #

st.subheader("Average Shipping Time per Month")
st.write("""
This section provides both a box plot and a bar chart showing the distribution and average of shipping times per month.
""")

# Calculate average shipping time per month
if not filtered_df['Shipping Date'].isna().all():
    avg_shipping_time_month = shipping_time_df.copy()
    avg_shipping_time_month['Shipping Month'] = avg_shipping_time_month['Shipping Date'].dt.to_period('M').dt.to_timestamp()

    # Box Plot
    fig_box = px.box(
        avg_shipping_time_month,
        x='Shipping Month',
        y='Time to Ship',
        title='Distribution of Shipping Times per Month',
        labels={
            'Shipping Month': 'Month',
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
    avg_shipping_time = avg_shipping_time_month.groupby('Shipping Month')['Time to Ship'].mean().reset_index()

    fig_bar = px.bar(
        avg_shipping_time,
        x='Shipping Month',
        y='Time to Ship',
        title='Average Shipping Time per Month',
        labels={
            'Shipping Month': 'Month',
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
else:
    st.write("No shipping time data available to display average shipping time per month.")

# -------------------------- Average Order Time per Club -------------------------- #

st.subheader("Average Order Time per Club")
st.write("""
This table displays the average shipping time (in days) for each club, along with the number of orders shipped by each club.
""")

# Calculate average shipping time and number of orders shipped per club
average_order_time = shipping_time_df.groupby('Club Name').agg(
    Average_Shipping_Time_Days=('Time to Ship', 'mean'),
    Number_of_Orders_Shipped=('Customer Reference', 'count')
).reset_index()

# Round the average shipping time
average_order_time['Average_Shipping_Time_Days'] = average_order_time['Average_Shipping_Time_Days'].round(2)

# Display the table with styling using st.dataframe for sortable columns
st.dataframe(average_order_time.style.format({
    'Average_Shipping_Time_Days': "{:.2f} days",
    'Number_of_Orders_Shipped': "{:,}"
}).set_properties(**{
    'text-align': 'center'
}), use_container_width=True)

# -------------------------- Final Touches -------------------------- #

st.markdown("---")
st.write("**Note:** This analysis is based on the data available in the `aggregated_orders.csv` file. Please ensure the data is up-to-date for accurate insights.")
