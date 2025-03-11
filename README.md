# Capelli Orders Analysis

The **Capelli Orders Analysis** repository contains a Streamlit application that analyzes aggregated order data from our Snowflake database. The app provides insights into delivery times, shipping performance, and order metrics across different clubs within the Rush Soccer network.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Data Loading Process](#data-loading-process)
- [Deployment in Snowflake](#deployment-in-snowflake)
- [Contributing](#contributing)
- [License](#license)

## Overview

This application analyzes aggregated order data from our Snowflake table `AGGREGATED_ORDERS3_2`. It is designed to explore delivery times and shipping performance based on the order creation dates, helping us monitor key operational metrics such as outstanding orders, shipping delays, and overall shipping performance by club.

## Features

- **Interactive Dashboard:** Built with Streamlit, the dashboard offers an intuitive UI for exploring order data.
- **Dynamic Filtering:** Filter data by club or other criteria using sidebar options.
- **Data Aggregation & Metrics:** Automatically preprocesses and aggregates data to calculate metrics such as "Time to Ship," "Order Age," and categorizes orders as "Over 5 weeks" or "Under 5 weeks."
- **Visualizations:** Uses Plotly to generate interactive charts (scatter plots, bar charts, line charts) for tracking shipping performance, order volumes, and correlation analysis.
- **Downloadable Reports:** Easily export aggregated data and analysis results as CSV files.
- **Snowflake Integration:** Leverages Snowflake’s secure data environment with Snowpark for seamless data access and processing.

## Architecture

- **Streamlit:** Provides the web-based user interface.
- **Snowflake & Snowpark:** Powers secure data storage and querying. The app directly queries the `AGGREGATED_ORDERS3_2` table.
- **Plotly:** Used for data visualization.
- **Python Libraries:** Pandas, NumPy, and logging are used for data processing and debugging.

## Setup & Installation

### Prerequisites

- Python 3.8 or above.
- Access to a Snowflake account with Snowpark enabled.
- Required Python packages listed in `requirements.txt` (e.g., streamlit, pandas, numpy, plotly, snowflake-snowpark-python).

### Local Development

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/awall1010/capelli-orders-analysis.git
   cd capelli-orders-analysis
