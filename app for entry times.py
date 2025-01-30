import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Function to filter trade log by date range
def filter_trade_log_by_date(data, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = data[(data['Date Opened'] >= start_date) & (data['Date Opened'] <= end_date)]
    return filtered_data

# Function to load trade log from a specific folder
def load_trade_log(file_path):
    trade_log_df = pd.read_csv(file_path)
    trade_log_df['Time Opened'] = pd.to_datetime(trade_log_df['Time Opened'], format='%H:%M:%S').dt.time
    trade_log_df['Date Opened'] = pd.to_datetime(trade_log_df['Date Opened'])  # Ensure date is in datetime format
    return trade_log_df

# Function to calculate maximum drawdown for each entry time
def calculate_max_drawdown(data, initial_value=100000):
    max_drawdowns = {}
    total_pnl = {}
    strategies = data.groupby('Time Opened')

    for time, group in strategies:
        group = group.sort_values(by='Date Opened').reset_index(drop=True)
        group['rolling_p/l'] = initial_value + group['P/L'].cumsum()
        cumulative = group['rolling_p/l']
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak * 100  # percentage drawdown
        max_drawdown_value = drawdown.min()
        max_drawdowns[time] = max_drawdown_value
        total_pnl[time] = (group['P/L']).sum()

    max_drawdowns_df = pd.DataFrame({
        'Entry Time': list(max_drawdowns.keys()),
        'Max Drawdown (%)': list(max_drawdowns.values()),
        'Total P/L': list(total_pnl.values())
    })
    max_drawdowns_df['P/L to MDD Ratio'] = max_drawdowns_df['Total P/L'] / abs(max_drawdowns_df['Max Drawdown (%)']).replace(0, np.nan)  # Avoid division by zero
    max_drawdowns_df = max_drawdowns_df.sort_values(by='Entry Time')
    return max_drawdowns_df

# Function to calculate the average winner divided by the max loser for each entry time
def calculate_avg_winner_vs_max_loser(data):
    time_entries = []
    avg_winners = []
    max_losers = []
    ratios = []

    grouped = data.groupby('Time Opened')

    for time, group in grouped:
        avg_winner = group[group['P/L'] > 0]['P/L'].mean()
        max_loser = group[group['P/L'] < 0]['P/L'].min()

        if pd.notna(avg_winner) and pd.notna(max_loser) and max_loser != 0:
            ratio = avg_winner / abs(max_loser)
        else:
            ratio = 0

        time_entries.append(time)
        avg_winners.append(avg_winner)
        max_losers.append(max_loser)
        ratios.append(ratio)

    ratio_df = pd.DataFrame({
        'Time Opened': time_entries,
        'Avg Winner': avg_winners,
        'Max Loser': max_losers,
        'Avg Winner / Max Loser': ratios
    })

    return ratio_df

# Function to calculate expectancy for each entry time
def calculate_expectancy(data):
    time_entries = []
    expectancies = []

    grouped = data.groupby('Time Opened')

    for time, group in grouped:
        avg_winner = group[group['P/L'] > 0]['P/L'].mean()
        avg_loser = group[group['P/L'] < 0]['P/L'].mean()
        win_pct = len(group[group['P/L'] > 0]) / len(group) * 100

        if pd.notna(avg_winner) and pd.notna(avg_loser):
            expectancy = (avg_winner * (win_pct / 100)) - (abs(avg_loser) * (1 - (win_pct / 100)))
        else:
            expectancy = 0

        time_entries.append(time)
        expectancies.append(expectancy)

    expectancy_df = pd.DataFrame({
        'Time Opened': time_entries,
        'Expectancy': expectancies
    })

    return expectancy_df

# Function to calculate composite score with revised hybrid metrics
def calculate_composite_score(data, weight_pnl_mdd, weight_expectancy, weight_avg_win_max_loss):
    # Normalize weights to ensure they sum to 1
    total_weight = weight_pnl_mdd + weight_expectancy + weight_avg_win_max_loss
    if total_weight == 0:
        st.error("The sum of weights cannot be zero.")
        return data  # Return data unaltered if weights are zero to avoid errors

    weight_pnl_mdd /= total_weight
    weight_expectancy /= total_weight
    weight_avg_win_max_loss /= total_weight

    # Normalize each metric for comparability
    data['Normalized P/L to MDD Ratio'] = (data['P/L to MDD Ratio'] - data['P/L to MDD Ratio'].min()) / (data['P/L to MDD Ratio'].max() - data['P/L to MDD Ratio'].min()) if weight_pnl_mdd > 0 else 0
    data['Normalized Expectancy'] = (data['Expectancy'] - data['Expectancy'].min()) / (data['Expectancy'].max() - data['Expectancy'].min()) if weight_expectancy > 0 else 0
    data['Normalized Avg Winner / Max Loser'] = (data['Avg Winner / Max Loser'] - data['Avg Winner / Max Loser'].min()) / (data['Avg Winner / Max Loser'].max() - data['Avg Winner / Max Loser'].min()) if weight_avg_win_max_loss > 0 else 0

    # Calculate composite score based on normalized weighted metrics
    data['Composite Score'] = (
        weight_pnl_mdd * data['Normalized P/L to MDD Ratio'] +
        weight_expectancy * data['Normalized Expectancy'] +
        weight_avg_win_max_loss * data['Normalized Avg Winner / Max Loser']
    )

    return data

# Function to plot interactive bar chart using Plotly
def plot_interactive_bar_chart(data, x, y, title, xlabel, ylabel, color='blue'):
    fig = px.bar(data, x=x, y=y, title=title, labels={x: xlabel, y: ylabel}, color_discrete_sequence=[color])
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)

# Streamlit App
st.title('Trade Log Analysis App')

# User input for selecting the trade log number
st.sidebar.header('Select Trade Log Number')
trade_log_number = st.sidebar.number_input('Trade Log Number', min_value=1, step=1, value=10)

# Define the full path to your CSV file based on selected trade log number
file_path = rf'C:\Users\m.kowalczyk\Downloads\trade-log ({trade_log_number}).csv'

# Load the trade log file
try:
    trade_log_df = load_trade_log(file_path)
except FileNotFoundError:
    st.error(f"File not found at the specified path: {file_path}. Please check the path and ensure the file exists.")
    st.stop()

# User inputs for date range
st.sidebar.header('Specify Date Range')
start_date = st.sidebar.date_input('Start Date', trade_log_df['Date Opened'].min().date())
end_date = st.sidebar.date_input('End Date', trade_log_df['Date Opened'].max().date())

# User inputs for weights
st.sidebar.header('Adjust Composite Score Weights')
weight_pnl_mdd = st.sidebar.slider('Weight for P/L to MDD Ratio', 0.0, 1.0, 0.4)
weight_expectancy = st.sidebar.slider('Weight for Expectancy', 0.0, 1.0, 0.4)
weight_avg_win_max_loss = st.sidebar.slider('Weight for Avg Winner / Max Loser Ratio', 0.0, 1.0, 0.2)

# Filter the data by date range
filtered_trade_log_df = filter_trade_log_by_date(trade_log_df, start_date, end_date)

# Button to run the analysis
if st.sidebar.button('Run Backtest'):
    st.write(f"## Results from {start_date} to {end_date}")

    # Calculate maximum drawdown
    max_drawdowns_df = calculate_max_drawdown(filtered_trade_log_df)

    # Calculate Avg Winner to Max Loser Ratio
    avg_winner_vs_max_loser_df = calculate_avg_winner_vs_max_loser(filtered_trade_log_df)

    # Calculate Expectancy
    expectancy_df = calculate_expectancy(filtered_trade_log_df)

    # Merge DataFrames
    merged_df = pd.merge(max_drawdowns_df, avg_winner_vs_max_loser_df, left_on='Entry Time', right_on='Time Opened')
    merged_df = pd.merge(merged_df, expectancy_df, left_on='Entry Time', right_on='Time Opened')

    # Calculate Composite Score with user-specified weights
    composite_df = calculate_composite_score(
        merged_df,
        weight_pnl_mdd,
        weight_expectancy,
        weight_avg_win_max_loss
    )

    # Plot Composite Score by Entry Time
    plot_interactive_bar_chart(
        composite_df,
        x='Entry Time',
        y='Composite Score',
        title='Composite Score by Entry Time',
        xlabel='Entry Time (HH:MM)',
        ylabel='Composite Score',
        color='dodgerblue'
    )

    # Plot P/L to MDD Ratio by Entry Time
    plot_interactive_bar_chart(
        composite_df,
        x='Entry Time',
        y='P/L to MDD Ratio',
        title='P/L to MDD Ratio by Entry Time',
        xlabel='Entry Time (HH:MM)',
        ylabel='P/L to MDD Ratio',
        color='red'
    )

    # Plot Expectancy (Weighted) by Entry Time (Now appears before Avg Winner / Max Loser)
    plot_interactive_bar_chart(
        composite_df,
        x='Entry Time',
        y='Expectancy',
        title='Expectancy (Weighted) by Entry Time',
        xlabel='Entry Time (HH:MM)',
        ylabel='Expectancy (Weighted)',
        color='purple'
    )

    # Plot Avg Winner / Max Loser (Weighted) by Entry Time
    plot_interactive_bar_chart(
        composite_df,
        x='Entry Time',
        y='Avg Winner / Max Loser',
        title='Avg Winner / Max Loser (Weighted) by Entry Time',
        xlabel='Entry Time (HH:MM)',
        ylabel='Avg Winner / Max Loser (Weighted)',
        color='orange'
    )
