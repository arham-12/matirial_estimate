import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
# Load recipe data
@st.cache_data
def load_recipe_data(path='menu_recipes.csv'):
    return pd.read_csv(path)

# Load sales data
@st.cache_data
def load_sales_data(path='restaurant_sales_custom.csv'):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Prophet holidays
pakistan_holidays = pd.DataFrame({
    'holiday': 'public_holiday',
    'ds': pd.to_datetime([
        '2023-03-23', '2023-05-01', '2023-06-28', '2023-07-01',
        '2023-08-14', '2023-09-06', '2023-11-09', '2023-12-25',
        '2024-03-23', '2024-04-10', '2024-06-17', '2024-08-14'
    ]),
    'lower_window': 0,
    'upper_window': 1
})

def prepare_data(df, branch, item):
    filtered = df[(df['branch'] == branch) & (df['item_name'] == item)]
    return filtered[['date', 'qty_sold']].rename(columns={'date': 'ds', 'qty_sold': 'y'})

def create_model():
    model = Prophet(
        holidays=pakistan_holidays,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.add_country_holidays(country_name='PK')
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
    return model

def forecast_sales(df, branch, item):
    df = df.sort_values('ds')
    split_date = df['ds'].max() - pd.Timedelta(days=30)
    train = df[df['ds'] <= split_date]
    test = df[df['ds'] > split_date]

    if train.empty or test.empty:
        return None, None, None, None

    model = create_model()
    model.fit(train)

    future = model.make_future_dataframe(periods=len(test), freq='D', include_history=False)
    forecast = model.predict(future)

    merged = pd.merge(forecast[['ds', 'yhat']], test[['ds', 'y']], on='ds', how='left').dropna()

    mae = mean_absolute_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    
    total_actual = merged['y'].sum()
    total_forecast = merged['yhat'].sum()

    return total_actual, total_forecast, merged, model

def calculate_materials(recipe_df, item, forecasted_sales):
    materials = recipe_df[recipe_df['item_name'] == item].copy()
    materials['Total Required'] = materials['quantity_per_unit'] * int(forecasted_sales)
    return materials[['material', 'quantity_per_unit', 'Total Required']]

# ----------------- Streamlit App -----------------

st.set_page_config(page_title="Sales Forecast & Material Estimator", layout="wide")




st.markdown("""
<style>
    /* Define variables for both light and dark themes */
    [data-theme="light"] {
        --primary-color: #3498db;
        --secondary-color: #2c3e50;
        --accent-color: #e74c3c;
        --text-dark: #2c3e50;
        --text-light: #95a5a6;
        --background-color: #f8f9fa;
        --metric-bg: white;
        --expander-bg: #ADD8E6;
        --table-head-bg: linear-gradient(135deg, #3498db, #2c3e50);
    }

    [data-theme="dark"] {
        --primary-color: #2980b9;
        --secondary-color: #ecf0f1;
        --accent-color: #e74c3c;
        --text-dark: #ecf0f1;
        --text-light: #bdc3c7;
        --background-color: #1e1e1e;
        --metric-bg: #2c3e50;
        --expander-bg: #1a5276;
        --table-head-bg: linear-gradient(135deg, #2980b9, #34495e);
    }

    .main {
        background-color: var(--background-color);
        padding: 2rem 3rem;
        min-height: 100vh;
    }

    [data-testid="stHeader"] {
        background: var(--background-color);
    }

    header [data-testid="stToolbar"] {   
        display: none !important;
    }

    h1, h3, h4 {
        color: var(--secondary-color);
    }

    h1 {
        border-bottom: 3px solid var(--primary-color);
        padding-bottom: 0.5rem;
        margin-bottom: 2rem;
        font-size: 2.5rem;
        font-weight: 700;
    }

    h3 {
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    h4 {
        font-size: 1.25rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, var(--secondary-color) 0%, var(--primary-color) 100%);
        padding: 2rem 1.5rem;
        border-right: 1px solid #dfe6e9;
        box-shadow: 4px 0 15px rgba(0, 0, 0, 0.05);
    }

    .stSelectbox {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        transition: all 0.3s ease;
    }

    .stSelectbox:hover {
        background: rgba(255, 255, 255, 0.15) !important;
    }

    [data-testid="stMetric"] {
        background: var(--metric-bg);
        border-radius: 12px;
        padding: 1.75rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }

    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        color: var(--text-light) !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: var(--secondary-color) !important;
        font-weight: 800 !important;
    }

    [data-testid="stExpander"] {
        margin-top: 2rem;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 12px;
    }

    [data-testid="stExpander"] summary {
        background: var(--expander-bg);
        padding: 1.25rem;
        font-weight: 700;
        color: white;
        border-radius: 12px;
    }

    [data-testid="stExpander"] summary:hover {
        background: var(--expander-bg);
    }

    [data-testid="stTableStyledTable"] {
        border-radius: 24px;
        background: var(--metric-bg);
        border: 1px solid rgba(0, 0, 0, 0.08);
    }

    [data-testid="stTableStyledTable"] thead tr {
        background: var(--table-head-bg);
    }

    [data-testid="stTableStyledTable"] thead th {
        color: white;
        font-weight: 600;
        padding: 1.2rem 1.5rem;
        font-size: 0.95rem;         
        text-transform: uppercase;
    }

    [data-testid="stTableStyledTable"] tbody td {
        padding: 1rem 1.5rem;
        color: var(--text-dark);
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    }

    [data-testid="stFullScreenFrame"] {
        border-radius: 24px;
    }

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 0, 0, 0.3);
    }

    @media (max-width: 768px) {
        .main {
            padding: 1.5rem;
        }
        [data-testid="stMetric"] {
            margin-bottom: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Sales Forecast & 🧾 Material Estimation")
# Sidebar Inputs
st.sidebar.header("Select Inputs")
df = load_sales_data()
recipe_df = load_recipe_data()

branches = df['branch'].unique()
items = df['item_name'].unique()

branch = st.sidebar.selectbox("Select Branch", branches)
item = st.sidebar.selectbox("Select Item", items)

filtered_df = prepare_data(df, branch, item)

if filtered_df.empty:
    st.warning("No data available for the selected branch and item.")
else:
    total_actual, total_forecast, merged_df, model = forecast_sales(filtered_df, branch, item)

    if total_actual is None or merged_df is None:
        st.error("Not enough data for forecasting.")
    else:
        # Sidebar date selection based on available forecast dates
        forecast_dates = merged_df['ds'].dt.date.unique()
        selected_date = st.sidebar.date_input("Select Forecast Date for Materials", forecast_dates[0],
                                              min_value=forecast_dates.min(), max_value=forecast_dates.max())

        
        

        # Filter for selected date
        selected_row = merged_df[merged_df['ds'] == pd.to_datetime(selected_date)]

        if not selected_row.empty:
            selected_forecast_qty = int(round(selected_row['yhat'].values[0]))
            selected_actual_qty = selected_row['y'].values[0] if 'y' in selected_row.columns and not pd.isna(selected_row['y'].values[0]) else None

          
            # Create 3 columns for metrics in a row
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            metric_col1.metric(label="Forecasted Sales", value=f"{selected_forecast_qty} units")

            if selected_actual_qty is not None:
                metric_col2.metric(label="Actual Sales", value=f"{selected_actual_qty:.2f} units")
                abs_error = abs(selected_forecast_qty - selected_actual_qty)
                metric_col3.metric(label="📉 Absolute Error", value=f"{abs_error:.2f} units")
            else:
                metric_col2.info("Actual sales data not available")
                metric_col3.empty()  # no error to show

            fig_col, table_col = st.columns([1, 1])

            with fig_col:
                st.subheader(f"Forecast & Materials for {selected_date.strftime('%Y-%m-%d')}")

                # Interactive bar chart for only selected date
                chart_data = [
                go.Bar(name='Forecasted', x=['Forecasted'], y=[selected_forecast_qty], marker_color='#ADD8E6') 
                ]

                if selected_actual_qty is not None:
                    chart_data.append(go.Bar(name='Actual', x=['Actual'], y=[selected_actual_qty], marker_color='blue'))

                comp_fig = go.Figure(data=chart_data)
                comp_fig.update_layout(
                    title=f"Sales on {selected_date.strftime('%Y-%m-%d')}",
                    yaxis_title='Units Sold',
                    barmode='group'
                )
                st.plotly_chart(comp_fig, use_container_width=True)

            with table_col:
                # Show materials needed based on forecast
                materials_needed = calculate_materials(recipe_df, item, selected_forecast_qty)
                st.subheader("Required Materials")
                st.table(materials_needed) #Table displayed here

        else:
            st.warning("No forecast available for the selected date.")
        with st.expander("🔍 Detailed Forecast Data"):
            st.table(merged_df[['ds', 'y', 'yhat']].rename(columns={
                'ds': 'Date', 'y': 'Actual', 'yhat': 'Forecasted'
            }))
