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
:root {
  --primary-color: #3498db;
  --secondary-color: #2c3e50;
  --accent-color: #e74c3c;
  --text-dark: #2c3e50;
  --text-light: #95a5a6;
  --background-light: #f8f9fa;
}

/* Main content styling */
section [data-testid="stMain"] {
  background-color: var(--background-light) !important;
  padding: 2rem 3rem !important;
  min-height: 100vh !important;
}

[data-testid="stMainBlockContainer"] {
    background: var(--background-light) !important;
}
[data-testid="stHeader"] {
  background: var(--background-light) !important;
}
header [data-testid="stToolbar"] {
  display: none !important;
}

/* Headings */
h1 {
  color: var(--secondary-color) !important;
  border-bottom: 3px solid var(--primary-color) !important;
  padding-bottom: 0.5rem !important;
  margin-bottom: 2rem !important;
  font-size: 2.5rem !important;
  font-weight: 700 !important;
}
h3, h4 {
  color: var(--secondary-color) !important;
  font-weight: 500 !important;
  margin-bottom: 1rem !important;
}
h3 {
  font-size: 1.5rem !important;
}
h4 {
  font-size: 1.25rem !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
  background: linear-gradient(160deg, var(--secondary-color), var(--primary-color)) !important;
  padding: 2rem 1.5rem !important;
  border-right: 1px solid #dfe6e9 !important;
  box-shadow: 4px 0 15px rgba(0, 0, 0, 0.05) !important;
}

/* Selectbox */
.stSelectbox {
  background: rgba(255, 255, 255, 0.1) !important;
  border-radius: 8px !important;
  padding: 0.5rem !important;
  transition: all 0.3s ease !important;
}
.stSelectbox:hover {
  background: rgba(255, 255, 255, 0.15) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
  background: white !important;
  border-radius: 12px !important;
  padding: 1.75rem !important;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
  border: 1px solid rgba(0, 0, 0, 0.05) !important;
  transition: transform 0.3s ease !important;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-5px) !important;
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12) !important;
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

/* Expander Styling */
[data-testid="stExpander"] {
  margin-top: 2rem !important;
  border: 1px solid rgba(0, 0, 0, 0.08) !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
  background: #ADD8E6 !important;
  padding: 1.25rem !important;
  font-weight: 700 !important;
  color: white !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] summary:hover {
  background: #90cce6 !important;
}

/* Table Styling */
[data-testid="stTableStyledTable"] {
  border-radius: 24px !important;
  background: white !important;
  border: 1px solid rgba(0, 0, 0, 0.08) !important;
  transition: all 0.3s ease !important;

}
[data-testid="stTable"]  {
    height:400px !important;

}
            



[data-testid="stTableStyledTable"]:hover {
  transform: translateY(-2px) !important;
    
}
[data-testid="stTableStyledTable"] thead tr {
  background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
  height: 100px !important;
}
[data-testid="stTableStyledTable"] thead th {
  color: white !important;
  font-weight: 600 !important;
  padding: 1.2rem 1.5rem !important;
  font-size: 0.95rem !important;
  text-transform: uppercase !important;
}
[data-testid="stTableStyledTable"] tbody td {
  padding: 1rem 1.5rem !important;
  color: var(--text-dark) !important;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06) !important;
}

/* Fullscreen frames */
[data-testid="stFullScreenFrame"] {
  border-radius: 24px !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
  width: 8px !important;
  height: 8px !important;
}
::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.05) !important;
}
::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.2) !important;
  border-radius: 4px !important;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.3) !important;
}

/* Responsive Design */
@media (max-width: 768px) {
  .main {
    padding: 1rem 1rem !important;
  }
  [data-testid="stMetric"] {
    margin-bottom: 1.5rem !important;
    padding: 1rem !important;
  }
  h1 {
    font-size: 2rem !important;
  }
  h3 {
    font-size: 1.25rem !important;
  }
  h4 {
    font-size: 1rem !important;
  }
  [data-testid="stSidebar"] {
    padding: 1rem !important;
  }
  [data-testid="stTableStyledTable"] thead th,
  [data-testid="stTableStyledTable"] tbody td {
    padding: 0.75rem 1rem !important;
    font-size: 0.9rem !important;
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
