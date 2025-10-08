import pandas as pd
from pymongo import MongoClient
from prophet import Prophet
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import certifi


# ------------------- Load Environment Variables -------------------
load_dotenv()  # Reads .env file
uri = os.getenv("MONGODB_URI")


# ------------------- MongoDB Connection -------------------
client = MongoClient(uri, tlsCAFile=certifi.where())
db = client['cloud_optimization']
collection = db['usage_data']


# Load data
data = pd.DataFrame(list(collection.find()))
data['date'] = pd.to_datetime(data['date'])

# ------------------- Streamlit Layout -------------------
st.set_page_config(page_title="Cloud Cost Optimization Dashboard", layout="wide")


# Wrap your main dashboard content in a container with fade-in class


st.title("☁️ Cloud Cost Optimization Dashboard")

# ------------------- Improved Sidebar Filters -------------------
st.sidebar.header("🔍 Filters & Settings")


with st.sidebar.expander("📍 Regions", expanded=False):
    regions = data['region'].unique().tolist()
    selected_regions = st.multiselect(
        "Select Regions:",
        options=regions,
        default=regions,
        help="Filter the data by cloud region(s)"
    )

with st.sidebar.expander("🛠 Services", expanded=False):
    services = data['service'].unique().tolist()
    selected_services = st.multiselect(
        "Select Services:",
        options=services,
        default=services,
        help="Filter the data by service type"
    )

with st.sidebar.expander("📅 Date Range", expanded=False):
    min_date = data['date'].min()
    max_date = data['date'].max()
    date_range = st.date_input(
        "Select Date Range:",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date,
        help="Filter the data by date range"
    )

with st.sidebar.expander("⚙ Advanced Options", expanded=False):
    usage_threshold = st.slider(
        "Low Usage Threshold (Hours)",
        min_value=0,
        max_value=1000,
        value=800,
        help="Define what counts as 'low usage' for recommendations"
    )

# Add a small info footer in the sidebar
st.sidebar.markdown("---")
st.sidebar.info("💡 Adjust filters to dynamically update the dashboard charts and recommendations.")


# ------------------- Filtered Data -------------------
def get_filtered_data():
    df = pd.DataFrame(list(collection.find()))
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['region'].isin(selected_regions)) &
            (df['service'].isin(selected_services)) &
            (df['date'] >= pd.to_datetime(date_range[0])) &
            (df['date'] <= pd.to_datetime(date_range[1]))&
    (data['usage_hours'] < usage_threshold)]
    return df

filtered_data = get_filtered_data()
import streamlit as st

# ------------------- Dashboard Metrics -------------------
st.subheader(" OVERVIEW METRICS")

# Data Calculations
total_cost = round(filtered_data['cost'].sum(), 2)
idle_instances = int(len(filtered_data[(filtered_data['status'] == 'idle') & (filtered_data['usage_hours'] < 10)]))
low_usage_buckets = int(len(filtered_data[(filtered_data['usage_hours'] < 5)]))

# --- Animated Counter Function (Non-blocking JS-based) ---
def animated_counter(label, value, prefix=""):
    html_code = f"""
    <div style='text-align:center; padding:10px;'>
        <h4 style='margin-bottom:2px; color:white;'>{label}</h4>
        <span id='{label}' style='font-size:36px; font-weight:bold; color:white;'>0</span>
    </div>
    <script>
        const el = document.getElementById("{label}");
        const target = {value};
        const duration = 500;  // 1 second
        const stepTime = 20;
        let current = 0;
        const increment = target / (duration / stepTime);
        const timer = setInterval(() => {{
            current += increment;
            if (current >= target) {{
                current = target;
                clearInterval(timer);
            }}
            el.textContent = "{prefix}" + (Math.round(current * 100) / 100).toLocaleString();
        }}, stepTime);
    </script>
    """
    st.components.v1.html(html_code, height=100)

# --- Layout ---
col1, col2, col3 = st.columns(3)

with col1:
    animated_counter(" Total Cost", total_cost, prefix="$")

with col2:
    animated_counter(" Idle Instances", idle_instances)

with col3:
    animated_counter(" Low-Usage Buckets", low_usage_buckets)

# ------------------- Cost Charts -------------------
st.subheader("COST DISTRIBUTION")
col1, col2 = st.columns(2)

with col1:
    service_cost = filtered_data.groupby('service')['cost'].sum().reset_index()
    fig_service = px.bar(service_cost, x='service', y='cost', color='service', text='cost', title="Total Cost by Service")
    st.plotly_chart(fig_service, use_container_width=True)

with col2:
    region_cost = filtered_data.groupby('region')['cost'].sum().reset_index()
    fig_region = px.pie(region_cost, names='region', values='cost', title="Total Cost by Region")
    st.plotly_chart(fig_region, use_container_width=True)


# ---- Aggregate baseline (before optimization) ----
baseline_cost = filtered_data.groupby('date')['cost'].sum().reset_index()
baseline_cost.rename(columns={'date': 'ds', 'cost': 'y'}, inplace=True)

# ---- Aggregate optimized (exclude stopped instances) ----
optimized_data = filtered_data[filtered_data['status'] != 'stopped']
optimized_cost = optimized_data.groupby('date')['cost'].sum().reset_index()
optimized_cost.rename(columns={'date': 'ds', 'cost': 'y'}, inplace=True)

# ---- Train a single Prophet model for baseline ----
model = Prophet(daily_seasonality=True, seasonality_prior_scale=10)
model.fit(baseline_cost)

# ---- Forecast next 30 days ----
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# ---- Calculate optimization ratio ----
if baseline_cost['y'].sum() > 0:
    ratio = optimized_cost['y'].sum() / baseline_cost['y'].sum()
else:
    ratio = 1.0

# ---- Compute optimized forecast based on ratio ----
forecast['yhat_optimized'] = forecast['yhat'] * ratio

# ---- Calculate projected savings ----
predicted_savings = forecast['yhat'].sum() - forecast['yhat_optimized'].sum()

# ---- Display metric ----
st.metric("💡 Projected Monthly Savings", f"${predicted_savings:,.2f}")

# ---- Plot Forecast Comparison ----
fig_forecast = go.Figure()

# 1️⃣ Actual Cost (Historical Data)
fig_forecast.add_trace(go.Bar(
    x=baseline_cost['ds'],
    y=baseline_cost['y'],
    name='Actual Cost',
    marker_color='rgba(128,128,128,0.5)',
    opacity=0.7
))

# 2️⃣ Forecast Before Optimization
fig_forecast.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Forecast (Before Optimization)',
    line=dict(color='red', width=3)
))

# 3️⃣ Forecast After Optimization (Scaled)
fig_forecast.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat_optimized'],
    mode='lines',
    name='Forecast (After Optimization)',
    line=dict(color='green', width=3, )
))

# 4️⃣ Highlight Savings Area (Between Red & Green Lines)
fig_forecast.add_trace(go.Scatter(
    x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
    y=pd.concat([forecast['yhat'], forecast['yhat_optimized'][::-1]]),
    fill='toself',
    fillcolor='rgba(0, 255, 0, 0.1)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='Projected Savings Area'
))

# ---- Chart Styling ----
fig_forecast.update_layout(
    title="📊 Cost Forecast: Before vs After Optimization",
    xaxis_title="Date",
    yaxis_title="Predicted Cost ($)",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        y=-0.25,
        x=0.05,
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor='LightGray',
        borderwidth=1
    ),
    margin=dict(l=40, r=40, t=60, b=40)
)

# ---- Display Chart ----
st.plotly_chart(fig_forecast, use_container_width=True)


# ------------------- Top Expensive Instances (Dynamic) -------------------
st.subheader("🔥 Top Expensive Instances")

# Get number of top instances dynamically
top_n = min(10, filtered_data['instance_type'].nunique())

top_instances = (
    filtered_data.groupby('instance_type')['cost']
    .sum()
    .reset_index()
    .sort_values(by='cost', ascending=False)
    .head(top_n)
)

st.dataframe(top_instances)

# ------------------- Tabs for Recommendations & Handled Items -------------------
tab1, tab2 = st.tabs(["⚡ Recommendations", "📦 Handled Items"])

# ---------- Recommendations Tab ----------
with tab1:
    st.subheader(" Smart Recommendations")
    
    # Group by service/bucket
    if 'service' in filtered_data.columns:
        grouped_services = filtered_data.groupby('service')
    else:
        st.warning("No 'service' column found in data.")
        grouped_services = {}

    # Iterate through each service category
    for service, group in grouped_services:
        with st.expander(f" {service} — {len(group)} Resources", expanded=False):
            
            # ---------- Idle Instances ----------
            idle_instances = group[(group['status'] == 'idle') & (group['usage_hours'] < 10) & (group['service'] != 'S3')]
            if not idle_instances.empty:
                st.markdown("###  Idle Instances ")
                for idx, row in idle_instances.iterrows():
                    with st.container():
                        st.write(f"🟡**ID:** {row['_id']} | **Type:** {row['instance_type']} | **Region:** {row['region']}")
                        st.write(f"Usage Hours: {row['usage_hours']}, Cost: ${row['cost']:.2f}")
                        st.code(f"# Stop instance '{row['instance_type']}' in {row['region']}", language='bash')
                        if st.button(f"Mark {row['instance_type']} as Stopped", key=f"stop_{row['_id']}"):
                            collection.update_one({"_id": row["_id"]}, {"$set": {"status": "stopped"}})
                            st.success(f"{row['instance_type']} marked as stopped!")
                            st.rerun()
            else:
                st.info("✅ No idle instances found for this service.")

            # ---------- Low Usage Buckets ----------
            low_usage_buckets = group[(group['usage_hours'] < 5) & (group['status'] == 'idle') ]
            if not low_usage_buckets.empty:
                st.markdown("### 🧹 Low-Usage Buckets (Consider cleaning)")
                for idx, row in low_usage_buckets.iterrows():
                    with st.container():
                        st.write(f"**ID:** {row['_id']} | **Bucket:** {row.get('bucket_name', 'N/A')} | **Region:** {row['region']}")
                        st.write(f"Usage Hours: {row['usage_hours']}, Cost: ${row['cost']:.2f}")
                        st.code(f"# Clean bucket '{row.get('bucket_name', 'unknown')}' in {row['region']}", language='bash')
                        if st.button(f"Mark {row.get('bucket_name', 'unknown')} as Cleaned", key=f"clean_{row['_id']}"):
                            collection.update_one({"_id": row["_id"]}, {"$set": {"status": "cleaned"}})
                            st.success(f"{row.get('bucket_name', 'unknown')} marked as cleaned!")
                            st.rerun()
            else:
                st.info("✅ No low-usage buckets for this service.")

# ---------- Handled Items Tab ----------
with tab2:
    st.subheader("Stopped Instances ")
    stopped_instances = list(collection.find({"status":"stopped"}).sort("date", -1).limit(5))
    for inst in stopped_instances:
        st.success(f"🛑 ID: {inst['_id']} | {inst['instance_type']} ({inst['region']})")

    st.subheader("Cleaned Buckets ")
    cleaned_buckets = list(collection.find({"status":"cleaned"}).sort("date", -1).limit(5))
    for bucket in cleaned_buckets:
        st.success(f"✅ ID: {bucket['_id']} | {bucket['bucket_name']} ({bucket['region']})")