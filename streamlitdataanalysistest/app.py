#app.py
#analytics app 27/08/2025 - Enhanced Version with Department Selector

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
# Using only available libraries - no sklearn needed
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Advanced Incident Analytics", layout="wide")

st.title("📊 Advanced Incident Analytics & Safety Intelligence")
st.markdown("*Comprehensive insights to improve workplace safety and prevent future incidents*")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your incident CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- Enhanced Data Cleaning ---
    # Handle various date formats (revert to working version)
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Standardize time columns and handle 00:00:00 issue
    time_columns = [col for col in df.columns if 'time' in col.lower()]
    valid_time_column = None
    
    for col in time_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce').dt.time
        # Check if this time column has meaningful data (not all 00:00:00)
        if col in df.columns:
            time_values = df[col].dropna()
            if len(time_values) > 0:
                # Count non-midnight times
                non_midnight = sum(1 for t in time_values if t != pd.Timestamp('00:00:00').time())
                if non_midnight > len(time_values) * 0.1:  # If more than 10% are not midnight
                    valid_time_column = col
                    break
    
    # Extract hour from datetime for time analysis
    if 'incident_date' in df.columns:
        incident_datetime = pd.to_datetime(df['incident_date'])
        
        # Try to get hour from time column first if available and valid
        if valid_time_column and valid_time_column in df.columns:
            # Convert time to hour
            df['hour'] = df[valid_time_column].apply(
                lambda x: x.hour if pd.notna(x) else pd.to_datetime(df.loc[df[valid_time_column]==x, 'incident_date']).dt.hour
            )
            st.info(f"ℹ️ Using '{valid_time_column}' column for time analysis (found meaningful time data)")
        else:
            # Fallback to extracting hour from incident_date
            df['hour'] = incident_datetime.dt.hour
            if valid_time_column:
                st.warning(f"⚠️ Time column '{valid_time_column}' contains mostly 00:00:00 values. Using incident_date for time analysis.")
            else:
                st.info("ℹ️ No valid time column found. Using incident_date for time analysis.")
        
        df['day_of_week'] = incident_datetime.dt.day_name()
        df['month'] = incident_datetime.dt.month
        df['year'] = incident_datetime.dt.year
    
    # Clean text columns
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    # --- DEPARTMENT SELECTOR ---
    st.subheader("🔍 Department Analysis Filter")
    
    # Create department selector
    if 'department' in df.columns:
        departments = ['All Departments'] + sorted(df['department'].dropna().unique().tolist())
        selected_department = st.selectbox(
            "Select Department for Analysis:",
            departments,
            help="Choose a specific department or 'All Departments' for organization-wide analysis"
        )
        
        # Filter data based on selection
        if selected_department == 'All Departments':
            df_filtered = df.copy()
            st.info(f"📊 Analyzing all departments ({len(df)} total incidents)")
        else:
            df_filtered = df[df['department'] == selected_department].copy()
            st.info(f"📊 Analyzing {selected_department} department ({len(df_filtered)} incidents)")
            
            if len(df_filtered) == 0:
                st.warning(f"No incidents found for {selected_department} department.")
                st.stop()
    else:
        df_filtered = df.copy()
        st.warning("⚠️ No 'department' column found. Showing all data.")
        selected_department = 'All Departments'
    
    st.subheader("🔎 Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df_filtered.head())
    with col2:
        st.write("**Data Quality Summary:**")
        missing_data = df_filtered.isnull().sum()
        st.write(f"Total Records: {len(df_filtered)}")
        st.write(f"Columns: {len(df_filtered.columns)}")
        if selected_department != 'All Departments':
            st.write(f"Department: {selected_department}")
        if missing_data.sum() > 0:
            st.write("Missing Values:")
            for col, missing in missing_data[missing_data > 0].items():
                st.write(f"  • {col}: {missing} ({missing/len(df_filtered)*100:.1f}%)")

    # --- Enhanced KPI Dashboard ---
    st.subheader("📈 Executive Dashboard")
    
    # Calculate advanced metrics with proper date filtering
    total_incidents = len(df_filtered)
    current_date = datetime.now()
    current_month = current_date.month
    current_year = current_date.year
    
    # Initialize trend variables
    incident_trend = 0
    current_month_incidents = 0
    last_month_incidents = 0
    
    # Filter for current period if date exists with proper error handling
    if 'incident_date' in df_filtered.columns and len(df_filtered) > 0:
        # Clean and handle date data
        df_filtered_dates = df_filtered.dropna(subset=['incident_date'])
        
        if len(df_filtered_dates) > 0:
            # Current month incidents
            df_current_month = df_filtered_dates[
                (df_filtered_dates['month'] == current_month) & 
                (df_filtered_dates['year'] == current_year)
            ]
            current_month_incidents = len(df_current_month)
            
            # Last month calculation with year rollover handling
            if current_month == 1:  # January
                last_month = 12
                last_year = current_year - 1
            else:
                last_month = current_month - 1
                last_year = current_year
            
            df_last_month = df_filtered_dates[
                (df_filtered_dates['month'] == last_month) & 
                (df_filtered_dates['year'] == last_year)
            ]
            last_month_incidents = len(df_last_month)
            
            # Calculate percentage change
            if last_month_incidents > 0:
                incident_trend = ((current_month_incidents - last_month_incidents) / last_month_incidents) * 100
            elif current_month_incidents > 0:
                incident_trend = 100  # 100% increase from 0
            else:
                incident_trend = 0
    
    # Injury rate calculation - now specifically for 'was_injured' column
    injury_rate = 0
    if 'was_injured' in df_filtered.columns and len(df_filtered) > 0:
        # Handle different data types (boolean, text, numbers)
        try:
            if df_filtered['was_injured'].dtype == 'bool':
                injury_rate = (df_filtered['was_injured'].sum() / len(df_filtered)) * 100
            elif df_filtered['was_injured'].dtype == 'object':
                # Count 'Yes', 'True', '1', etc. as injuries
                injured_count = df_filtered['was_injured'].dropna().apply(
                    lambda x: str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured']
                ).sum()
                injury_rate = (injured_count / len(df_filtered)) * 100
            else:
                # Numeric - count values > 0
                injury_rate = (df_filtered['was_injured'].fillna(0).astype(float).gt(0).sum() / len(df_filtered)) * 100
        except:
            injury_rate = 0
    
    # Display KPIs with department context
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Total Incidents", total_incidents)
    
    # Format trend display
    trend_display = f"{incident_trend:+.1f}%" if incident_trend != 0 else "0%"
    col2.metric("This Month", current_month_incidents, trend_display)
    
    col3.metric("Injury Rate", f"{injury_rate:.1f}%")
    
    if selected_department == 'All Departments':
        col4.metric("Departments", df_filtered['department'].nunique() if 'department' in df_filtered.columns else 0)
        col5.metric("Locations", df_filtered['location'].nunique() if 'location' in df_filtered.columns else 0)
    else:
        # Show department-specific metrics
        dept_locations = df_filtered['location'].nunique() if 'location' in df_filtered.columns else 0
        col4.metric("Locations", dept_locations)
        
        # Show department's share of total incidents
        if 'department' in df.columns:
            total_org_incidents = len(df)
            dept_share = (total_incidents / total_org_incidents) * 100 if total_org_incidents > 0 else 0
            col5.metric("Dept Share", f"{dept_share:.1f}%")

    # Add comparison info for department analysis
    if selected_department != 'All Departments':
        st.info(f"🔄 **Month-over-Month**: {current_month_incidents} incidents this month vs {last_month_incidents} last month ({trend_display} change)")

    # --- ENHANCED PREDICTIVE ANALYTICS ---
    st.subheader("🔮 Advanced Predictive Analytics & Forecasting")
    
    if 'incident_date' in df_filtered.columns and len(df_filtered) > 10:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**📊 Incident Trend Forecast - {selected_department}**")
            
            # Create monthly aggregation for forecasting
            df_temp = df_filtered.copy()
            df_temp['year'] = df_temp['incident_date'].dt.year
            df_temp['month'] = df_temp['incident_date'].dt.month
            
            df_monthly = df_temp.groupby(['year', 'month']).size().reset_index(name='incident_count')
            # Create date column properly
            df_monthly['date'] = pd.to_datetime(df_monthly[['year', 'month']].assign(day=1))
            df_monthly = df_monthly.sort_values('date')
            
            if len(df_monthly) >= 3:
                # Enhanced forecasting with numpy polynomial fitting (replaces sklearn)
                df_monthly['month_num'] = range(len(df_monthly))
                X = np.array(df_monthly['month_num'])
                y = np.array(df_monthly['incident_count'])
                
                # Use numpy polyfit for trend analysis (linear regression equivalent)
                if len(X) >= 2:
                    # Fit linear trend
                    coeffs = np.polyfit(X, y, 1)  # Linear fit
                    
                    # Create next 3 months prediction
                    future_months = np.array([len(df_monthly) + i for i in range(1, 4)])
                    future_predictions = np.polyval(coeffs, future_months)
                    future_predictions = np.maximum(future_predictions, 0)  # Ensure non-negative
                    
                    # Also calculate moving average for comparison
                    window = min(3, len(df_monthly))
                    moving_avg = df_monthly['incident_count'].tail(window).mean()
                    
                    # Blend trend and moving average (70% trend, 30% moving average)
                    future_predictions = future_predictions * 0.7 + moving_avg * 0.3
                else:
                    # Fallback to simple average if insufficient data
                    future_predictions = np.array([df_monthly['incident_count'].mean()] * 3)
                
                last_date = df_monthly['date'].max()
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                           periods=3, freq='MS')
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_monthly['date'], y=df_monthly['incident_count'], 
                                       mode='lines+markers', name='Actual Incidents'))
                fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, 
                                       mode='lines+markers', name='ML Forecast', 
                                       line=dict(dash='dash', color='red')))
                fig.update_layout(title=f"3-Month Trend-Based Forecast - {selected_department}", 
                                xaxis_title="Date", yaxis_title="Incidents")
                st.plotly_chart(fig, use_container_width=True)
                
                # Display predictions with confidence note
                st.write("**📈 Next 3 Months Predictions (Trend Analysis):**")
                for i, (date, pred) in enumerate(zip(future_dates, future_predictions)):
                    st.write(f"• {date.strftime('%B %Y')}: ~{int(pred)} incidents")
                st.write("*Based on historical trend analysis and recent patterns*")
        
        with col2:
            st.write(f"**🎯 Enhanced Risk Indicators - {selected_department}**")
            
            # Calculate risk scores
            risk_factors = []
            
            # Location risk within department
            if 'location' in df_filtered.columns and len(df_filtered) > 0:
                loc_incidents = df_filtered['location'].value_counts()
                if len(loc_incidents) > 0:
                    high_risk_loc = loc_incidents.index[0]
                    risk_factors.append(f"🔴 High Risk Location: {high_risk_loc} ({loc_incidents.iloc[0]} incidents)")
            
            # Enhanced time-based risk with 00:00:00 handling
            if 'hour' in df_filtered.columns:
                hour_risk = df_filtered['hour'].value_counts()
                if len(hour_risk) > 0:
                    peak_hour = hour_risk.index[0]
                    # Check if peak hour is meaningful
                    if peak_hour == 0 and len(hour_risk) > 1:
                        # If midnight is peak, also show second highest
                        second_peak = hour_risk.index[1]
                        risk_factors.append(f"⏰ Peak Risk Times: {peak_hour}:00 ({hour_risk.iloc[0]} incidents), {second_peak}:00 ({hour_risk.iloc[1]} incidents)")
                    else:
                        risk_factors.append(f"⏰ Peak Risk Time: {peak_hour}:00 ({hour_risk.iloc[0]} incidents)")
            
            # Day of week risk
            if 'day_of_week' in df_filtered.columns:
                day_risk = df_filtered['day_of_week'].value_counts()
                if len(day_risk) > 0:
                    risky_day = day_risk.index[0]
                    risk_factors.append(f"📅 Highest Risk Day: {risky_day} ({day_risk.iloc[0]} incidents)")
            
            # Department comparison (only show if analyzing specific department)
            if selected_department != 'All Departments' and 'department' in df.columns:
                total_org_incidents = len(df)
                dept_incidents = len(df_filtered)
                org_monthly_avg = total_org_incidents / max(df['month'].nunique(), 1) if 'month' in df.columns else 0
                dept_monthly_avg = dept_incidents / max(df_filtered['month'].nunique(), 1) if 'month' in df_filtered.columns else 0
                
                if org_monthly_avg > 0:
                    dept_vs_org_ratio = dept_monthly_avg / org_monthly_avg
                    if dept_vs_org_ratio > 1.2:
                        risk_factors.append(f"⚠️ Department Risk: {dept_vs_org_ratio:.1f}x higher than organization average")
                    elif dept_vs_org_ratio < 0.8:
                        risk_factors.append(f"✅ Department Performance: {(1/dept_vs_org_ratio):.1f}x better than organization average")
            
            # Injury severity prediction
            if 'was_injured' in df_filtered.columns and len(df_filtered) > 0:
                try:
                    if df_filtered['was_injured'].dtype == 'bool':
                        injury_count = df_filtered['was_injured'].sum()
                    else:
                        # For non-boolean, create a binary injury indicator
                        injury_count = df_filtered['was_injured'].apply(
                            lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured'] else 0
                        ).sum()
                    
                    if injury_count > 0:
                        injury_pct = (injury_count / len(df_filtered)) * 100
                        risk_factors.append(f"🏥 Injury Rate: {injury_pct:.1f}% ({injury_count} injuries)")
                except:
                    pass
            
            if not risk_factors:
                risk_factors.append("✅ Insufficient data for detailed risk analysis")
            
            for factor in risk_factors:
                st.write(factor)

    # --- NEW: TOP 5 LOCATION PREDICTIONS (Department-Specific) ---
    if 'location' in df_filtered.columns and 'incident_date' in df_filtered.columns and len(df_filtered) > 10:
        st.subheader(f"🗺️ Location Risk Forecasting - {selected_department}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📍 Top 5 Predicted High-Risk Locations (Next 3 Months)**")
            
            # Calculate location trends using numpy and pandas only
            df_temp = df_filtered.copy()
            df_temp['year_month'] = df_temp['incident_date'].dt.to_period('M')
            
            location_trends = df_temp.groupby(['location', 'year_month']).size().reset_index(name='incidents')
            
            # Calculate weighted prediction scores
            location_monthly_avg = location_trends.groupby('location')['incidents'].mean().sort_values(ascending=False)
            
            # Get recent trend (last 3 months if available)
            recent_periods = sorted(location_trends['year_month'].unique())[-3:]
            recent_data = location_trends[location_trends['year_month'].isin(recent_periods)]
            recent_avg = recent_data.groupby('location')['incidents'].mean()
            
            # Calculate trend slope for each location using numpy
            location_trend_scores = {}
            for location in df_filtered['location'].unique():
                loc_data = location_trends[location_trends['location'] == location].sort_values('year_month')
                if len(loc_data) >= 3:
                    # Calculate trend using polyfit
                    x = np.arange(len(loc_data))
                    y = loc_data['incidents'].values
                    if len(y) > 1 and np.var(y) > 0:
                        trend_coeff = np.polyfit(x, y, 1)[0]  # Slope of linear trend
                    else:
                        trend_coeff = 0
                else:
                    trend_coeff = 0
                
                # Combine historical average, recent performance, and trend
                hist_avg = location_monthly_avg.get(location, 0)
                recent_perf = recent_avg.get(location, 0)
                
                # Weighted score: 50% historical, 30% recent, 20% trend
                prediction_score = (hist_avg * 0.5 + recent_perf * 0.3 + 
                                  max(0, hist_avg + trend_coeff) * 0.2)
                location_trend_scores[location] = prediction_score
            
            # Sort by prediction score
            top_5_locations = dict(sorted(location_trend_scores.items(), 
                                        key=lambda x: x[1], reverse=True)[:5])
            
            for i, (location, score) in enumerate(top_5_locations.items(), 1):
                historical_total = df_filtered[df_filtered['location'] == location].shape[0]
                st.write(f"{i}. **{location}** - Predicted: ~{score:.1f} incidents/month (Historical: {historical_total} total)")
        
        with col2:
            # Visual representation
            if len(top_5_locations) > 0:
                locations = list(top_5_locations.keys())
                scores = list(top_5_locations.values())
                
                fig = px.bar(x=scores, y=locations,
                            orientation='h', 
                            title=f"Top 5 Predicted High-Risk Locations - {selected_department}",
                            labels={'x': 'Predicted Monthly Incidents', 'y': 'Location'})
                st.plotly_chart(fig, use_container_width=True)

    # --- NEW: INCIDENT CATEGORY TREND PREDICTIONS (Department-Specific) ---
    category_col = None
    if 'label' in df_filtered.columns:
        category_col = 'label'
    elif 'category' in df_filtered.columns:
        category_col = 'category'
    elif 'incident_category' in df_filtered.columns:
        category_col = 'incident_category'
    
    if category_col and 'incident_date' in df_filtered.columns and len(df_filtered) > 10:
        st.subheader(f"🏷️ Incident Category Trend Predictions - {selected_department}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Category Trend Forecast (Next 3 Months)**")
            
            # Analyze category trends using numpy for trend calculation
            df_temp = df_filtered.copy()
            df_temp['year_month'] = df_temp['incident_date'].dt.to_period('M')
            
            category_trends = df_temp.groupby([category_col, 'year_month']).size().reset_index(name='incidents')
            
            # Calculate growth rates using numpy
            category_growth = {}
            for category in df_filtered[category_col].unique():
                cat_data = category_trends[category_trends[category_col] == category].sort_values('year_month')
                if len(cat_data) >= 3:
                    # Use numpy to calculate trend slope
                    x = np.arange(len(cat_data))
                    y = cat_data['incidents'].values
                    if len(y) > 1 and np.var(y) > 0:
                        trend_slope = np.polyfit(x, y, 1)[0]  # Linear trend slope
                        recent_avg = cat_data['incidents'].tail(2).mean()
                        historical_avg = cat_data['incidents'].mean()
                        
                        # Calculate percentage change based on trend
                        if historical_avg > 0:
                            growth_rate = (trend_slope / historical_avg) * 100
                        else:
                            growth_rate = 0
                    else:
                        recent_avg = cat_data['incidents'].mean()
                        growth_rate = 0
                elif len(cat_data) >= 2:
                    recent_avg = cat_data['incidents'].tail(1).iloc[0]
                    historical_avg = cat_data['incidents'].mean()
                    growth_rate = ((recent_avg - historical_avg) / historical_avg) * 100 if historical_avg > 0 else 0
                else:
                    recent_avg = cat_data['incidents'].mean() if len(cat_data) > 0 else 0
                    growth_rate = 0
                
                category_growth[category] = {
                    'current_avg': recent_avg,
                    'growth_rate': growth_rate,
                    'total_incidents': df_filtered[df_filtered[category_col] == category].shape[0]
                }
            
            # Sort by predicted risk (combination of current incidents and growth)
            sorted_categories = sorted(category_growth.items(), 
                                     key=lambda x: x[1]['current_avg'] + (x[1]['growth_rate'] * 0.1), 
                                     reverse=True)
            
            st.write("**Top Categories by Predicted Risk:**")
            for i, (category, data) in enumerate(sorted_categories[:5], 1):
                trend_emoji = "📈" if data['growth_rate'] > 5 else "📉" if data['growth_rate'] < -5 else "➡️"
                st.write(f"{i}. **{category}** {trend_emoji}")
                st.write(f"   Current: ~{data['current_avg']:.1f}/month | Growth: {data['growth_rate']:+.1f}% | Total: {data['total_incidents']}")
        
        with col2:
            # Category trend visualization
            if len(sorted_categories) > 0:
                categories = [item[0] for item in sorted_categories[:5]]
                growth_rates = [item[1]['growth_rate'] for item in sorted_categories[:5]]
                
                fig = go.Figure(data=go.Bar(
                    x=categories,
                    y=growth_rates,
                    marker_color=['red' if x > 5 else 'green' if x < -5 else 'orange' for x in growth_rates]
                ))
                fig.update_layout(title=f"Category Growth Rate Trends - {selected_department}",
                                xaxis_title="Category", 
                                yaxis_title="Growth Rate (%)")
                st.plotly_chart(fig, use_container_width=True)

    # --- ADVANCED TIME ANALYSIS (Department-Specific) ---
    if 'incident_date' in df_filtered.columns:
        st.subheader(f"⏰ Time Pattern Analysis - {selected_department}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Heatmap by hour and day
            if 'hour' in df_filtered.columns and 'day_of_week' in df_filtered.columns:
                heatmap_data = df_filtered.groupby(['day_of_week', 'hour']).size().reset_index(name='incidents')
                heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='incidents').fillna(0)
                
                # Order days correctly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_pivot = heatmap_pivot.reindex(day_order)
                
                fig = px.imshow(heatmap_pivot, 
                              title=f"Incident Heatmap: Day vs Hour - {selected_department}",
                              labels=dict(x="Hour of Day", y="Day of Week", color="Incidents"),
                              color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Seasonal trends
            if 'month' in df_filtered.columns:
                monthly_incidents = df_filtered.groupby('month').size().reset_index(name='incidents')
                monthly_incidents['month_name'] = monthly_incidents['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
                
                fig = px.bar(monthly_incidents, x='month_name', y='incidents',
                           title=f"Seasonal Incident Patterns - {selected_department}")
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Weekly patterns
            if 'day_of_week' in df_filtered.columns:
                daily_incidents = df_filtered['day_of_week'].value_counts().reindex(
                    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                ).reset_index()
                daily_incidents.columns = ['day', 'incidents']
                
                fig = px.line(daily_incidents, x='day', y='incidents', 
                            markers=True, title=f"Weekly Incident Pattern - {selected_department}")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

    # --- ENHANCED DEPARTMENT & LOCATION INSIGHTS ---
    st.subheader("🏢 Department & Location Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if selected_department == 'All Departments' and 'department' in df.columns:
            # Show department comparison when viewing all departments
            # Create a safe aggregation function
            agg_dict = {df.columns[0]: 'count'}
            
            # Add injury rate if injury column exists
            if 'was_injured' in df.columns:
                if df['was_injured'].dtype == 'bool':
                    agg_dict['was_injured'] = 'mean'
                else:
                    # Create binary injury indicator for aggregation
                    df['injury_binary'] = df['was_injured'].apply(
                        lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured'] else 0
                    )
                    agg_dict['injury_binary'] = 'mean'
            
            dept_analysis = df.groupby('department').agg(agg_dict).reset_index()
            
            if 'was_injured' in df.columns or 'injury_binary' in dept_analysis.columns:
                injury_rate_col = 'was_injured' if 'was_injured' in dept_analysis.columns else 'injury_binary'
                dept_analysis.columns = ['department', 'incident_count', 'injury_rate']
                dept_analysis['injury_rate'] *= 100
                
                # Enhanced bubble chart with color coding and better legend
                fig = px.scatter(dept_analysis, x='incident_count', y='injury_rate', 
                               size='incident_count', hover_name='department',
                               title="Department Risk Matrix (with Color Legend)",
                               labels={'incident_count': 'Total Incidents', 
                                     'injury_rate': 'Injury Rate (%)'},
                               color='injury_rate',
                               color_continuous_scale='Reds',
                               size_max=30)
                
                # Add color bar title
                fig.update_coloraxes(colorbar_title="Injury Rate (%)")
                
                # Add quadrant lines for better interpretation
                max_incidents = dept_analysis['incident_count'].max()
                avg_injury_rate = dept_analysis['injury_rate'].mean()
                
                fig.add_hline(y=avg_injury_rate, line_dash="dash", line_color="gray", 
                            annotation_text=f"Avg Injury Rate ({avg_injury_rate:.1f}%)")
                fig.add_vline(x=max_incidents/2, line_dash="dash", line_color="gray",
                            annotation_text="Mid Incident Count")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation guide
                st.write("**📊 Matrix Interpretation:**")
                st.write("• **Top Right (Red)**: High incidents + High injury rate = Critical priority")
                st.write("• **Top Left**: Low incidents + High injury rate = Severe incidents focus")
                st.write("• **Bottom Right**: High incidents + Low injury rate = Prevention focus")
                st.write("• **Bottom Left**: Low risk departments")
                
            else:
                fig = px.bar(dept_analysis, x='department', y=dept_analysis.columns[1],
                           title="Incidents by Department")
                st.plotly_chart(fig, use_container_width=True)
        
        elif 'location' in df_filtered.columns:
            # Show location analysis for selected department
            location_incidents = df_filtered['location'].value_counts().head(10).reset_index()
            location_incidents.columns = ['location', 'incidents']
            
            fig = px.bar(location_incidents, x='incidents', y='location', 
                        orientation='h', title=f"Top 10 Incident Locations - {selected_department}")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'location' in df_filtered.columns:
            # Always show location analysis in the right column
            if selected_department != 'All Departments':
                # For specific department, show location breakdown with injury rates if available
                if 'was_injured' in df_filtered.columns:
                    try:
                        if df_filtered['was_injured'].dtype == 'bool':
                            location_analysis = df_filtered.groupby('location').agg({
                                'was_injured': ['count', 'sum', 'mean']
                            }).reset_index()
                            location_analysis.columns = ['location', 'total_incidents', 'injuries', 'injury_rate']
                        else:
                            # Create binary injury indicator
                            df_filtered['injury_binary'] = df_filtered['was_injured'].apply(
                                lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured'] else 0
                            )
                            location_analysis = df_filtered.groupby('location').agg({
                                'injury_binary': ['count', 'sum', 'mean']
                            }).reset_index()
                            location_analysis.columns = ['location', 'total_incidents', 'injuries', 'injury_rate']
                        
                        location_analysis['injury_rate'] *= 100
                        location_analysis = location_analysis[location_analysis['total_incidents'] >= 2].head(10)  # Filter locations with at least 2 incidents
                        
                        if len(location_analysis) > 0:
                            fig = px.scatter(location_analysis, x='total_incidents', y='injury_rate', 
                                           size='total_incidents', hover_name='location',
                                           title=f"Location Risk Matrix - {selected_department}",
                                           labels={'total_incidents': 'Total Incidents', 
                                                 'injury_rate': 'Injury Rate (%)'},
                                           color='injury_rate',
                                           color_continuous_scale='Reds')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough data for location injury analysis")
                    except:
                        # Fallback to simple location count
                        location_incidents = df_filtered['location'].value_counts().head(10).reset_index()
                        location_incidents.columns = ['location', 'incidents']
                        
                        fig = px.pie(location_incidents, names='location', values='incidents',
                                   title=f"Location Distribution - {selected_department}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # Simple location distribution for department
                    location_incidents = df_filtered['location'].value_counts().head(10).reset_index()
                    location_incidents.columns = ['location', 'incidents']
                    
                    fig = px.pie(location_incidents, names='location', values='incidents',
                               title=f"Location Distribution - {selected_department}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # For all departments, show top locations across organization
                location_incidents = df_filtered['location'].value_counts().head(10).reset_index()
                location_incidents.columns = ['location', 'incidents']
                
                fig = px.bar(location_incidents, x='incidents', y='location', 
                            orientation='h', title="Top 10 Incident Locations - All Departments")
                st.plotly_chart(fig, use_container_width=True)

    # --- INCIDENT CATEGORIZATION & ANALYSIS (Department-Specific) ---
    if category_col:
        st.subheader(f"🏷️ Incident Category Analysis - {selected_department}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = df_filtered[category_col].value_counts().reset_index()
            category_counts.columns = ['category', 'count']
            
            fig = px.pie(category_counts, names='category', values='count',
                        title=f"Incident Distribution by Category - {selected_department}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category trends over time if date is available
            if 'incident_date' in df_filtered.columns:
                category_trends = df_filtered.groupby([df_filtered['incident_date'].dt.date, category_col]).size().reset_index(name='count')
                
                fig = px.line(category_trends, x='incident_date', y='count', 
                            color=category_col, title=f"Category Trends Over Time - {selected_department}")
                st.plotly_chart(fig, use_container_width=True)

    # --- ENHANCED TEXT ANALYSIS (Department-Specific) ---
    if 'incident_description' in df_filtered.columns:
        st.subheader(f"📝 Advanced Text Analytics - {selected_department}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**☁️ Incident Description Word Cloud**")
            text = " ".join(str(desc) for desc in df_filtered['incident_description'].dropna())
            if text.strip():
                # Remove common stop words
                stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
                words = text.lower().split()
                filtered_text = " ".join([word for word in words if word not in stop_words and len(word) > 2])
                
                wordcloud = WordCloud(width=400, height=300, background_color="white").generate(filtered_text)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("No text available for analysis.")
        
        with col2:
            st.write("**🔍 Key Terms Frequency**")
            if text.strip():
                # Extract key terms
                words = text.lower().split()
                word_freq = pd.Series(words).value_counts().head(10)
                word_freq = word_freq[word_freq.index.str.len() > 3]  # Filter short words
                
                fig = px.bar(x=word_freq.values, y=word_freq.index, 
                           orientation='h', title="Most Common Terms")
                st.plotly_chart(fig, use_container_width=True)

    # --- ENHANCED BUSINESS RECOMMENDATIONS (Department-Specific) ---
    st.subheader(f"💡 AI-Driven Recommendations - {selected_department}")
    
    recommendations = []
    
    # Time-based recommendations
    if 'hour' in df_filtered.columns:
        peak_hours = df_filtered['hour'].value_counts().head(3)
        if len(peak_hours) > 0:
            if peak_hours.index[0] == 0 and len(peak_hours) > 1:
                # Handle midnight peak by focusing on second highest
                recommendations.append(f"🕐 **Peak Risk Hours**: Focus on {peak_hours.index[1]}:00-{peak_hours.index[1]+1}:00 ({peak_hours.iloc[1]} incidents). Note: Midnight times may indicate data quality issues.")
            else:
                recommendations.append(f"🕐 **Peak Risk Hours**: Increase safety supervision during {peak_hours.index[0]}:00-{peak_hours.index[0]+1}:00 ({peak_hours.iloc[0]} incidents)")
    
    # Location recommendations
    if 'location' in df_filtered.columns:
        high_risk_locations = df_filtered['location'].value_counts().head(2)
        if len(high_risk_locations) > 0:
            recommendations.append(f"📍 **Focus Location**: Prioritize safety measures at {high_risk_locations.index[0]} ({high_risk_locations.iloc[0]} incidents)")
    
    # Department-specific vs organizational comparison
    if selected_department != 'All Departments' and 'department' in df.columns:
        # Compare department performance to organizational average
        total_org_incidents = len(df)
        org_monthly_avg = total_org_incidents / max(df['month'].nunique(), 1) if 'month' in df.columns else 0
        dept_monthly_avg = len(df_filtered) / max(df_filtered['month'].nunique(), 1) if 'month' in df_filtered.columns else 0
        
        if org_monthly_avg > 0:
            dept_vs_org_ratio = dept_monthly_avg / org_monthly_avg
            if dept_vs_org_ratio > 1.5:
                recommendations.append(f"🚨 **Critical**: Department incident rate is {dept_vs_org_ratio:.1f}x higher than organization average - immediate intervention needed")
            elif dept_vs_org_ratio > 1.2:
                recommendations.append(f"⚠️ **Above Average**: Department incident rate is {dept_vs_org_ratio:.1f}x higher than organization average")
            elif dept_vs_org_ratio < 0.7:
                recommendations.append(f"✅ **Best Practice**: Department performs {(1/dept_vs_org_ratio):.1f}x better than organization average - share best practices")
    
    # Predictive recommendations
    if 'location' in df_filtered.columns and len(df_filtered) > 10:
        recommendations.append("🗺️ **Location Focus**: Review top predicted high-risk locations above for targeted interventions")
    
    if category_col and len(df_filtered) > 10:
        recommendations.append("📈 **Category Trends**: Monitor growing incident categories identified in trend analysis")
    
    # Injury prevention (department-specific)
    if 'was_injured' in df_filtered.columns and injury_rate > 0:
        if injury_rate > 15:
            recommendations.append(f"🏥 **Critical Injury Rate**: {injury_rate:.1f}% injury rate in {selected_department} requires immediate safety protocol review")
        elif injury_rate > 8:
            recommendations.append(f"⚠️ **High Injury Rate**: {injury_rate:.1f}% injury rate in {selected_department} needs attention")
        
        # Compare injury rate to organization if viewing specific department
        if selected_department != 'All Departments':
            org_injury_rate = 0
            try:
                if df['was_injured'].dtype == 'bool':
                    org_injury_rate = (df['was_injured'].sum() / len(df)) * 100
                else:
                    injured_count = df['was_injured'].dropna().apply(
                        lambda x: str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured']
                    ).sum()
                    org_injury_rate = (injured_count / len(df)) * 100
                
                if org_injury_rate > 0:
                    if injury_rate > org_injury_rate * 1.5:
                        recommendations.append(f"🎯 **Department Focus**: {selected_department} injury rate ({injury_rate:.1f}%) is {(injury_rate/org_injury_rate):.1f}x higher than organization average ({org_injury_rate:.1f}%)")
                    elif injury_rate < org_injury_rate * 0.7:
                        recommendations.append(f"✅ **Safety Excellence**: {selected_department} injury rate ({injury_rate:.1f}%) is significantly better than organization average ({org_injury_rate:.1f}%)")
            except:
                pass
    
    # Trend-based recommendations
    if 'incident_date' in df_filtered.columns and len(df_filtered) > 20:
        recent_30d = df_filtered[df_filtered['incident_date'] >= (datetime.now() - timedelta(days=30))]
        if len(recent_30d) > len(df_filtered) * 0.4:  # If 40% of incidents in last 30 days
            recommendations.append(f"📈 **Trend Alert**: Recent surge in {selected_department} incidents detected - conduct immediate safety audit")
    
    # Department-specific data quality recommendations
    if valid_time_column is None and any('time' in col.lower() for col in df_filtered.columns):
        recommendations.append(f"⚠️ **Data Quality**: Consider improving time data collection in {selected_department} - many incidents show 00:00:00 timestamps")
    
    # Monthly trend recommendation
    if incident_trend > 20:
        recommendations.append(f"🚨 **Monthly Surge**: {incident_trend:+.1f}% increase this month in {selected_department} - immediate review recommended")
    elif incident_trend < -20:
        recommendations.append(f"✅ **Improvement**: {abs(incident_trend):.1f}% decrease this month in {selected_department} - analyze what's working")
    
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations[:8], 1):  # Show top 8 recommendations
            st.write(f"{i}. {rec}")
    else:
        st.write(f"📊 **Data Analysis**: Upload more comprehensive data for personalized recommendations for {selected_department}")

    # --- DEPARTMENT COMPARISON SECTION (Only when viewing all departments) ---
    if selected_department == 'All Departments' and 'department' in df.columns:
        st.subheader("🔄 Department Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Department ranking by incidents
            dept_incidents = df.groupby('department').size().reset_index(name='incidents')
            dept_incidents = dept_incidents.sort_values('incidents', ascending=True)
            
            fig = px.bar(dept_incidents, x='incidents', y='department',
                        orientation='h', title="Department Incident Ranking",
                        color='incidents', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Department monthly trend comparison
            if 'incident_date' in df.columns:
                dept_monthly = df.groupby(['department', 'year', 'month']).size().reset_index(name='incidents')
                dept_monthly['date'] = pd.to_datetime(dept_monthly[['year', 'month']].assign(day=1))
                
                # Show trends for top 5 departments by incident count
                top_depts = dept_incidents.tail(5)['department'].tolist()
                dept_monthly_filtered = dept_monthly[dept_monthly['department'].isin(top_depts)]
                
                fig = px.line(dept_monthly_filtered, x='date', y='incidents', 
                            color='department', title="Top 5 Departments - Monthly Trends")
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Upload your incident CSV file to unlock powerful safety analytics and predictive insights")
    st.write("**Expected columns for optimal analysis:**")
    st.write("• Reporter Name, Person Involved, Incident Date & Time")
    st.write("• Department & Location, Incident Description")
    st.write("• Label/Category, Injury Information (was_injured: boolean)")
    
