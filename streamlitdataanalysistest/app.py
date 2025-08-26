#analytics app 27/08/2025 - Enhanced Version

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
    
    st.subheader("🔎 Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.head())
    with col2:
        st.write("**Data Quality Summary:**")
        missing_data = df.isnull().sum()
        st.write(f"Total Records: {len(df)}")
        st.write(f"Columns: {len(df.columns)}")
        if missing_data.sum() > 0:
            st.write("Missing Values:")
            for col, missing in missing_data[missing_data > 0].items():
                st.write(f"  • {col}: {missing} ({missing/len(df)*100:.1f}%)")

    # --- Enhanced KPI Dashboard ---
    st.subheader("📈 Executive Dashboard")
    
    # Calculate advanced metrics
    total_incidents = len(df)
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    # Filter for current period if date exists
    if 'incident_date' in df.columns:
        df_current_month = df[(df['month'] == current_month) & (df['year'] == current_year)]
        df_last_month = df[(df['month'] == current_month-1) & (df['year'] == current_year)]
        
        # Calculate trends
        current_month_incidents = len(df_current_month)
        last_month_incidents = len(df_last_month)
        
        if last_month_incidents > 0:
            incident_trend = ((current_month_incidents - last_month_incidents) / last_month_incidents) * 100
        else:
            incident_trend = 0
    else:
        incident_trend = 0
        current_month_incidents = 0
    
    # Injury rate calculation - now specifically for 'was_injured' column
    injury_rate = 0
    if 'was_injured' in df.columns:
        # Handle different data types (boolean, text, numbers)
        try:
            if df['was_injured'].dtype == 'bool':
                injury_rate = (df['was_injured'].sum() / len(df)) * 100 if len(df) > 0 else 0
            elif df['was_injured'].dtype == 'object':
                # Count 'Yes', 'True', '1', etc. as injuries
                injured_count = df['was_injured'].dropna().apply(
                    lambda x: str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured']
                ).sum()
                injury_rate = (injured_count / len(df)) * 100 if len(df) > 0 else 0
            else:
                # Numeric - count values > 0
                injury_rate = (df['was_injured'].fillna(0).astype(float).gt(0).sum() / len(df)) * 100 if len(df) > 0 else 0
        except:
            injury_rate = 0
    
    # Display KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Incidents", total_incidents)
    col2.metric("This Month", current_month_incidents, f"{incident_trend:+.1f}%")
    col3.metric("Injury Rate", f"{injury_rate:.1f}%")
    col4.metric("Departments", df['department'].nunique() if 'department' in df.columns else 0)
    col5.metric("Locations", df['location'].nunique() if 'location' in df.columns else 0)

    # --- ENHANCED PREDICTIVE ANALYTICS ---
    st.subheader("🔮 Advanced Predictive Analytics & Forecasting")
    
    if 'incident_date' in df.columns and len(df) > 10:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Incident Trend Forecast**")
            
            # Create monthly aggregation for forecasting
            df_temp = df.copy()
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
                fig.update_layout(title="3-Month Trend-Based Incident Forecast", 
                                xaxis_title="Date", yaxis_title="Incidents")
                st.plotly_chart(fig, use_container_width=True)
                
                # Display predictions with confidence note
                st.write("**📈 Next 3 Months Predictions (Trend Analysis):**")
                for i, (date, pred) in enumerate(zip(future_dates, future_predictions)):
                    st.write(f"• {date.strftime('%B %Y')}: ~{int(pred)} incidents")
                st.write("*Based on historical trend analysis and recent patterns*")
        
        with col2:
            st.write("**🎯 Enhanced Risk Indicators**")
            
            # Calculate risk scores
            risk_factors = []
            
            # Department risk
            if 'department' in df.columns:
                dept_incidents = df['department'].value_counts()
                high_risk_dept = dept_incidents.index[0] if len(dept_incidents) > 0 else "Unknown"
                risk_factors.append(f"🔴 High Risk Department: {high_risk_dept} ({dept_incidents.iloc[0]} incidents)")
            
            # Enhanced time-based risk with 00:00:00 handling
            if 'hour' in df.columns:
                hour_risk = df['hour'].value_counts()
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
            if 'day_of_week' in df.columns:
                day_risk = df['day_of_week'].value_counts()
                if len(day_risk) > 0:
                    risky_day = day_risk.index[0]
                    risk_factors.append(f"📅 Highest Risk Day: {risky_day} ({day_risk.iloc[0]} incidents)")
            
            # Injury severity prediction
            if 'was_injured' in df.columns and 'department' in df.columns:
                try:
                    if df['was_injured'].dtype == 'bool':
                        injury_by_dept = df.groupby('department')['was_injured'].mean().sort_values(ascending=False)
                    else:
                        # For non-boolean, create a binary injury indicator
                        df['injury_binary'] = df['was_injured'].apply(
                            lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured'] else 0
                        )
                        injury_by_dept = df.groupby('department')['injury_binary'].mean().sort_values(ascending=False)
                    
                    if len(injury_by_dept) > 0 and injury_by_dept.iloc[0] > 0:
                        high_injury_dept = injury_by_dept.index[0]
                        injury_pct = injury_by_dept.iloc[0] * 100
                        risk_factors.append(f"🏥 Injury Hotspot: {high_injury_dept} ({injury_pct:.1f}% injury rate)")
                except:
                    pass
            
            for factor in risk_factors:
                st.write(factor)

    # --- NEW: TOP 5 LOCATION PREDICTIONS ---
    if 'location' in df.columns and 'incident_date' in df.columns and len(df) > 20:
        st.subheader("🗺️ Location Risk Forecasting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📍 Top 5 Predicted High-Risk Locations (Next 3 Months)**")
            
            # Calculate location trends using numpy and pandas only
            df_temp = df.copy()
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
            for location in df['location'].unique():
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
                historical_total = df[df['location'] == location].shape[0]
                st.write(f"{i}. **{location}** - Predicted: ~{score:.1f} incidents/month (Historical: {historical_total} total)")
        
        with col2:
            # Visual representation
            if len(top_5_locations) > 0:
                locations = list(top_5_locations.keys())
                scores = list(top_5_locations.values())
                
                fig = px.bar(x=scores, y=locations,
                            orientation='h', 
                            title="Top 5 Predicted High-Risk Locations",
                            labels={'x': 'Predicted Monthly Incidents', 'y': 'Location'})
                st.plotly_chart(fig, use_container_width=True)

    # --- NEW: INCIDENT CATEGORY TREND PREDICTIONS ---
    category_col = None
    if 'label' in df.columns:
        category_col = 'label'
    elif 'category' in df.columns:
        category_col = 'category'
    elif 'incident_category' in df.columns:
        category_col = 'incident_category'
    
    if category_col and 'incident_date' in df.columns and len(df) > 15:
        st.subheader("🏷️ Incident Category Trend Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Category Trend Forecast (Next 3 Months)**")
            
            # Analyze category trends using numpy for trend calculation
            df_temp = df.copy()
            df_temp['year_month'] = df_temp['incident_date'].dt.to_period('M')
            
            category_trends = df_temp.groupby([category_col, 'year_month']).size().reset_index(name='incidents')
            
            # Calculate growth rates using numpy
            category_growth = {}
            for category in df[category_col].unique():
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
                    'total_incidents': df[df[category_col] == category].shape[0]
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
                fig.update_layout(title="Category Growth Rate Trends",
                                xaxis_title="Category", 
                                yaxis_title="Growth Rate (%)")
                st.plotly_chart(fig, use_container_width=True)

    # --- ADVANCED TIME ANALYSIS ---
    if 'incident_date' in df.columns:
        st.subheader("⏰ Time Pattern Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Heatmap by hour and day
            if 'hour' in df.columns and 'day_of_week' in df.columns:
                heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='incidents')
                heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='incidents').fillna(0)
                
                # Order days correctly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_pivot = heatmap_pivot.reindex(day_order)
                
                fig = px.imshow(heatmap_pivot, 
                              title="Incident Heatmap: Day vs Hour",
                              labels=dict(x="Hour of Day", y="Day of Week", color="Incidents"),
                              color_continuous_scale="Reds")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Seasonal trends
            if 'month' in df.columns:
                monthly_incidents = df.groupby('month').size().reset_index(name='incidents')
                monthly_incidents['month_name'] = monthly_incidents['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
                
                fig = px.bar(monthly_incidents, x='month_name', y='incidents',
                           title="Seasonal Incident Patterns")
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Weekly patterns
            if 'day_of_week' in df.columns:
                daily_incidents = df['day_of_week'].value_counts().reindex(
                    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                ).reset_index()
                daily_incidents.columns = ['day', 'incidents']
                
                fig = px.line(daily_incidents, x='day', y='incidents', 
                            markers=True, title="Weekly Incident Pattern")
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

    # --- ENHANCED DEPARTMENT & LOCATION INSIGHTS ---
    st.subheader("🏢 Department & Location Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'department' in df.columns:
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
    
    with col2:
        if 'location' in df.columns:
            location_incidents = df['location'].value_counts().head(10).reset_index()
            location_incidents.columns = ['location', 'incidents']
            
            fig = px.bar(location_incidents, x='incidents', y='location', 
                        orientation='h', title="Top 10 Incident Locations")
            st.plotly_chart(fig, use_container_width=True)

    # --- INCIDENT CATEGORIZATION & ANALYSIS ---
    if category_col:
        st.subheader("🏷️ Incident Category Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = df[category_col].value_counts().reset_index()
            category_counts.columns = ['category', 'count']
            
            fig = px.pie(category_counts, names='category', values='count',
                        title="Incident Distribution by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category trends over time if date is available
            if 'incident_date' in df.columns:
                category_trends = df.groupby([df['incident_date'].dt.date, category_col]).size().reset_index(name='count')
                
                fig = px.line(category_trends, x='incident_date', y='count', 
                            color=category_col, title="Category Trends Over Time")
                st.plotly_chart(fig, use_container_width=True)

    # --- ENHANCED TEXT ANALYSIS ---
    if 'incident_description' in df.columns:
        st.subheader("📝 Advanced Text Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**☁️ Incident Description Word Cloud**")
            text = " ".join(str(desc) for desc in df['incident_description'].dropna())
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

    # --- ENHANCED BUSINESS RECOMMENDATIONS ---
    st.subheader("💡 AI-Driven Recommendations")
    
    recommendations = []
    
    # Time-based recommendations
    if 'hour' in df.columns:
        peak_hours = df['hour'].value_counts().head(3)
        if len(peak_hours) > 0:
            if peak_hours.index[0] == 0 and len(peak_hours) > 1:
                # Handle midnight peak by focusing on second highest
                recommendations.append(f"🕐 **Peak Risk Hours**: Focus on {peak_hours.index[1]}:00-{peak_hours.index[1]+1}:00 ({peak_hours.iloc[1]} incidents). Note: Midnight times may indicate data quality issues.")
            else:
                recommendations.append(f"🕐 **Peak Risk Hours**: Increase safety supervision during {peak_hours.index[0]}:00-{peak_hours.index[0]+1}:00 ({peak_hours.iloc[0]} incidents)")
    
    # Department recommendations
    if 'department' in df.columns:
        high_risk_depts = df['department'].value_counts().head(2)
        if len(high_risk_depts) > 0:
            recommendations.append(f"🏢 **Focus Area**: Prioritize safety training in {high_risk_depts.index[0]} department ({high_risk_depts.iloc[0]} incidents)")
    
    # Predictive recommendations
    if 'location' in df.columns and len(df) > 20:
        recommendations.append("🗺️ **Location Focus**: Review top 5 predicted high-risk locations above for targeted interventions")
    
    if category_col and len(df) > 15:
        recommendations.append("📈 **Category Trends**: Monitor growing incident categories identified in trend analysis")
    
    # Injury prevention
    if 'was_injured' in df.columns and injury_rate > 0:
        if injury_rate > 10:
            recommendations.append(f"🏥 **Critical**: {injury_rate:.1f}% injury rate requires immediate safety protocol review")
        
        # Department-specific injury recommendations
        if 'department' in df.columns:
            try:
                if df['was_injured'].dtype == 'bool':
                    dept_injury_rates = df.groupby('department')['was_injured'].agg(['sum', 'count', 'mean'])
                else:
                    # Create binary injury indicator
                    df['injury_binary'] = df['was_injured'].apply(
                        lambda x: 1 if pd.notna(x) and str(x).strip().lower() in ['yes', 'true', '1', 'y', 'injured'] else 0
                    )
                    dept_injury_rates = df.groupby('department')['injury_binary'].agg(['sum', 'count', 'mean'])
                
                dept_injury_rates = dept_injury_rates[dept_injury_rates['count'] >= 3]  # Only departments with 3+ incidents
                if len(dept_injury_rates) > 0 and dept_injury_rates['mean'].max() > 0:
                    worst_dept = dept_injury_rates['mean'].idxmax()
                    worst_rate = dept_injury_rates.loc[worst_dept, 'mean'] * 100
                    recommendations.append(f"🎯 **Targeted Intervention**: {worst_dept} has {worst_rate:.1f}% injury rate - implement enhanced safety measures")
            except:
                pass
    
    # Trend-based recommendations
    if 'incident_date' in df.columns and len(df) > 30:
        recent_30d = df[df['incident_date'] >= (datetime.now() - timedelta(days=30))]
        if len(recent_30d) > len(df) * 0.3:  # If 30% of incidents in last 30 days
            recommendations.append("📈 **Trend Alert**: Recent surge in incidents detected - conduct immediate safety audit")
    
    # Data quality recommendations
    if valid_time_column is None and any('time' in col.lower() for col in df.columns):
        recommendations.append("⚠️ **Data Quality**: Consider improving time data collection - many incidents show 00:00:00 timestamps")
    
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations[:7], 1):  # Increased to top 7
            st.write(f"{i}. {rec}")
    else:
        st.write("📊 **Data Analysis**: Upload more comprehensive data for personalized recommendations")

    # --- EXPORT INSIGHTS ---
    st.subheader("📤 Export Analytics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Report"):
            st.success("Analytics report generated! (Feature in development)")
    
    with col2:
        if st.button("📧 Email Insights"):
            st.success("Insights email sent! (Feature in development)")
    
    with col3:
        if st.button("📅 Schedule Reports"):
            st.success("Report scheduling configured! (Feature in development)")

else:
    st.info("👆 Upload your incident CSV file to unlock powerful safety analytics and predictive insights")
    st.write("**Expected columns for optimal analysis:**")
    st.write("• Reporter Name, Person Involved, Incident Date & Time")
    st.write("• Department & Location, Incident Description")
    st.write("• Label/Category, Injury Information (was_injured: boolean)")
    
    st.write("**🆕 New Features in this version:**")
    st.write("• ✅ Smart time column detection (handles 00:00:00 issue)")
    st.write("• ✅ Top 5 incident location predictions for next 3 months")
    st.write("• ✅ Incident category trend forecasting")
    st.write("• ✅ Enhanced department matrix with color legend")
    st.write("• ✅ Advanced trend-based incident forecasting (using numpy)")
    st.write("• ✅ Advanced risk assessment and recommendations")