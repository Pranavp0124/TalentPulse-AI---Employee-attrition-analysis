"""
app.py
======
TalentPulse AI - Enterprise Employee Attrition & Retention Intelligence Platform
A production-ready Streamlit application combining interactive workforce analytics,
exploratory data analysis, AI-powered risk assessment, batch scoring, and model governance.
"""

import os
import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import model helper and feature definitions
from attrition_model import (
    AttritionPredictor,
    train_and_save_model,
    ALL_FEATURES,
    FEATURES_NUM,
    FEATURES_CAT
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="TalentPulse AI | Employee Attrition Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished enterprise look
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
    }
    div[data-testid="stMetricLabel"] > div {
        color: #94A3B8 !important;
        font-weight: 500;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] > div {
        color: #F8FAFC !important;
        font-weight: 700;
        font-size: 1.8rem;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px 8px 0px 0px;
        padding: 0px 18px;
        font-weight: 600;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(37, 99, 235, 0.15) !important;
        color: #60A5FA !important;
        border-bottom: 2px solid #3B82F6 !important;
    }

    /* Cards and Containers */
    .highlight-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
    }
    .risk-high {
        color: #EF4444;
        font-weight: 700;
    }
    .risk-med {
        color: #F59E0B;
        font-weight: 700;
    }
    .risk-low {
        color: #10B981;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data & Model Loaders (Cached)
# ---------------------------------------------------------
@st.cache_data
def load_data(filepath="HR_Employee_Attrition.csv"):
    if not os.path.exists(filepath):
        # Fallback to local alternate or regenerate
        fallback = Path(__file__).parent / "HR_Employee_Attrition.csv"
        if fallback.exists():
            filepath = str(fallback)
        else:
            raise FileNotFoundError(f"Dataset '{filepath}' not found.")
    df = pd.read_csv(filepath)
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    # Ensure categorical Attrition has standard Yes/No
    if 'Attrition' in df.columns and df['Attrition'].dtype != 'object':
        df['Attrition'] = df['Attrition'].map({1: 'Yes', 0: 'No'})
    return df


@st.cache_resource
def load_model(model_path="model.pkl"):
    from attrition_model import load_predictor
    return load_predictor(model_path)


try:
    df_raw = load_data()
    predictor = load_model()
except Exception as e:
    st.error(f"Error initializing system: {e}")
    st.stop()


# ---------------------------------------------------------
# Sidebar Navigation & Filters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 💼 **TalentPulse AI**")
    st.caption("Workforce Retention & Attrition Intelligence")
    st.markdown("---")

    st.subheader("🔍 Cohort Filters")
    
    # Department Filter
    all_depts = sorted(df_raw['Department'].dropna().unique().tolist())
    selected_depts = st.multiselect("Department", options=all_depts, default=all_depts)
    
    # OverTime Filter
    overtime_options = ["All", "Yes (Overtime)", "No (Standard)"]
    selected_overtime = st.selectbox("Overtime Status", options=overtime_options, index=0)

    # Job Role Filter
    available_roles = sorted(df_raw[df_raw['Department'].isin(selected_depts)]['JobRole'].dropna().unique().tolist())
    selected_roles = st.multiselect("Job Role", options=available_roles, default=available_roles)

    # Age Slider
    min_age, max_age = int(df_raw['Age'].min()), int(df_raw['Age'].max())
    age_range = st.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    st.markdown("---")
    
    # Filter dataset
    filtered_df = df_raw[
        (df_raw['Department'].isin(selected_depts)) &
        (df_raw['JobRole'].isin(selected_roles)) &
        (df_raw['Age'].between(age_range[0], age_range[1]))
    ]
    if selected_overtime == "Yes (Overtime)":
        filtered_df = filtered_df[filtered_df['OverTime'] == 'Yes']
    elif selected_overtime == "No (Standard)":
        filtered_df = filtered_df[filtered_df['OverTime'] == 'No']

    st.info(f"📊 **Cohort Size:** {len(filtered_df):,} / {len(df_raw):,} employees")

    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(37, 99, 235, 0.15); border: 1px solid rgba(59, 130, 246, 0.35); border-radius: 10px; padding: 14px; margin-bottom: 12px;">
        <div style="font-size: 0.8rem; color: #93C5FD; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">🎓 College Capstone Showcase</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-top: 4px;">Developed by Pranav</div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 2px;">Machine Learning & HR Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Project Highlights:**
    - 🧠 **Model:** Soft-Voting Ensemble (LR + RF + GB)
    - 📊 **Dataset:** IBM HR Analytics (1,470 records)
    - 🎯 **Accuracy:** 81% \| **ROC-AUC:** 0.79
    - 💻 **Stack:** Python, Streamlit, Scikit-Learn, Plotly
    """)


# ---------------------------------------------------------
# Header & Navigation Tabs
# ---------------------------------------------------------
st.title("💼 Employee Attrition Analysis & Retention AI")
st.markdown("##### 🎓 **College Capstone Project** &nbsp;|&nbsp; Designed & Developed by **Pranav**")
st.caption("Empowering HR leaders with predictive intelligence to diagnose turnover drivers, retain top talent, and optimize organizational health.")

tab_exec, tab_eda, tab_predict, tab_batch, tab_model = st.tabs([
    "📊 Executive KPI Dashboard",
    "🔍 Exploratory Analysis (EDA)",
    "🔮 Individual Risk Predictor",
    "📁 Batch Cohort Assessment",
    "📈 Model Architecture & ROI"
])

PLOTLY_THEME = "plotly_dark"
COLOR_PRIMARY = "#3B82F6"
COLOR_SECONDARY = "#10B981"
COLOR_ALERT = "#EF4444"
COLOR_WARN = "#F59E0B"


# =========================================================
# TAB 1: EXECUTIVE KPI DASHBOARD
# =========================================================
with tab_exec:
    st.subheader("Workforce Retention Overview")
    
    if filtered_df.empty:
        st.warning("No employees match the current filter selection. Please adjust your sidebar filters.")
    else:
        # KPI calculations
        total_employees = len(filtered_df)
        attrited_count = len(filtered_df[filtered_df['Attrition'] == 'Yes'])
        retained_count = total_employees - attrited_count
        attrition_rate = (attrited_count / total_employees) * 100 if total_employees > 0 else 0
        avg_monthly_income = filtered_df['MonthlyIncome'].mean()
        avg_tenure = filtered_df['YearsAtCompany'].mean()
        # Estimated cost of turnover (Industry benchmark: ~50% of annual salary per departure)
        estimated_turnover_cost = attrited_count * (avg_monthly_income * 12 * 0.5)

        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        kpi_col1.metric("Active Headcount", f"{retained_count:,}", f"{total_employees:,} Total")
        kpi_col2.metric("Attrition Rate", f"{attrition_rate:.1f}%", f"{attrited_count} Departures", delta_color="inverse")
        kpi_col3.metric("Avg Monthly Salary", f"${avg_monthly_income:,.0f}")
        kpi_col4.metric("Avg Tenure", f"{avg_tenure:.1f} Yrs")
        kpi_col5.metric("Est. Turnover Cost", f"${estimated_turnover_cost/1e6:.2f}M", "Benchmark: 0.5x Salary", delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 1: Attrition by Role & Attrition Distribution Donut
        row1_col1, row1_col2 = st.columns([3, 2])

        with row1_col1:
            # Attrition rate by Job Role
            role_attr = filtered_df.groupby('JobRole')['Attrition'].apply(
                lambda s: (s == 'Yes').mean() * 100
            ).reset_index(name='AttritionRate').sort_values('AttritionRate', ascending=True)

            fig_role = px.bar(
                role_attr,
                x='AttritionRate',
                y='JobRole',
                orientation='h',
                title="<b>Attrition Rate by Job Role (%)</b>",
                text=role_attr['AttritionRate'].apply(lambda x: f"{x:.1f}%"),
                color='AttritionRate',
                color_continuous_scale=['#10B981', '#F59E0B', '#EF4444'],
                template=PLOTLY_THEME
            )
            fig_role.update_layout(
                xaxis_title="Attrition Rate (%)",
                yaxis_title="",
                coloraxis_showscale=False,
                height=380,
                margin=dict(l=10, r=20, t=40, b=30)
            )
            st.plotly_chart(fig_role, use_container_width=True)

        with row1_col2:
            # Attrition breakdown donut
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Retained', 'Attrited'],
                values=[retained_count, attrited_count],
                hole=0.62,
                marker=dict(colors=['#10B981', '#EF4444']),
                textinfo='label+percent',
                textfont=dict(size=14)
            )])
            fig_donut.update_layout(
                title="<b>Workforce Retention Proportion</b>",
                template=PLOTLY_THEME,
                height=380,
                margin=dict(l=10, r=10, t=40, b=10),
                annotations=[dict(text=f"<b>{attrition_rate:.1f}%</b><br>Attrition", x=0.5, y=0.5, font_size=18, showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Row 2: Department breakdown & Salary Distribution
        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            dept_summary = filtered_df.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
            fig_dept = px.bar(
                dept_summary,
                x='Department',
                y='Count',
                color='Attrition',
                barmode='group',
                title="<b>Headcount Breakdown by Department</b>",
                color_discrete_map={'No': '#3B82F6', 'Yes': '#EF4444'},
                template=PLOTLY_THEME
            )
            fig_dept.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=30))
            st.plotly_chart(fig_dept, use_container_width=True)

        with row2_col2:
            fig_box = px.box(
                filtered_df,
                x='Attrition',
                y='MonthlyIncome',
                color='Attrition',
                notched=True,
                title="<b>Monthly Compensation vs Attrition Disparity</b>",
                color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'},
                template=PLOTLY_THEME
            )
            fig_box.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=30), showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)


# =========================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
with tab_eda:
    st.subheader("🔬 Deep-Dive Root Cause Analysis")
    st.markdown("Examine the key behavioral, organizational, and environmental drivers contributing to talent turnover.")

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        # Overtime Impact
        ot_summary = df_raw.groupby('OverTime')['Attrition'].apply(
            lambda s: (s == 'Yes').mean() * 100
        ).reset_index(name='AttritionRate')

        fig_ot = px.bar(
            ot_summary,
            x='OverTime',
            y='AttritionRate',
            color='OverTime',
            title="<b>The Overtime Hazard: Attrition Rate by Overtime</b>",
            text=ot_summary['AttritionRate'].apply(lambda x: f"{x:.1f}%"),
            color_discrete_map={'Yes': '#EF4444', 'No': '#3B82F6'},
            template=PLOTLY_THEME
        )
        fig_ot.update_layout(yaxis_title="Attrition Rate (%)", height=320, showlegend=False)
        st.plotly_chart(fig_ot, use_container_width=True)
        st.caption("💡 **Insight:** Employees working frequent overtime experience ~30.5% turnover, nearly triple the standard 10.4% baseline.")

    with eda_col2:
        # WorkLifeBalance & Job Satisfaction
        sat_summary = df_raw.groupby(['JobSatisfaction', 'WorkLifeBalance'])['Attrition'].apply(
            lambda s: (s == 'Yes').mean() * 100
        ).reset_index(name='AttritionRate')
        
        sat_pivot = sat_summary.pivot(index='JobSatisfaction', columns='WorkLifeBalance', values='AttritionRate')
        fig_heat = px.imshow(
            sat_pivot,
            labels=dict(x="Work-Life Balance (1=Poor, 4=Best)", y="Job Satisfaction (1=Low, 4=High)", color="Attrition %"),
            x=['1 (Low)', '2 (Med)', '3 (High)', '4 (Best)'],
            y=['1 (Low)', '2 (Med)', '3 (High)', '4 (Best)'],
            title="<b>Heatmap: Satisfaction & Work-Life Balance vs Attrition %</b>",
            color_continuous_scale="Reds",
            text_auto=".1f",
            template=PLOTLY_THEME
        )
        fig_heat.update_layout(height=320)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("💡 **Insight:** Employees with both low satisfaction and poor work-life balance reach attrition rates exceeding 40%.")

    st.markdown("---")
    eda_col3, eda_col4 = st.columns(2)

    with eda_col3:
        # Distance From Home vs Business Travel
        fig_dist = px.histogram(
            df_raw,
            x='DistanceFromHome',
            color='Attrition',
            barmode='overlay',
            nbins=30,
            title="<b>Commute Strain: Distance From Home vs Attrition</b>",
            color_discrete_map={'No': '#3B82F6', 'Yes': '#EF4444'},
            template=PLOTLY_THEME
        )
        fig_dist.update_layout(height=320, xaxis_title="Distance From Home (Miles)")
        st.plotly_chart(fig_dist, use_container_width=True)

    with eda_col4:
        # Years at Company vs Years with Current Manager
        fig_tenure = px.scatter(
            df_raw,
            x='YearsAtCompany',
            y='YearsWithCurrManager',
            color='Attrition',
            size='MonthlyIncome',
            hover_data=['JobRole', 'Department'],
            title="<b>Managerial Relationship & Tenure Dynamics</b>",
            color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'},
            opacity=0.7,
            template=PLOTLY_THEME
        )
        fig_tenure.update_layout(height=320)
        st.plotly_chart(fig_tenure, use_container_width=True)

    st.markdown("---")
    with st.expander("🛠️ Interactive Custom Variable Explorer"):
        st.markdown("Select any variables to explore correlations across the workforce:")
        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_raw.select_dtypes(include=['object']).columns.tolist()

        c1, c2, c3 = st.columns(3)
        var_x = c1.selectbox("X-Axis (Numeric)", options=num_cols, index=num_cols.index('Age') if 'Age' in num_cols else 0)
        var_y = c2.selectbox("Y-Axis (Numeric)", options=num_cols, index=num_cols.index('MonthlyIncome') if 'MonthlyIncome' in num_cols else 1)
        var_color = c3.selectbox("Color By", options=['Attrition'] + cat_cols, index=0)

        fig_custom = px.scatter(
            df_raw,
            x=var_x,
            y=var_y,
            color=var_color,
            color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'} if var_color == 'Attrition' else None,
            title=f"<b>Scatter Analysis: {var_x} vs {var_y} colored by {var_color}</b>",
            template=PLOTLY_THEME
        )
        st.plotly_chart(fig_custom, use_container_width=True)


# =========================================================
# TAB 3: INDIVIDUAL EMPLOYEE RISK PREDICTOR
# =========================================================
with tab_predict:
    st.subheader("🔮 Individual Attrition Risk Predictor")
    st.markdown("Configure an employee profile to compute attrition probability, diagnose primary risk drivers, and view personalized retention recommendations.")

    # Form layout
    with st.form("individual_prediction_form"):
        col_p1, col_p2, col_p3 = st.columns(3)

        with col_p1:
            st.markdown("##### 👤 **Demographics & Experience**")
            pred_age = st.slider("Age", min_value=18, max_value=65, value=34)
            pred_marital = st.selectbox("Marital Status", options=['Single', 'Married', 'Divorced'], index=0)
            pred_travel = st.selectbox("Business Travel", options=['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], index=0)
            pred_distance = st.slider("Distance From Home (Miles)", min_value=1, max_value=35, value=8)
            pred_num_companies = st.number_input("Prior Companies Worked", min_value=0, max_value=12, value=2)

        with col_p2:
            st.markdown("##### 💼 **Job Role & Compensation**")
            pred_dept = st.selectbox("Department", options=predictor.categories.get('Department', ['Research & Development', 'Sales', 'Human Resources']), index=0)
            pred_role = st.selectbox("Job Role", options=predictor.categories.get('JobRole', ['Research Scientist', 'Sales Executive', 'Laboratory Technician']), index=0)
            
            # Show role median income for guidance
            role_median = int(df_raw[df_raw['JobRole'] == pred_role]['MonthlyIncome'].median())
            pred_income = st.number_input(
                f"Monthly Income ($) [Median: ${role_median:,}]",
                min_value=1000,
                max_value=30000,
                value=role_median,
                step=250
            )
            pred_stock = st.selectbox("Stock Option Level", options=[0, 1, 2, 3], index=0)
            pred_hike = st.slider("Percent Salary Hike (%)", min_value=10, max_value=30, value=15)

        with col_p3:
            st.markdown("##### ⏱️ **Workload & Satisfaction**")
            pred_overtime = st.selectbox("Overtime Active?", options=['Yes', 'No'], index=0)
            pred_job_sat = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=2, format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x])
            pred_env_sat = st.select_slider("Environment Satisfaction", options=[1, 2, 3, 4], value=2, format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x])
            pred_work_life = st.select_slider("Work-Life Balance", options=[1, 2, 3, 4], value=2, format_func=lambda x: {1: "1 - Bad", 2: "2 - Average", 3: "3 - Good", 4: "4 - Best"}[x])
            pred_involvement = st.select_slider("Job Involvement", options=[1, 2, 3, 4], value=2)
            
            st.markdown("---")
            pred_total_years = st.slider("Total Working Years", min_value=0, max_value=40, value=max(1, pred_age - 22))
            pred_years_company = st.slider("Years at Current Company", min_value=0, max_value=pred_total_years, value=min(4, pred_total_years))
            pred_years_manager = st.slider("Years with Current Manager", min_value=0, max_value=pred_years_company, value=min(2, pred_years_company))

        submit_btn = st.form_submit_button("⚡ Run Predictive Risk Assessment", use_container_width=True)

    if submit_btn:
        input_dict = {
            'Age': pred_age,
            'MonthlyIncome': pred_income,
            'TotalWorkingYears': pred_total_years,
            'YearsAtCompany': pred_years_company,
            'YearsWithCurrManager': pred_years_manager,
            'DistanceFromHome': pred_distance,
            'StockOptionLevel': pred_stock,
            'EnvironmentSatisfaction': pred_env_sat,
            'JobSatisfaction': pred_job_sat,
            'WorkLifeBalance': pred_work_life,
            'JobInvolvement': pred_involvement,
            'NumCompaniesWorked': pred_num_companies,
            'PercentSalaryHike': pred_hike,
            'Department': pred_dept,
            'JobRole': pred_role,
            'MaritalStatus': pred_marital,
            'OverTime': pred_overtime,
            'BusinessTravel': pred_travel
        }

        input_df = pd.DataFrame([input_dict])
        proba = float(predictor.predict_proba(input_df)[0][1])
        pct_risk = proba * 100

        st.markdown("<br>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            # Half-gauge meter
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct_risk,
                number={'suffix': "%", 'font': {'size': 44, 'color': '#FFFFFF'}},
                title={'text': "<b>Predicted Attrition Probability</b><br><span style='font-size:0.8em;color:#94A3B8'>Model Confidence</span>", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': "#FFFFFF", 'thickness': 0.25},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 35], 'color': '#10B981'},
                        {'range': [35, 65], 'color': '#F59E0B'},
                        {'range': [65, 100], 'color': '#EF4444'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': pct_risk
                    }
                }
            ))
            fig_gauge.update_layout(height=320, template=PLOTLY_THEME, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("### Risk Evaluation & Diagnosis")
            if pct_risk >= 65:
                risk_badge = "🔴 <span class='risk-high'>HIGH ATTRITION HAZARD</span>"
                status_desc = "The predictive engine flags severe flight risk indicators. Proactive retention intervention is critically recommended within the next 30 days."
            elif pct_risk >= 35:
                risk_badge = "🟡 <span class='risk-med'>MODERATE ATTRITION RISK</span>"
                status_desc = "The employee exhibits notable warning signals. Regular 1-on-1 check-ins and engagement alignment are advised."
            else:
                risk_badge = "🟢 <span class='risk-low'>LOW FLIGHT RISK</span>"
                status_desc = "The employee demonstrates strong organizational stability and high retention probability."

            st.markdown(f"#### Status: {risk_badge}", unsafe_allow_html=True)
            st.write(status_desc)

            # Detect key drivers
            st.markdown("##### 📌 Primary Risk Factors Identified:")
            drivers = []
            if pred_overtime == 'Yes':
                drivers.append("⚠️ **Active OverTime**: Consistently working beyond normal hours increases attrition probability.")
            if pred_income < role_median * 0.9:
                gap = int(role_median - pred_income)
                drivers.append(f"⚠️ **Below-Benchmark Compensation**: Monthly income is ${gap:,} below the peer median (${role_median:,}).")
            if pred_work_life <= 2:
                drivers.append("⚠️ **Strained Work-Life Balance**: Reported low work-life equilibrium accelerates burnout.")
            if pred_job_sat <= 2:
                drivers.append("⚠️ **Low Job Satisfaction**: Dissatisfaction with daily responsibilities.")
            if pred_stock == 0:
                drivers.append("⚠️ **No Equity Alignment**: Level 0 Stock Option provides zero financial vesting lock-in.")
            if pred_travel == 'Travel_Frequently':
                drivers.append("⚠️ **Frequent Business Travel**: High travel cadence increases friction.")
            if pred_years_company <= 2 and pred_total_years > 5:
                drivers.append("⚠️ **Early Tenure Transition Window**: First 24 months at a company are statistically highest-turnover.")

            if not drivers:
                st.success("✅ No critical risk warning flags triggered for this profile.")
            else:
                for d in drivers[:4]:
                    st.markdown(d)

        # Actionable Retention Playbook
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Executive Retention Action Plan")
        action_col1, action_col2, action_col3 = st.columns(3)

        with action_col1:
            st.markdown("""
            **1. Workload & Wellbeing**
            - Evaluate current project deadlines and reallocate overtime duties.
            - Offer hybrid or flexible scheduling options to relieve commute strain.
            """)
        with action_col2:
            st.markdown("""
            **2. Compensation & Recognition**
            - Conduct an off-cycle market equity review if income is below median.
            - Explore retention bonuses or long-term stock option grants.
            """)
        with action_col3:
            st.markdown("""
            **3. Leadership & Career Path**
            - Schedule a dedicated career growth roadmap discussion with leadership.
            - Identify mentorship or lateral mobility opportunities within other teams.
            """)


# =========================================================
# TAB 4: BATCH COHORT ASSESSMENT
# =========================================================
with tab_batch:
    st.subheader("📁 Enterprise Batch Assessment & Talent Risk Matrix")
    st.markdown("Upload organizational employee CSV datasets to score retention risks en masse and export prioritized retention matrices.")

    batch_source = st.radio(
        "Choose Batch Data Source:",
        options=["Load Pre-configured Test Cohort (50 Employees)", "Upload Custom Employee CSV"],
        horizontal=True
    )

    batch_df = None
    if batch_source == "Load Pre-configured Test Cohort (50 Employees)":
        sample_path = "sample_test_cohort.csv"
        if os.path.exists(sample_path):
            batch_df = pd.read_csv(sample_path)
            st.success(f"Loaded demo cohort of {len(batch_df)} employee profiles.")
        else:
            st.info("Generating sample test cohort...")
            from generate_dummy_data import generate_cohort
            batch_df = generate_cohort(50)
            batch_df.to_csv(sample_path, index=False)
            st.success(f"Generated and loaded {len(batch_df)} profiles.")
    else:
        uploaded_file = st.file_uploader("Upload CSV file with employee data", type=['csv'])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Successfully uploaded dataset with {len(batch_df)} records.")
            except Exception as ex:
                st.error(f"Error parsing uploaded CSV: {ex}")

    if batch_df is not None:
        # Compute probabilities
        with st.spinner("Scoring employee cohort through AI model..."):
            probs = predictor.predict_proba(batch_df)[:, 1]
            scored_df = batch_df.copy()
            scored_df['Attrition_Risk_%'] = (probs * 100).round(1)
            scored_df['Risk_Category'] = pd.cut(
                scored_df['Attrition_Risk_%'],
                bins=[-1, 35, 65, 100],
                labels=['Low Risk', 'Moderate Risk', 'High Risk']
            )
            scored_df['Retention_Priority'] = scored_df['Risk_Category'].map({
                'High Risk': 'P1 - Immediate Action',
                'Moderate Risk': 'P2 - Watchlist',
                'Low Risk': 'P3 - Healthy'
            })

        # Batch Summary Cards
        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        n_high = (scored_df['Risk_Category'] == 'High Risk').sum()
        n_med = (scored_df['Risk_Category'] == 'Moderate Risk').sum()
        n_low = (scored_df['Risk_Category'] == 'Low Risk').sum()

        b_c1.metric("Batch Evaluated", f"{len(scored_df):,} Employees")
        b_c2.metric("Critical High Risk (P1)", f"{n_high}", f"{(n_high/len(scored_df))*100:.1f}%", delta_color="inverse")
        b_c3.metric("Moderate Watchlist (P2)", f"{n_med}", f"{(n_med/len(scored_df))*100:.1f}%")
        b_c4.metric("Stable Low Risk (P3)", f"{n_low}", f"{(n_low/len(scored_df))*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # Talent Matrix Scatter
        fig_scatter = px.scatter(
            scored_df,
            x='MonthlyIncome',
            y='Attrition_Risk_%',
            color='Risk_Category',
            size='YearsAtCompany' if 'YearsAtCompany' in scored_df.columns else None,
            hover_data=['JobRole', 'Department'] if 'JobRole' in scored_df.columns else None,
            color_discrete_map={'Low Risk': '#10B981', 'Moderate Risk': '#F59E0B', 'High Risk': '#EF4444'},
            title="<b>Talent Retention Matrix: Monthly Income vs Attrition Risk</b>",
            template=PLOTLY_THEME
        )
        fig_scatter.add_hline(y=65, line_dash="dash", line_color="#EF4444", annotation_text="High Risk Threshold (65%)")
        fig_scatter.add_hline(y=35, line_dash="dash", line_color="#F59E0B", annotation_text="Moderate Risk Threshold (35%)")
        fig_scatter.update_layout(height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Scored table
        st.markdown("##### 🔍 Prioritized Employee Risk Roster")
        sorted_batch = scored_df.sort_values('Attrition_Risk_%', ascending=False)
        st.dataframe(sorted_batch, use_container_width=True)

        # CSV Download Button
        csv_buffer = io.StringIO()
        sorted_batch.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export Enriched Attrition Risk Report (CSV)",
            data=csv_buffer.getvalue(),
            file_name="TalentPulse_Attrition_Risk_Report.csv",
            mime="text/csv"
        )


# =========================================================
# TAB 5: MODEL ARCHITECTURE & ROI
# =========================================================
with tab_model:
    st.subheader("📈 Machine Learning Architecture & Governance")
    st.markdown("Transparency into pipeline engineering, validation metrics, decision drivers, and financial return on investment.")

    m_col1, m_col2 = st.columns([1, 1])

    with m_col1:
        st.markdown("##### ⚙️ Production Pipeline Pipeline Architecture")
        st.markdown("""
        - **Preprocessing Engine:**
          - `StandardScaler`: Normalizes numerical ranges (Age, MonthlyIncome, Tenure).
          - `OneHotEncoder`: Encodes categorical signals (Department, JobRole, OverTime, etc.).
        - **Predictive Ensemble:**
          - **Soft-Voting Ensemble** combining three complementary classifiers:
            1. *Balanced Logistic Regression*: Linear baseline with calibrated probabilities.
            2. *Random Forest Classifier (250 trees, max depth 8)*: Non-linear interaction capture.
            3. *Gradient Boosting Classifier (100 estimators)*: Sequential residual refinement.
        - **Classification Calibration:**
          - Probability threshold calibrated to **0.40** to balance high recall with actionable precision for retention campaigns.
        """)

        # Display Metrics
        metrics = predictor.metrics
        st.markdown("##### 🏆 Validation Performance Metrics")
        val1, val2, val3, val4 = st.columns(4)
        val1.metric("ROC-AUC Score", f"{metrics.get('roc_auc', 0.79):.3f}")
        val2.metric("Accuracy", f"{metrics.get('accuracy', 0.81)*100:.1f}%")
        val3.metric("Recall (Sensitivity)", f"{metrics.get('recall', 0.53)*100:.1f}%")
        val4.metric("Precision", f"{metrics.get('precision', 0.42)*100:.1f}%")

    with m_col2:
        # Confusion Matrix Heatmap
        cm = metrics.get('confusion_matrix', [[212, 35], [22, 25]])
        cm_df = pd.DataFrame(
            cm,
            index=['Actual Stay (0)', 'Actual Leave (1)'],
            columns=['Pred Stay (0)', 'Pred Leave (1)']
        )
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            title="<b>Holdout Test Confusion Matrix</b>",
            color_continuous_scale="Blues",
            template=PLOTLY_THEME
        )
        fig_cm.update_layout(height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    
    # Feature Importances
    st.markdown("##### 🧠 Global Feature Importance Ranking")
    st.caption("Relative weight assigned by the machine learning ensemble across primary features:")
    
    imp_dict = predictor.feature_importances
    imp_df = pd.DataFrame(list(imp_dict.items()), columns=['Feature', 'Importance (%)']).head(10)
    imp_df = imp_df.sort_values('Importance (%)', ascending=True)

    fig_imp = px.bar(
        imp_df,
        x='Importance (%)',
        y='Feature',
        orientation='h',
        title="<b>Top 10 Retention Predictors (%)</b>",
        text=imp_df['Importance (%)'].apply(lambda x: f"{x:.1f}%"),
        color='Importance (%)',
        color_continuous_scale="Blues",
        template=PLOTLY_THEME
    )
    fig_imp.update_layout(height=360, margin=dict(l=10, r=20, t=40, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 💰 Business Value & ROI Estimation")
    st.markdown("""
    Employee replacement cost typically ranges from **50% to 150%** of an employee's annual salary, accounting for recruiting fees, onboarding lag, lost institutional knowledge, and team disruption.
    
    * **Scenario:** In an organization of 1,000 employees with a 16% baseline turnover (160 departures/year):
      * Standard Turnover Cost (at $65k avg salary): **$5,200,000 / year**
      * With **TalentPulse AI**, predicting and intervening on early flight risks with a conservative **25% intervention success rate**:
      * **Estimated Annual Organizational Savings:** **$1,300,000 / year**
    """)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.9rem; padding: 16px 0; line-height: 1.6;">
    <b style="color: #F8FAFC; font-size: 1rem;">TalentPulse AI</b> — Employee Attrition & Retention Intelligence System<br>
    🎓 <b>College Project Submission & Portfolio Showcase</b> &nbsp;|&nbsp; Designed & Developed by <b style="color: #60A5FA;">Pranav</b><br>
    <span style="font-size: 0.8rem; color: #64748B;">Built with Python, Streamlit, Scikit-Learn, and Plotly &bull; Open Source on GitHub</span>
</div>
""", unsafe_allow_html=True)
