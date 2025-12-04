"""
LandGCH Interactive Dashboard
==============================
A Streamlit application for exploring global land-use projections (2020-2050)
using HILDA+ data and Time-varying Markov Chain models.

Author: Angelos
Date: December 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="LandGCH Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Land use classes
LAND_CLASSES = ['Crops', 'TreeCrops', 'Forest', 'Grassland', 'Urban', 'Water', 'Other']

# Color scheme (colorblind-friendly)
COLORS = {
    'Crops': '#E69F00',      # Orange
    'TreeCrops': '#009E73',  # Green
    'Forest': '#0072B2',     # Blue
    'Grassland': '#D55E00',  # Red-orange
    'Urban': '#CC79A7',      # Pink
    'Water': '#56B4E9',      # Sky blue
    'Other': '#999999'       # Gray
}

# Scenario definitions
SCENARIOS = {
    'BAU': {
        'name': 'Business As Usual',
        'description': 'Baseline projection using historical trends',
        'color': '#666666'
    },
    'Fat': {
        'name': 'High Meat Diet (Fat)',
        'description': 'Increased livestock grazing, higher agricultural land demand',
        'multipliers': {
            'Crops': 1.06, 'TreeCrops': 1.02, 'Forest': 0.97,
            'Grassland': 1.12, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95
        },
        'color': '#d62728'
    },
    'EAT': {
        'name': 'EAT-Lancet Diet',
        'description': 'Plant-based diet, reduced agricultural footprint, forest restoration',
        'multipliers': {
            'Crops': 0.90, 'TreeCrops': 0.97, 'Forest': 1.08,
            'Grassland': 0.82, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.98
        },
        'color': '#2ca02c'
    },
    'NDC': {
        'name': 'Afforestation & Biofuels',
        'description': 'Forest expansion, biofuel crops, climate mitigation focus',
        'multipliers': {
            'Crops': 1.03, 'TreeCrops': 1.05, 'Forest': 1.07,
            'Grassland': 0.98, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95
        },
        'color': '#17becf'
    }
}

# ============================================================================
# DATA PATHS (USER MUST UPDATE THESE)
# ============================================================================

# NOTE: Users need to update these paths to point to their actual data files
DATA_PATHS = {
    'bau': r"yourpath\7.diets\bau\1.ALL_COUNTRIES_annual_projections_2020_2050.csv",
    'fat': r"yourpath\7.diets\fat\ALL_COUNTRIES_fat_annual_projections_2020_2050.csv",
    'eat': r"yourpath\7.diets\eat\ALL_COUNTRIES_eat_annual_projections_2020_2050.csv",
    'ndc': r"yourpath\7.diets\ndc\ALL_COUNTRIES_ndc_annual_projections_2020_2050.csv",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_data(scenario='bau'):
    """Load data for a specific scenario"""
    try:
        path = DATA_PATHS[scenario.lower()]
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"⚠️ Data file not found: {path}")
        st.info("Please update the DATA_PATHS dictionary in the code with your actual file locations.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_country_data(df, country_code):
    """Extract data for a specific country"""
    return df[df['Country'] == country_code].copy()

def create_trajectory_plot(country_df, land_classes=None, show_uncertainty=False):
    """Create interactive trajectory plot for a country"""
    if land_classes is None:
        land_classes = LAND_CLASSES
    
    fig = go.Figure()
    
    for lu in land_classes:
        if lu in country_df.columns:
            fig.add_trace(go.Scatter(
                x=country_df['Year'],
                y=country_df[lu],
                mode='lines',
                name=lu,
                line=dict(color=COLORS[lu], width=3),
                hovertemplate=f'{lu}<br>Year: %{{x}}<br>Area: %{{y:,.0f}} km²<extra></extra>'
            ))
    
    fig.update_layout(
        title='Land Use Projections (2020-2050)',
        xaxis_title='Year',
        yaxis_title='Area (km²)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(x=1.02, y=1, xanchor='left'),
        font=dict(size=12)
    )
    
    return fig

def create_stacked_area(country_df):
    """Create stacked area chart"""
    fig = go.Figure()
    
    for lu in LAND_CLASSES:
        if lu in country_df.columns:
            fig.add_trace(go.Scatter(
                x=country_df['Year'],
                y=country_df[lu],
                mode='lines',
                name=lu,
                line=dict(width=0.5, color=COLORS[lu]),
                stackgroup='one',
                fillcolor=COLORS[lu],
                hovertemplate=f'{lu}<br>%{{y:,.0f}} km²<extra></extra>'
            ))
    
    fig.update_layout(
        title='Land Use Composition Over Time',
        xaxis_title='Year',
        yaxis_title='Area (km²)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(x=1.02, y=1)
    )
    
    return fig

def create_scenario_comparison(countries_data, land_use='Crops'):
    """Compare scenarios across countries"""
    fig = go.Figure()
    
    for scenario, data in countries_data.items():
        if data is not None and not data.empty:
            # Normalize to 2020 = 100
            baseline = data[data['Year'] == 2020][land_use].values
            if len(baseline) > 0 and baseline[0] != 0:
                normalized = (data[land_use] / baseline[0]) * 100
                
                fig.add_trace(go.Scatter(
                    x=data['Year'],
                    y=normalized,
                    mode='lines',
                    name=SCENARIOS[scenario]['name'],
                    line=dict(width=3, color=SCENARIOS[scenario]['color']),
                    hovertemplate=f"{scenario}<br>Index: %{{y:.1f}}<extra></extra>"
                ))
    
    fig.update_layout(
        title=f'{land_use} - Scenario Comparison (Index: 2020 = 100)',
        xaxis_title='Year',
        yaxis_title='Index (2020 = 100)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(x=1.02, y=1)
    )
    
    return fig

def create_2050_comparison_bar(country_dfs, country_code):
    """Create bar chart comparing 2050 values across scenarios"""
    data_2050 = []
    
    for scenario, df in country_dfs.items():
        if df is not None and not df.empty:
            row_2050 = df[df['Year'] == 2050]
            if not row_2050.empty:
                for lu in LAND_CLASSES:
                    if lu in row_2050.columns:
                        data_2050.append({
                            'Scenario': SCENARIOS[scenario]['name'],
                            'Land Use': lu,
                            'Area': row_2050[lu].values[0]
                        })
    
    if not data_2050:
        return None
    
    df_2050 = pd.DataFrame(data_2050)
    
    fig = px.bar(
        df_2050,
        x='Land Use',
        y='Area',
        color='Scenario',
        barmode='group',
        title=f'{country_code} - Land Use in 2050 Across Scenarios',
        color_discrete_map={
            SCENARIOS['BAU']['name']: SCENARIOS['BAU']['color'],
            SCENARIOS['Fat']['name']: SCENARIOS['Fat']['color'],
            SCENARIOS['EAT']['name']: SCENARIOS['EAT']['color'],
            SCENARIOS['NDC']['name']: SCENARIOS['NDC']['color'],
        }
    )
    
    fig.update_layout(
        xaxis_title='Land Use Type',
        yaxis_title='Area (km²)',
        template='plotly_white',
        height=500
    )
    
    return fig

# ============================================================================
# MAIN APP STRUCTURE
# ============================================================================

def main():
    """Main application logic"""
    
    # Sidebar navigation
    st.sidebar.title("🌍 LandGCH Dashboard")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🔍 Country Explorer", "📊 Scenario Comparison"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About LandGCH**
    
    Global land-use forecasting model combining:
    - HILDA+ v2.0 (1960-2020)
    - Time-varying Markov Chains
    - Multiple dietary scenarios
    - Country-specific dynamics
    """)
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Country Explorer":
        show_country_explorer()
    elif page == "📊 Scenario Comparison":
        show_scenario_comparison()

# ============================================================================
# PAGE 1: HOME
# ============================================================================

def show_home_page():
    """Home page with model overview"""
    
    st.title("🌍 LandGCH: Global Land-Use Change Model")
    st.markdown("### Interactive Dashboard for Land-Use Projections (2020-2050)")
    
    st.markdown("---")
    
    # Model description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 📖 About the Model
        
        **LandGCH** is a comprehensive global land-use forecasting framework that combines:
        
        - **Historical Data**: HILDA+ v2.0 (1960-2020) at 1km² resolution
        - **Forecasting Method**: Time-varying Markov Chain models
        - **Validation**: Multi-period validation (10/20/30 years)
        - **Scenarios**: Multiple dietary and policy interventions
        - **Resolution**: Country-level projections with 7 land-use classes
        
        ### 🎯 Key Features
        
        1. **Baseline (BAU) Projections**: Business-as-usual forecasts based on historical trends
        2. **Scenario Analysis**: Three intervention scenarios (Fat, EAT-Lancet, NDC/Afforestation)
        3. **Country-Specific Dynamics**: Tailored constraints for different country types
        4. **Uncertainty Quantification**: Probabilistic projections with confidence bounds
        
        ### 📊 Land-Use Classes
        
        - **Crops**: Annual cropland
        - **TreeCrops**: Permanent crops (orchards, plantations)
        - **Forest**: All forest types
        - **Grassland**: Pasture and rangeland
        - **Urban**: Built-up areas
        - **Water**: Inland water bodies
        - **Other**: Sparse vegetation, bare land
        """)
    
    with col2:
        st.markdown("""
        ## 🎨 Scenarios
        
        ### 🥩 Fat Scenario
        High meat consumption, increased grazing land
        
        ### 🥗 EAT-Lancet
        Plant-based diet, reduced agricultural footprint
        
        ### 🌲 NDC/Afforestation
        Forest expansion, biofuels, climate mitigation
        
        ---
        
        ## 📅 Timeline
        
        - **Historical**: 1960-2020
        - **Baseline**: 2020
        - **Projections**: 2021-2050
        - **Scenarios applied**: 2026-2050
        
        ---
        
        ## 🔬 Methodology
        
        1. HILDA+ data processing
        2. Country boundary overlay
        3. Transition matrix calculation
        4. Markov Chain validation
        5. Scenario multipliers
        6. Future projections
        """)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("## 📈 Quick Statistics")
    
    try:
        df_bau = load_data('bau')
        if df_bau is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                n_countries = df_bau['Country'].nunique()
                st.metric("Countries Covered", f"{n_countries}")
            
            with col2:
                years = df_bau['Year'].nunique()
                st.metric("Projection Years", f"{years}")
            
            with col3:
                # Calculate global land area (2020)
                df_2020 = df_bau[df_bau['Year'] == 2020]
                total_land = df_2020[LAND_CLASSES].sum().sum() / 1e6  # Convert to million km²
                st.metric("Global Land Area (2020)", f"{total_land:.1f}M km²")
            
            with col4:
                st.metric("Land-Use Classes", len(LAND_CLASSES))
    except:
        pass
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate to **Country Explorer** or **Scenario Comparison**")

# ============================================================================
# PAGE 2: COUNTRY EXPLORER
# ============================================================================

def show_country_explorer():
    """Country-level analysis page"""
    
    st.title("🔍 Country Explorer")
    st.markdown("Analyze land-use projections for individual countries")
    
    # Load BAU data to get country list
    df_bau = load_data('bau')
    
    if df_bau is None:
        st.error("Unable to load data. Please check your file paths.")
        return
    
    # Country selection
    countries = sorted(df_bau['Country'].unique())
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_country = st.selectbox(
            "Select Country (ISO3 Code)",
            countries,
            index=countries.index('USA') if 'USA' in countries else 0
        )
        
        st.markdown("---")
        
        # Visualization options
        st.markdown("### 🎨 Display Options")
        
        show_stacked = st.checkbox("Show stacked area chart", value=False)
        
        selected_classes = st.multiselect(
            "Land-use classes to display",
            LAND_CLASSES,
            default=LAND_CLASSES
        )
        
        st.markdown("---")
        
        # Scenario selection for comparison
        st.markdown("### 📊 Compare Scenarios")
        compare_scenarios = st.multiselect(
            "Select scenarios to compare",
            ['BAU', 'Fat', 'EAT', 'NDC'],
            default=['BAU']
        )
    
    with col2:
        # Get country data
        country_df_bau = get_country_data(df_bau, selected_country)
        
        if country_df_bau.empty:
            st.warning(f"No data found for country: {selected_country}")
            return
        
        # Display country info
        st.markdown(f"## {selected_country}")
        
        # Show baseline (BAU) projection
        if 'BAU' in compare_scenarios:
            st.markdown("### Business As Usual (BAU) Projection")
            
            if show_stacked:
                fig = create_stacked_area(country_df_bau)
            else:
                fig = create_trajectory_plot(country_df_bau, selected_classes)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Multi-scenario comparison
        if len(compare_scenarios) > 1:
            st.markdown("### Scenario Comparison")
            
            # Load data for selected scenarios
            scenario_data = {}
            for scenario in compare_scenarios:
                df = load_data(scenario.lower())
                if df is not None:
                    scenario_data[scenario] = get_country_data(df, selected_country)
            
            # Land use selector for comparison
            compare_lu = st.selectbox(
                "Select land-use type to compare",
                selected_classes if selected_classes else LAND_CLASSES
            )
            
            fig = create_scenario_comparison(scenario_data, compare_lu)
            st.plotly_chart(fig, use_container_width=True)
        
        # 2050 comparison
        if len(compare_scenarios) > 1:
            st.markdown("### 2050 Snapshot")
            
            country_dfs = {}
            for scenario in compare_scenarios:
                df = load_data(scenario.lower())
                if df is not None:
                    country_dfs[scenario] = get_country_data(df, selected_country)
            
            fig = create_2050_comparison_bar(country_dfs, selected_country)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Raw Data"):
            st.dataframe(country_df_bau[['Year'] + LAND_CLASSES], use_container_width=True)

# ============================================================================
# PAGE 3: SCENARIO COMPARISON
# ============================================================================

def show_scenario_comparison():
    """Multi-country scenario comparison"""
    
    st.title("📊 Scenario Comparison")
    st.markdown("Compare land-use trajectories across countries and scenarios")
    
    # Load BAU data to get country list
    df_bau = load_data('bau')
    
    if df_bau is None:
        st.error("Unable to load data. Please check your file paths.")
        return
    
    countries = sorted(df_bau['Country'].unique())
    
    # Selection panel
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 🌍 Select Countries")
        
        # Quick selection buttons
        if st.button("Top 10 by area"):
            # Calculate total area per country in 2020
            df_2020 = df_bau[df_bau['Year'] == 2020]
            country_areas = df_2020.groupby('Country')[LAND_CLASSES].sum().sum(axis=1).nlargest(10)
            selected_countries = list(country_areas.index)
        else:
            selected_countries = ['USA', 'CHN', 'BRA', 'RUS', 'IND']
        
        selected_countries = st.multiselect(
            "Countries to compare",
            countries,
            default=selected_countries[:5]
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Scenarios")
        scenarios_to_compare = st.multiselect(
            "Select scenarios",
            ['BAU', 'Fat', 'EAT', 'NDC'],
            default=['BAU', 'EAT']
        )
        
        st.markdown("---")
        
        st.markdown("### 🎨 Options")
        land_use_compare = st.selectbox(
            "Land-use type",
            LAND_CLASSES
        )
        
        normalize = st.checkbox("Normalize to 2020 = 100", value=True)
    
    with col2:
        if not selected_countries:
            st.info("👈 Select countries from the sidebar to begin comparison")
            return
        
        if not scenarios_to_compare:
            st.warning("Please select at least one scenario")
            return
        
        st.markdown(f"## {land_use_compare} Comparison")
        
        # Create comparison plot
        fig = go.Figure()
        
        for country in selected_countries:
            for scenario in scenarios_to_compare:
                df = load_data(scenario.lower())
                if df is not None:
                    country_df = get_country_data(df, country)
                    
                    if not country_df.empty and land_use_compare in country_df.columns:
                        y_data = country_df[land_use_compare]
                        
                        if normalize:
                            baseline = country_df[country_df['Year'] == 2020][land_use_compare].values
                            if len(baseline) > 0 and baseline[0] != 0:
                                y_data = (y_data / baseline[0]) * 100
                        
                        line_style = 'solid' if scenario == 'BAU' else 'dash'
                        
                        fig.add_trace(go.Scatter(
                            x=country_df['Year'],
                            y=y_data,
                            mode='lines',
                            name=f"{country} - {scenario}",
                            line=dict(width=2, dash=line_style),
                            hovertemplate=f"{country} ({scenario})<br>%{{y:.1f}}<extra></extra>"
                        ))
        
        ylabel = 'Index (2020 = 100)' if normalize else 'Area (km²)'
        
        fig.update_layout(
            title=f'{land_use_compare} - Multi-Country & Scenario Comparison',
            xaxis_title='Year',
            yaxis_title=ylabel,
            hovermode='x unified',
            template='plotly_white',
            height=700,
            legend=dict(x=1.02, y=1, xanchor='left')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics (2050)")
        
        summary_data = []
        for country in selected_countries:
            for scenario in scenarios_to_compare:
                df = load_data(scenario.lower())
                if df is not None:
                    country_df = get_country_data(df, country)
                    row_2050 = country_df[country_df['Year'] == 2050]
                    row_2020 = country_df[country_df['Year'] == 2020]
                    
                    if not row_2050.empty and not row_2020.empty:
                        val_2050 = row_2050[land_use_compare].values[0]
                        val_2020 = row_2020[land_use_compare].values[0]
                        change = ((val_2050 - val_2020) / val_2020 * 100) if val_2020 != 0 else 0
                        
                        summary_data.append({
                            'Country': country,
                            'Scenario': scenario,
                            '2020 (km²)': f"{val_2020:,.0f}",
                            '2050 (km²)': f"{val_2050:,.0f}",
                            'Change (%)': f"{change:+.1f}%"
                        })
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
