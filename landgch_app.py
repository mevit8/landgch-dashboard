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

# Custom CSS - Modern styling with logo support
st.markdown("""
<style>
    /* Hide auto-generated navigation */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    
    /* Global styling */
    .stApp { background-color: #F8FAFC; }
    .block-container { padding-top: 1.5rem !important; max-width: 1200px; }

    /* Headers */
    h1 { font-size: 1.7rem !important; font-weight: 800; color: #0F172A; margin-bottom: 0.5rem !important; }
    h2 { font-size: 1.3rem !important; font-weight: 700; color: #1E293B; margin-top: 1rem !important; border-bottom: none !important; }
    h3 { font-size: 1.0rem !important; font-weight: 600; color: #475569; }

    /* Sidebar */
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 2px solid #e0e0e0;
    }
    
    [data-testid="stSidebar"] img {
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 1rem;
    }
    
    /* Compact inputs */
    .stNumberInput input { font-size: 0.85rem !important; padding: 0.4rem !important; }
    .element-container { margin-bottom: 0.5rem !important; }
    
    /* Navigation styling */
    [data-testid="stSidebar"] .stRadio > div {
        background: linear-gradient(180deg, #f0f2f6 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #e0e0e0;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        font-weight: 600;
        font-size: 1rem;
        padding: 0.5rem;
    }
    
    /* Page links styling */
    [data-testid="stSidebar"] a[data-testid="stPageLink"] {
        display: block;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.5rem;
        background-color: rgba(31, 119, 180, 0.1);
        border: 2px solid rgba(31, 119, 180, 0.2);
        color: #1f77b4;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    [data-testid="stSidebar"] a[data-testid="stPageLink"]:hover {
        background-color: rgba(31, 119, 180, 0.2);
        border-color: rgba(31, 119, 180, 0.4);
    }
    
    [data-testid="stSidebar"] a[data-testid="stPageLink"][aria-current="page"] {
        background-color: #1f77b4;
        color: white;
        border-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

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

# Scenario definitions - Full set with codes
SCENARIOS = {
    'BAU': {
        'name': 'Business As Usual',
        'code': 'BAU',
        'description': 'The baseline model land use annual projections to 2050.',
        'color': '#666666'
    },
    'Fat': {
        'name': 'High-meat diet',
        'code': 'Fat',
        'description': 'Rich in meat, this diet scenario assigns more area to pasture and feed crops.',
        'multipliers': {
            'Crops': 1.06, 'TreeCrops': 1.02, 'Forest': 0.97,
            'Grassland': 1.12, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95
        },
        'color': '#d62728'
    },
    'EAT': {
        'name': 'The Lancet EAT diet',
        'code': 'EAT',
        'description': 'A "healthy diet" scenario, increased area for vegetables/legumes and reduced pasture.',
        'multipliers': {
            'Crops': 0.90, 'TreeCrops': 0.97, 'Forest': 1.08,
            'Grassland': 0.82, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.98
        },
        'color': '#2ca02c'
    },
    'NDC': {
        'name': 'National Determined Contributions',
        'code': 'NDC',
        'description': 'Countries implement explicit forest gain targets, crops may increase due to bioenergy demand.',
        'multipliers': {
            'Crops': 1.03, 'TreeCrops': 1.05, 'Forest': 1.07,
            'Grassland': 0.98, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95
        },
        'color': '#17becf'
    },
    'Afforest': {
        'name': 'Afforestation/Reforestation',
        'code': 'Afforest',
        'description': 'Reduced deforestation/protection policies that restrict transitions out of forest.',
        'multipliers': {
            'Crops': 1.01, 'TreeCrops': 1.02, 'Forest': 1.07,
            'Grassland': 0.99, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95
        },
        'color': '#9467bd'
    },
    'Bioen': {
        'name': 'Bioenergy/energy crop expansion',
        'code': 'Bioen',
        'description': 'Swift in green fuels production, biofuel mandates increase area of crops for bioenergy.',
        'multipliers': {
            'Crops': 1.03, 'TreeCrops': 1.05, 'Forest': 1.03,
            'Grassland': 0.97, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95
        },
        'color': '#8c564b'
    },
    'Yieldint': {
        'name': 'Yield improvement/intensification',
        'code': 'Yieldint',
        'description': 'Higher yields reduce land need for the same output, lowering demand-driven conversion pressure.',
        'multipliers': {
            'Crops': 0.90, 'TreeCrops': 0.97, 'Forest': 1.03,
            'Grassland': 0.83, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.98
        },
        'color': '#e377c2'
    },
    'Landretir': {
        'name': 'Land retirement/Carbon pricing',
        'code': 'Landretir',
        'description': 'Potential incentives withdrawing marginal cropland to forest, or forcing retirement quotas.',
        'multipliers': {
            'Crops': 0.91, 'TreeCrops': 0.96, 'Forest': 1.07,
            'Grassland': 0.82, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.98
        },
        'color': '#7f7f7f'
    }
}

# ============================================================================
# DATA PATHS (USER MUST UPDATE THESE)
# ============================================================================

# NOTE: Users need to update these paths to point to their actual data files
# Using forward slashes for cross-platform compatibility (Windows/Linux)
DATA_PATHS = {
    'bau': "data/7.diets/bau/1.ALL_COUNTRIES_annual_projections_2020_2050.csv",
    'fat': "data/7.diets/fat/ALL_COUNTRIES_fat_annual_projections_2020_2050.csv",
    'eat': "data/7.diets/eat/ALL_COUNTRIES_eat_annual_projections_2020_2050.csv",
    'ndc': "data/7.diets/ndc/ALL_COUNTRIES_ndc_annual_projections_2020_2050.csv",
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
    
    # Sidebar with logo and info
    with st.sidebar:
        # Logo first
        st.image("https://unsdsn.globalclimatehub.org/wp-content/uploads/2022/09/logo.png", width=200)
        
        st.title("🌍 LandGCH Dashboard")
        st.markdown("---")
        
        # Scenarios expander
        with st.expander("📋 Scenarios", expanded=False):
            st.markdown("""
            Dietary choices and land use policies strongly affect the area of land needed (or allowed). 
            We have developed the following set of indicative scenarios:
            """)
            
            scenario_table = """
| Code | Name | Description |
|------|------|-------------|
| BAU | Business As Usual | Baseline model projections to 2050 |
| Fat | High-meat diet | More area to pasture and feed crops |
| EAT | The Lancet EAT diet | Increased vegetables/legumes, reduced pasture |
| NDC | National Determined Contributions | Forest gain targets, bioenergy crops |
| Afforest | Afforestation/Reforestation | Forest protection policies |
| Bioen | Bioenergy expansion | Biofuel mandates increase crop area |
| Yieldint | Yield intensification | Higher yields reduce land need |
| Landretir | Land retirement | Marginal cropland to forest |
"""
            st.markdown(scenario_table)
        
        st.markdown("---")
        
        st.markdown("""
        **About LandGCH**
        
        Global land-use forecasting model combining:
        - HILDA+ v2.0 (1960-2020)
        - Time-varying Markov Chains
        - Multiple dietary scenarios
        - Country-specific dynamics
        """)
    
    # Main content area with tabs
    tab_intro, tab_global, tab_country, tab_scenario, tab_custom = st.tabs([
        "📖 Introduction",
        "🌍 Global Results", 
        "🔍 Country Explorer",
        "📊 Scenario Comparison",
        "🔬 Custom Model"
    ])
    
    with tab_intro:
        show_introduction_page()
    
    with tab_global:
        show_global_results()
    
    with tab_country:
        show_country_explorer()
    
    with tab_scenario:
        show_scenario_comparison()
    
    with tab_custom:
        show_custom_model_tab()


def show_custom_model_tab():
    """Custom Model tab - links to full Custom Model page"""
    st.title("🔬 Custom Model Builder")
    st.markdown("""
    The Custom Model Builder allows you to create your own land-use projections using:
    
    - **Stage 1: Base Projection** - Run Markov Chain projection with your own transition matrix
    - **Stage 2: Scenario Adjustment** - Apply multipliers to adjust 2050 results
    
    This is a two-stage process that gives you full control over the projection parameters.
    """)
    
    st.markdown("---")
    
    if st.button("🚀 Open Custom Model Builder", type="primary", use_container_width=True):
        st.switch_page("pages/4_custom_model.py")
    
    st.info("Click the button above to access the full Custom Model Builder with transition matrix input and scenario adjustments.")


# ============================================================================
# PAGE 1: INTRODUCTION
# ============================================================================

def show_introduction_page():
    """Introduction page with model overview"""
    
    st.title("🌍 LandGCH: Global Land-Use Change Model")
    st.markdown("### Interactive Dashboard for Land-Use Projections (2020-2050)")
    
    st.markdown("---")
    
    # Model description - new content from requirements
    st.markdown("""
    Land use change is a critical driver of global environmental change, affecting water resources, 
    carbon sequestration, economic development, climate regulation, and food security. To assess 
    future changes, we developed a comprehensive land use projection model for 227 states globally, 
    spanning the period 2020-2050. Our approach leverages six decades of historical land use data 
    combined with a spatially-explicit, time-varying Markov Chain framework to capture the complex 
    dynamics of land use transitions while maintaining physical consistency and enabling rigorous validation.
    """)
    
    st.markdown("---")
    
    # Summary of approach
    st.markdown("## 📋 Summary of the LandGCH Approach")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        | Component | Description |
        |-----------|-------------|
        | **Historical Data** | HILDA+ v2.0 (2025) - annual, 1 km² resolution, 1960-2020 (Winkler et al.) |
        | **State Boundaries** | Natural Earth (v5.1.1) |
        | **Method** | Time-varying Markov Chain model |
        | **Validation** | Multi-period (10/20/30 years) |
        | **Scenarios** | Multiple dietary and land use policy interventions |
        | **Resolution** | Country-level projections with 7 land-use classes |
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Key Features of the Approach
        
        - **Spatial overlay**: HILDA+ with Natural Earth boundaries
        - **Transition matrices**: 60 annual + 3 multi-year period matrices (10/20/30 years)
        - **Time-Varying Markov Chain**: Captures non-linearities and heterogeneous transitions
        - **Validation**: Hierarchical time-window strategy with hindcasting
        - **Uncertainty**: Monte Carlo simulation (100 realizations) + bias correction
        """)
    
    st.markdown("---")
    
    # Land-Use Classes
    st.markdown("## 📊 Land-Use Classes")
    st.markdown("*Capturing primary functional distinctions relevant to economic development, carbon and water resource analyses*")
    
    land_use_descriptions = {
        "Crops": "Annual and perennial cropland under agricultural production",
        "TreeCrops": "Permanent tree crop plantations (same as in original HILDA+)",
        "Forest": "Natural and semi-natural forest ecosystems (evergreen, deciduous, mixed)",
        "Grassland": "Natural grasslands, pastures, and rangelands",
        "Urban": "Built-up areas including residential, commercial, and industrial zones",
        "Water": "Inland water bodies, rivers, and wetlands",
        "Other": "Barren land, bare soil, ice, and unclassified areas"
    }
    
    cols = st.columns(4)
    for i, (lu, desc) in enumerate(land_use_descriptions.items()):
        with cols[i % 4]:
            color = COLORS.get(lu, '#999999')
            st.markdown(f"""
            <div style='background-color: {color}20; border-left: 4px solid {color}; padding: 0.5rem; margin-bottom: 0.5rem; border-radius: 0.25rem;'>
                <strong style='color: {color};'>{lu}</strong><br>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Timeline
    st.markdown("## 📅 Timeline")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Historical**: 1960-2020")
    with col2:
        st.markdown("**Projections**: 2021-2050")
    with col3:
        st.markdown("**Scenarios applied**: 2026-2050")
    
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
    st.info("☝️ Use the tabs above to explore **Global Results**, **Country Explorer**, or **Scenario Comparison**")

# ============================================================================
# PAGE 2: GLOBAL RESULTS
# ============================================================================

def show_global_results():
    """Global results page with validation and projections"""
    
    st.title("🌍 Global Results")
    st.markdown("### Land Use Projection Validation and Global Trends")
    
    st.markdown("---")
    
    # Section 1: Validation
    st.markdown("## 📊 Land Use Projection Validation")
    
    st.markdown("""
    Our validation strategy uses a hierarchical time-window approach, selecting the optimal 
    transition matrix period based on empirical validation performance:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Recent Period (2010-2020)", "105 countries (46%)")
        st.caption("Predominantly stable, high-income nations with consistent land use trends")
    
    with col2:
        st.metric("Medium Period (2000-2020)", "7 countries (3%)")
        st.caption("Intermediate cases with recent volatility but stable long-term trends")
    
    with col3:
        st.metric("Long Period (1990-2020)", "116 countries (51%)")
        st.caption("Typically developing countries with long-term structural transitions, or large developed countries with recently changed policies")
    
    # Interactive validation map
    st.markdown("---")
    st.markdown("### Validation Period Selection by Country")
    
    # Country to validation period mapping (recreated from original figure)
    validation_data = {
        'recent': ['USA', 'CAN', 'RUS', 'CHN', 'AUS', 'NZL', 'JPN', 'KOR', 'TWN',
                   'DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'PRT', 'NLD', 'BEL', 'AUT', 'CHE',
                   'POL', 'CZE', 'SVK', 'HUN', 'ROU', 'BGR', 'GRC', 'HRV', 'SVN',
                   'DNK', 'SWE', 'NOR', 'FIN', 'EST', 'LVA', 'LTU', 'IRL', 'ISL',
                   'SAU', 'ARE', 'KWT', 'QAT', 'BHR', 'OMN', 'ISR', 'JOR',
                   'EGY', 'MAR', 'TUN', 'DZA', 'LBY',
                   'ZAF', 'BWA', 'NAM', 'KEN', 'TZA', 'UGA', 'RWA',
                   'CHL', 'URY', 'PRY',
                   'MYS', 'SGP', 'BRN', 'PHL',
                   'TUR', 'GEO', 'ARM', 'AZE', 'KAZ', 'UZB', 'TKM'],
        'medium': ['TCD', 'CAF', 'SSD', 'ERI', 'SOM', 'YEM', 'AFG'],
        'long': ['BRA', 'ARG', 'COL', 'PER', 'VEN', 'ECU', 'BOL', 'GUY', 'SUR',
                 'MEX', 'GTM', 'HND', 'SLV', 'NIC', 'CRI', 'PAN', 'BLZ',
                 'CUB', 'DOM', 'HTI', 'JAM',
                 'IND', 'PAK', 'BGD', 'NPL', 'LKA', 'MMR', 'THA', 'VNM', 'LAO', 'KHM',
                 'IDN', 'PNG',
                 'NGA', 'GHA', 'CIV', 'CMR', 'COD', 'COG', 'AGO', 'MOZ', 'ZMB', 'ZWE',
                 'MWI', 'MDG', 'SEN', 'MLI', 'BFA', 'NER', 'GIN', 'SLE', 'LBR',
                 'ETH', 'SDN',
                 'IRN', 'IRQ', 'SYR', 'LBN',
                 'UKR', 'BLR', 'MDA']
    }
    
    map_data = []
    for period, countries in validation_data.items():
        for country in countries:
            map_data.append({'Country': country, 'Period': period})
    
    df_validation = pd.DataFrame(map_data)
    period_labels = {
        'recent': 'Recent (2010-2020)',
        'medium': 'Medium (2000-2020)',
        'long': 'Long (1990-2020)'
    }
    df_validation['Period_Label'] = df_validation['Period'].map(period_labels)
    
    fig_validation = px.choropleth(
        df_validation,
        locations='Country',
        color='Period_Label',
        color_discrete_map={
            'Recent (2010-2020)': '#1f77b4',
            'Medium (2000-2020)': '#d3d3d3', 
            'Long (1990-2020)': '#7f7f7f'
        },
        title='Validation Period Used per Country',
        hover_name='Country',
        hover_data={'Period_Label': True, 'Country': False}
    )
    
    fig_validation.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth', bgcolor='rgba(0,0,0,0)'),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text='Validation Period'
    )
    
    st.plotly_chart(fig_validation, use_container_width=True)
    
    st.markdown("---")
    
    # Section 2: Global Overview
    st.markdown("## 🌐 Global Land Use Change Overview (2020-2050)")
    
    st.markdown("""
    **Overall:** slight cropland expansion (+0.8%), rapid tree crop growth (+32.7%), reflecting 
    documented trends in oil palm (Indonesia, Malaysia, West Africa), coffee, cocoa, and fruit 
    plantations. Unfortunately, the continued forest loss (-2.3%) aligns with IPCC baseline scenarios 
    showing persistent tropical deforestation partially offset by temperate reforestation, while the 
    accelerating urbanization (+16.8%) matches UN-Habitat projections of urban population share 
    increasing from 55% (2018) to 68% (2050).
    """)
    
    # Load data and create visualizations
    try:
        df_bau = load_data('bau')
        if df_bau is not None:
            df_2020 = df_bau[df_bau['Year'] == 2020]
            df_2050 = df_bau[df_bau['Year'] == 2050]
            
            # Calculate changes
            change_data = []
            for lu in LAND_CLASSES:
                area_2020 = df_2020[lu].sum()
                area_2050 = df_2050[lu].sum()
                abs_change = area_2050 - area_2020
                pct_change = (abs_change / area_2020 * 100) if area_2020 > 0 else 0
                change_data.append({
                    'Land Use': lu,
                    'Area 2020': area_2020,
                    'Area 2050': area_2050,
                    'Change km²': abs_change,
                    'Change %': pct_change
                })
            
            df_changes = pd.DataFrame(change_data)
            
            # Display summary table
            st.markdown("### Absolute (km²) and Relative (%) Changes in Area")
            display_df = df_changes.copy()
            display_df['Area 2020'] = display_df['Area 2020'].apply(lambda x: f"{x:,.0f}")
            display_df['Area 2050'] = display_df['Area 2050'].apply(lambda x: f"{x:,.0f}")
            display_df['Change km²'] = display_df['Change km²'].apply(lambda x: f"{x:+,.0f}")
            display_df['Change %'] = display_df['Change %'].apply(lambda x: f"{x:+.1f}%")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Chart 1: Grouped bars (2020 vs 2050)
            st.markdown("### Global Land Use — 2020 vs 2050 (BAU)")
            fig_grouped = go.Figure()
            fig_grouped.add_trace(go.Bar(
                name='2020',
                x=df_changes['Land Use'],
                y=df_changes['Area 2020'],
                marker_color='#5b9bd5'
            ))
            fig_grouped.add_trace(go.Bar(
                name='2050 (BAU)',
                x=df_changes['Land Use'],
                y=df_changes['Area 2050'],
                marker_color='#ff8c00'
            ))
            fig_grouped.update_layout(
                barmode='group',
                yaxis_title='Area (km²)',
                template='plotly_white',
                height=450,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig_grouped, use_container_width=True)
            
            # Chart 2: Absolute change bar chart
            st.markdown("### Absolute Change in Area: 2020 → 2050 (km²)")
            colors_abs = ['#2ca02c' if v >= 0 else '#d62728' for v in df_changes['Change km²']]
            fig_abs = go.Figure(go.Bar(
                x=df_changes['Land Use'],
                y=df_changes['Change km²'],
                marker_color=colors_abs,
                text=[f"{v:+,.0f}" for v in df_changes['Change km²']],
                textposition='outside'
            ))
            fig_abs.update_layout(
                yaxis_title='Absolute Change (km²)',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_abs, use_container_width=True)
            
            # Chart 3: Percent change bar chart
            st.markdown("### Percent Change in Area: 2020 → 2050 (%)")
            fig_pct = go.Figure(go.Bar(
                x=df_changes['Land Use'],
                y=df_changes['Change %'],
                marker_color='#1f77b4',
                text=[f"{v:+.1f}%" for v in df_changes['Change %']],
                textposition='outside'
            ))
            fig_pct.update_layout(
                yaxis_title='Percent Change (%)',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_pct, use_container_width=True)
            
            # Chart 4: Composition pies (2020 vs 2050)
            st.markdown("### Total Distribution of 2020 vs 2050 Land Uses")
            fig_pies = make_subplots(rows=1, cols=2, subplot_titles=(
                f"2020 (Total: {df_changes['Area 2020'].sum():,.0f} km²)",
                f"2050 (Total: {df_changes['Area 2050'].sum():,.0f} km²)"
            ), specs=[[{'type': 'pie'}, {'type': 'pie'}]])
            
            colors = [COLORS[lu] for lu in LAND_CLASSES]
            fig_pies.add_trace(go.Pie(
                labels=df_changes['Land Use'], values=df_changes['Area 2020'],
                marker_colors=colors, textinfo='percent', name='2020'
            ), row=1, col=1)
            fig_pies.add_trace(go.Pie(
                labels=df_changes['Land Use'], values=df_changes['Area 2050'],
                marker_colors=colors, textinfo='percent', name='2050'
            ), row=1, col=2)
            fig_pies.update_layout(height=450, showlegend=True)
            st.plotly_chart(fig_pies, use_container_width=True)
            
            st.markdown("---")
            
            # Section 3: European Results
            st.markdown("## 🇪🇺 Indicative Summary: European Land Use Projections to 2050")
            
            st.markdown("""
            Indicatively for Europe, overall, forests are broadly stable-to-growing in some countries 
            while a few countries show small declines. Cropland area moves only slightly, with modest 
            increases in Spain but small declines in several others (e.g., Poland, Romania, Greece). 
            Tree-crops and some traditional agricultural uses decline across most countries, whereas 
            urban area steadily expands everywhere. Water bodies remain essentially constant and 
            grassland shows mixed changes (small declines in some northern states, slight rises in 
            countries like Italy/Romania).
            """)
            
            # European countries chart
            eu_countries = ['DEU', 'FRA', 'ESP', 'ITA', 'POL', 'ROU', 'GRC', 'NLD', 'BEL', 'PRT', 'SWE', 'AUT']
            df_eu_2020 = df_2020[df_2020['Country'].isin(eu_countries)]
            df_eu_2050 = df_2050[df_2050['Country'].isin(eu_countries)]
            
            if not df_eu_2020.empty:
                eu_changes = []
                for country in eu_countries:
                    c_2020 = df_eu_2020[df_eu_2020['Country'] == country]
                    c_2050 = df_eu_2050[df_eu_2050['Country'] == country]
                    if not c_2020.empty and not c_2050.empty:
                        for lu in ['Forest', 'Crops', 'Urban', 'Grassland']:
                            val_2020 = c_2020[lu].values[0]
                            val_2050 = c_2050[lu].values[0]
                            pct = ((val_2050 - val_2020) / val_2020 * 100) if val_2020 > 0 else 0
                            eu_changes.append({'Country': country, 'Land Use': lu, 'Change %': pct})
                
                if eu_changes:
                    df_eu = pd.DataFrame(eu_changes)
                    fig_eu = px.bar(
                        df_eu, x='Country', y='Change %', color='Land Use',
                        barmode='group', title='European Countries: Projected Change 2020-2050 (%)',
                        color_discrete_map={'Forest': COLORS['Forest'], 'Crops': COLORS['Crops'], 
                                           'Urban': COLORS['Urban'], 'Grassland': COLORS['Grassland']}
                    )
                    fig_eu.update_layout(template='plotly_white', height=450)
                    st.plotly_chart(fig_eu, use_container_width=True)
            
            st.markdown("---")
            
            # Section 4: Decadal Changes
            st.markdown("## 📈 Decadal Changes of Main Land Uses Across Countries with Strong Variations")
            
            st.markdown("""
            Urban land expansion is consistently positive for all countries, forest cover is projected 
            to increase in most countries, particularly Egypt, which shows extremely high positive change 
            across all decades up to 2050. Conversely, Brazil, Australia, and Argentina are expected to 
            see declines, raising concerns about deforestation and ecosystem service losses in these major 
            forest-rich nations. Crops show mixed trends across countries. These changes underscore the 
            dynamic impacts of climate, policy, and market forces on agricultural land use globally.
            """)
            
            # Decadal heatmap for key countries
            key_countries = ['EGY', 'BRA', 'AUS', 'ARG', 'USA', 'CHN', 'IND', 'IDN', 'RUS', 'DEU']
            decades = [(2020, 2030), (2030, 2040), (2040, 2050)]
            
            heatmap_data = []
            for country in key_countries:
                country_data = df_bau[df_bau['Country'] == country]
                if not country_data.empty:
                    for start, end in decades:
                        row_start = country_data[country_data['Year'] == start]
                        row_end = country_data[country_data['Year'] == end]
                        if not row_start.empty and not row_end.empty:
                            for lu in ['Forest', 'Urban', 'Crops']:
                                val_start = row_start[lu].values[0]
                                val_end = row_end[lu].values[0]
                                pct = ((val_end - val_start) / val_start * 100) if val_start > 0 else 0
                                heatmap_data.append({
                                    'Country': country,
                                    'Decade': f'{start}-{end}',
                                    'Land Use': lu,
                                    'Change %': pct
                                })
            
            if heatmap_data:
                df_heat = pd.DataFrame(heatmap_data)
                
                # Create heatmap for each land use
                for lu in ['Forest', 'Urban', 'Crops']:
                    df_lu = df_heat[df_heat['Land Use'] == lu]
                    if not df_lu.empty:
                        pivot = df_lu.pivot(index='Country', columns='Decade', values='Change %')
                        
                        fig_heat = go.Figure(data=go.Heatmap(
                            z=pivot.values,
                            x=pivot.columns.tolist(),
                            y=pivot.index.tolist(),
                            colorscale='RdYlGn',
                            zmid=0,
                            text=np.round(pivot.values, 1),
                            texttemplate='%{text}%',
                            textfont={"size": 10},
                            colorbar=dict(title="% Change")
                        ))
                        fig_heat.update_layout(
                            title=f'{lu} - Decadal Change Across Countries (%)',
                            template='plotly_white',
                            height=400
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)
            
            st.markdown("---")
            
            # Section 5: Tropical Deforestation
            st.markdown("## 🌴 Main Land Use Change Results: Tropical Deforestation Sites")
            
            st.markdown("""
            Major tropical forest areas show ongoing declines through 2050. Cropland and tree-crop/plantation 
            areas expand noticeably, while grassland and urban footprints generally rise as well, indicating 
            conversion of natural land to agriculture and settlements.
            """)
            
            # Tropical countries
            tropical_countries = ['BRA', 'IDN', 'COD', 'PER', 'COL', 'VNM', 'MMR', 'MYS']
            df_trop_2020 = df_2020[df_2020['Country'].isin(tropical_countries)]
            df_trop_2050 = df_2050[df_2050['Country'].isin(tropical_countries)]
            
            if not df_trop_2020.empty:
                trop_changes = []
                for country in tropical_countries:
                    c_2020 = df_trop_2020[df_trop_2020['Country'] == country]
                    c_2050 = df_trop_2050[df_trop_2050['Country'] == country]
                    if not c_2020.empty and not c_2050.empty:
                        for lu in LAND_CLASSES:
                            val_2020 = c_2020[lu].values[0]
                            val_2050 = c_2050[lu].values[0]
                            pct = ((val_2050 - val_2020) / val_2020 * 100) if val_2020 > 0 else 0
                            trop_changes.append({'Country': country, 'Land Use': lu, 'Change %': pct})
                
                if trop_changes:
                    df_trop = pd.DataFrame(trop_changes)
                    
                    # Focus on Forest, Crops, TreeCrops, Urban
                    df_trop_main = df_trop[df_trop['Land Use'].isin(['Forest', 'Crops', 'TreeCrops', 'Urban', 'Grassland'])]
                    
                    fig_trop = px.bar(
                        df_trop_main, x='Country', y='Change %', color='Land Use',
                        barmode='group', title='Tropical Deforestation Sites: Projected Change 2020-2050 (%)',
                        color_discrete_map={lu: COLORS[lu] for lu in LAND_CLASSES}
                    )
                    fig_trop.update_layout(template='plotly_white', height=500)
                    st.plotly_chart(fig_trop, use_container_width=True)
                    
    except Exception as e:
        st.warning(f"Could not load data for visualizations: {e}")

# ============================================================================
# PAGE 3: COUNTRY EXPLORER
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
            index=None,
            placeholder="Choose a country..."
        )
        
        if selected_country is None:
            st.info("👆 Please select a country to view projections")
        
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
        # Check if country is selected
        if selected_country is None:
            st.markdown("## 👈 Select a Country")
            st.markdown("Use the dropdown on the left to choose a country and view its land-use projections.")
            return
        
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
# PAGE 4: SCENARIO COMPARISON
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
        quick_select = None
        if st.button("Top 10 by area"):
            # Calculate total area per country in 2020
            df_2020 = df_bau[df_bau['Year'] == 2020]
            country_areas = df_2020.groupby('Country')[LAND_CLASSES].sum().sum(axis=1).nlargest(10)
            quick_select = list(country_areas.index)
        
        selected_countries = st.multiselect(
            "Countries to compare",
            countries,
            default=quick_select if quick_select else [],
            placeholder="Choose countries..."
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Scenarios")
        scenarios_to_compare = st.multiselect(
            "Select scenarios",
            ['BAU', 'Fat', 'EAT', 'NDC', 'Afforest', 'Bioen', 'Yieldint', 'Landretir'],
            default=[],
            placeholder="Choose scenarios..."
        )
        
        st.markdown("---")
        
        st.markdown("### 🎨 Options")
        land_use_compare = st.selectbox(
            "Land-use type",
            LAND_CLASSES,
            index=None,
            placeholder="Choose land-use type..."
        )
        
        normalize = st.checkbox("Normalize to 2020 = 100", value=True)
    
    with col2:
        if not selected_countries:
            st.markdown("## 👈 Select Countries")
            st.markdown("Use the options on the left to choose countries, scenarios, and land-use type for comparison.")
            return
        
        if not scenarios_to_compare:
            st.info("👈 Please select at least one scenario to compare")
            return
        
        if land_use_compare is None:
            st.info("👈 Please select a land-use type to compare")
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
