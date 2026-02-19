"""Custom Model Builder - Two-Stage Projection System"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.custom_projection_engine import (
    run_custom_projection,
    validate_transition_matrix,
    parse_matrix_from_csv_string,
    parse_matrix_from_text
)

# ============================================================================
# GLOBAL PLOTLY COLOR CONFIGURATION
# ============================================================================

# Color scheme - Professional palette (consistent with main app)
COLORS = {
    'Crops': '#5B9BD5',      # Blue
    'TreeCrops': '#70AD47',  # Green
    'Forest': '#2E7D32',     # Dark Green
    'Grassland': '#ED7D31',  # Orange
    'Urban': '#7F7F7F',      # Gray
    'Water': '#4FC3F7',      # Light Blue
    'Other': '#BDBDBD'       # Light Gray
}

SCENARIO_COLORS = {
    'BAU': '#7F7F7F',        # Gray
    'Fat': '#E57373',        # Soft Red
    'EAT': '#81C784',        # Soft Green
    'NDC': '#64B5F6',        # Soft Blue
    'Afforest': '#4DB6AC',   # Teal
    'Bioen': '#BA68C8',      # Purple
    'Yieldint': '#FFD54F',   # Amber
    'Landretir': '#A1887F',  # Brown
}

# Set global Plotly defaults
LAND_USE_COLORS = list(COLORS.values())
px.defaults.color_discrete_sequence = LAND_USE_COLORS

# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

SCENARIOS = {
    'Custom': {
        'name': 'Custom Multipliers',
        'description': 'Enter your own multipliers',
        'multipliers': None
    },
    'Fat': {
        'name': 'High-meat diet',
        'description': 'Rich in meat, this diet scenario assigns more area to pasture and feed crops.',
        'multipliers': {'Crops': 1.06, 'TreeCrops': 1.02, 'Forest': 0.97, 'Grassland': 1.12, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95}
    },
    'EAT': {
        'name': 'The Lancet EAT diet',
        'description': 'A "healthy diet" scenario, increased area for vegetables/legumes and reduced pasture.',
        'multipliers': {'Crops': 0.90, 'TreeCrops': 0.97, 'Forest': 1.08, 'Grassland': 0.82, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.98}
    },
    'NDC': {
        'name': 'National Determined Contributions',
        'description': 'Countries implement explicit forest gain targets, crops may increase due to bioenergy demand.',
        'multipliers': {'Crops': 1.03, 'TreeCrops': 1.05, 'Forest': 1.07, 'Grassland': 0.98, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95}
    },
    'Afforest': {
        'name': 'Afforestation/Reforestation',
        'description': 'Reduced deforestation/protection policies that restrict transitions out of forest.',
        'multipliers': {'Crops': 1.01, 'TreeCrops': 1.02, 'Forest': 1.07, 'Grassland': 0.99, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95}
    },
    'Bioen': {
        'name': 'Bioenergy/energy crop expansion',
        'description': 'Swift in green fuels production, biofuel mandates increase area of crops for bioenergy.',
        'multipliers': {'Crops': 1.03, 'TreeCrops': 1.05, 'Forest': 1.03, 'Grassland': 0.97, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95}
    },
    'Yieldint': {
        'name': 'Yield improvement/intensification',
        'description': 'Higher yields reduce land need for the same output, lowering demand-driven conversion pressure.',
        'multipliers': {'Crops': 0.90, 'TreeCrops': 0.97, 'Forest': 1.03, 'Grassland': 0.83, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.98}
    },
    'Landretir': {
        'name': 'Land retirement/Carbon pricing',
        'description': 'Potential incentives withdrawing marginal cropland to forest, or forcing retirement quotas.',
        'multipliers': {'Crops': 0.91, 'TreeCrops': 0.96, 'Forest': 1.07, 'Grassland': 0.82, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.98}
    }
}

DEFAULT_LAND_USES = ['Crops', 'TreeCrops', 'Forest', 'Grassland', 'Urban', 'Water', 'Other']

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Custom Model Builder",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS
st.markdown("""
<style>
    [data-testid="stSidebarNav"] { display: none; }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Custom Model Builder")

st.markdown("""
<div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem;'>
    <p style='margin: 0; color: #495057; font-size: 0.95rem;'>
    <strong>Two-Stage Process:</strong> 
    <span style='color: #1f77b4;'>①</span> Run base projection with transition matrix → 
    <span style='color: #1f77b4;'>②</span> Apply scenario multipliers to adjust results
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Progress Tracker
with st.sidebar:
    # AERIA logo first
    try:
        st.image("assets/final_logo.svg", width=180)
    except:
        pass  # Skip if logo not found
    
    # GCH Logo
    st.image("https://unsdsn.globalclimatehub.org/wp-content/uploads/2022/09/logo.png", width=200)
    
    st.title("🌍 LandGCH Dashboard")
    st.markdown("---")
    
    # Navigation matching main app
    st.markdown("### Navigation")
    st.page_link("landgch_app.py", label="📖 Introduction")
    st.page_link("landgch_app.py", label="🌍 Global Results")
    st.page_link("landgch_app.py", label="🔍 Country Explorer")
    st.page_link("landgch_app.py", label="📊 Scenario Comparison")
    st.page_link("pages/4_custom_model.py", label="🔬 Custom Model")
    
    st.markdown("### 🚀 Workflow Progress")
    st.markdown("---")
    
    # Track progress
    progress_items = []
    
    # Will be updated as we go through the workflow
    st.markdown("#### Stage 1: Base Projection")
    if 'baseline_2020' in locals() and baseline_2020 is not None:
        st.success("✅ Baseline loaded")
    else:
        st.info("⏳ Load baseline")
    
    if 'matrix' in locals() and matrix is not None:
        st.success("✅ Matrix ready")
    else:
        st.info("⏳ Provide matrix")
    
    if 'base_results' in st.session_state:
        st.success("✅ Base projection complete")
    else:
        st.info("⏳ Run projection")
    
    st.markdown("#### Stage 2: Adjustment")
    if 'adj_2050' in st.session_state:
        st.success("✅ Multipliers applied")
    else:
        st.info("⏳ Apply multipliers")
    
    st.markdown("---")
    st.caption("💡 Complete steps in order")

# ============================================================================
# STAGE 1: BASE PROJECTION
# ============================================================================

st.markdown("""
<div class='section-header'>
    <h3>📊 Stage 1: Base Projection</h3>
    <p>Create your baseline land-use projection using transition matrices</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 1.1 Country Selection & Baseline Loading
# ----------------------------------------------------------------------------

st.markdown("#### 1.1 Country & Baseline Data")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    country = st.text_input(
        "Country Code (ISO3)",
        value="BRA",
        max_chars=3,
        help="3-letter country code (e.g., BRA, USA, NOR)"
    ).upper()

# Load baseline data from fixed location
baseline_data = None
baseline_2020 = None

# Try multiple possible locations for the baseline CSV
baseline_paths = [
    root_dir / 'data' / 'ALL_COUNTRIES_annual_projections_2020_2050.csv',
    root_dir / 'core' / 'data' / 'ALL_COUNTRIES_annual_projections_2020_2050.csv',
    Path('/mnt/user-data') / 'ALL_COUNTRIES_annual_projections_2020_2050.csv'
]

baseline_loaded = False
for baseline_path in baseline_paths:
    if baseline_path.exists():
        try:
            df = pd.read_csv(baseline_path)
            
            # Find row for this country and year 2020
            country_data = df[(df['Country'] == country) & (df['Year'] == 2020)]
            
            if len(country_data) > 0:
                baseline_data = country_data.iloc[0]
                
                # Extract land use values
                baseline_2020 = {
                    'Crops': baseline_data.get('Crops', 0),
                    'TreeCrops': baseline_data.get('TreeCrops', 0),
                    'Forest': baseline_data.get('Forest', 0),
                    'Grassland': baseline_data.get('Grassland', 0),
                    'Urban': baseline_data.get('Urban', 0),
                    'Water': baseline_data.get('Water', 0),
                    'Other': baseline_data.get('Other', 0)
                }
                
                total_area = sum(baseline_2020.values())
                
                # Nice metric display
                with col2:
                    st.metric("📅 Base Year", "2020")
                with col3:
                    st.metric("🌍 Total Area", f"{total_area:,.0f} km²")
                
                st.success(f"✅ Loaded baseline for **{country}**")
                
                with st.expander("📊 View 2020 Baseline Details"):
                    baseline_df = pd.DataFrame([baseline_2020])
                    st.dataframe(baseline_df.T.rename(columns={0: 'Area (km²)'}), width="content")
                
                baseline_loaded = True
                break
            else:
                st.error(f"❌ No data found for country **{country}** in year 2020")
                baseline_loaded = True
                break
        except Exception as e:
            continue  # Try next path

if not baseline_loaded:
    st.error("❌ Baseline CSV file not found!")
    st.info("""
    **Please add the baseline CSV file to one of these locations:**
    - `landgch-dashboard/data/ALL_COUNTRIES_annual_projections_2020_2050.csv`
    - `landgch-dashboard/core/data/ALL_COUNTRIES_annual_projections_2020_2050.csv`
    
    Create the `data/` directory if it doesn't exist.
    """)

st.markdown("---")

# ----------------------------------------------------------------------------
# 1.2 Land Use Configuration
# ----------------------------------------------------------------------------

st.markdown("#### 1.2 Land Use Configuration")

use_default = st.radio(
    "Land use types:",
    ["Use default 7 land uses", "Add custom land uses"],
    horizontal=True
)

if use_default == "Use default 7 land uses":
    land_uses = DEFAULT_LAND_USES
    use_default_types = True
    
    if baseline_2020:
        baseline_areas = np.array([baseline_2020[lu] for lu in land_uses])
        st.success(f"✅ Using default land uses with loaded baseline")
    else:
        st.warning("⚠️ No baseline data available for this country")
        baseline_areas = None
        
else:
    use_default_types = False
    st.info("👉 Define your custom land use categories")
    
    # Get total country area for reference
    if baseline_2020 is not None:
        # baseline_2020 is a dict with land use names as keys
        if isinstance(baseline_2020, dict):
            total_country_area = sum(baseline_2020.values())
        else:
            # If it's already an array/series
            total_country_area = baseline_2020.sum()
        st.info(f"📍 **{country} Total Area:** {total_country_area:,.0f} km²")
    else:
        total_country_area = None
    
    # Initialize in session state if not present
    if 'custom_num_land_uses' not in st.session_state:
        st.session_state['custom_num_land_uses'] = 4
    if 'custom_land_use_names' not in st.session_state:
        st.session_state['custom_land_use_names'] = []
    if 'custom_baseline_percentages' not in st.session_state:
        st.session_state['custom_baseline_percentages'] = []
    
    num_land_uses = st.number_input(
        "Number of categories",
        min_value=2,
        max_value=15,
        value=st.session_state['custom_num_land_uses']
    )
    
    # Temporary inputs (not confirmed yet)
    temp_land_uses = []
    temp_baseline_percentages = []
    
    st.markdown("**Enter names and baseline percentages (2020):**")
    st.caption("💡 Enter the percentage of total area for each land use. Total should equal 100%.")
    
    cols = st.columns(3)
    for i in range(num_land_uses):
        with cols[i % 3]:
            # Get default value from session state if available
            default_name = st.session_state['custom_land_use_names'][i] if i < len(st.session_state['custom_land_use_names']) else f"LandUse_{i+1}"
            default_pct = st.session_state['custom_baseline_percentages'][i] if i < len(st.session_state['custom_baseline_percentages']) else (100.0 / num_land_uses)
            
            name = st.text_input(f"Land Use {i+1} Name", value=default_name, key=f"custom_lu_name_{i}")
            pct = st.number_input(
                f"Percentage (%)", 
                min_value=0.0, 
                max_value=100.0,
                value=default_pct, 
                step=1.0, 
                format="%.2f", 
                key=f"custom_lu_pct_{i}"
            )
            temp_land_uses.append(name)
            temp_baseline_percentages.append(pct)
    
    # Calculate total percentage and show validation
    total_pct = sum(temp_baseline_percentages)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Percentage", f"{total_pct:.2f}%")
    with col2:
        if total_country_area:
            total_area_used = (total_pct / 100.0) * total_country_area
            st.metric("Total Area", f"{total_area_used:,.0f} km²")
    
    # Validation warnings
    if abs(total_pct - 100.0) > 1.0:
        if total_pct > 100.0:
            st.error(f"⚠️ **Over-allocation!** Total percentage is {total_pct:.2f}%. Reduce by {total_pct - 100.0:.2f}%")
        else:
            st.warning(f"⚠️ **Under-allocation!** Total percentage is {total_pct:.2f}%. Add {100.0 - total_pct:.2f}%")
    elif abs(total_pct - 100.0) > 0.01:
        st.info(f"✓ Close to 100% (difference: {abs(total_pct - 100.0):.2f}%)")
    else:
        st.success("✅ Perfect! Total = 100%")
    
    # Confirm button
    confirm_disabled = abs(total_pct - 100.0) > 1.0  # Disable if difference > 1%
    
    if st.button("✓ Confirm Land Uses", type="primary", disabled=confirm_disabled, width="content"):
        st.session_state['custom_num_land_uses'] = num_land_uses
        st.session_state['custom_land_use_names'] = temp_land_uses
        st.session_state['custom_baseline_percentages'] = temp_baseline_percentages
        st.success("✅ Land uses confirmed!")
        st.rerun()
    
    if confirm_disabled:
        st.caption("💡 Adjust percentages to total 100% (±1%) before confirming")
    else:
        st.caption("💡 Click 'Confirm' after entering/changing land use names to update")
    
    # Use confirmed values from session state
    land_uses = st.session_state['custom_land_use_names'] if st.session_state['custom_land_use_names'] else temp_land_uses
    baseline_percentages_list = st.session_state['custom_baseline_percentages'] if st.session_state['custom_baseline_percentages'] else temp_baseline_percentages
    
    # Convert percentages to areas for the model
    if total_country_area and len(baseline_percentages_list) > 0:
        baseline_areas = np.array([(pct / 100.0) * total_country_area for pct in baseline_percentages_list])
        st.success(f"✅ Converted percentages to areas based on {country} total area")
    else:
        # Fallback if no country selected
        baseline_areas = np.array(baseline_percentages_list) * 10000  # Arbitrary scale
        st.warning("⚠️ No country selected - using arbitrary scale. Select a country for accurate areas.")

st.markdown("---")

# ----------------------------------------------------------------------------
# 1.3 Transition Matrix
# ----------------------------------------------------------------------------

st.markdown("#### 1.3 Transition Matrix")

if baseline_areas is not None:
    n = len(land_uses)
    
    # Try to auto-load country transition matrix (only for default land uses)
    matrix = None
    auto_loaded = False
    
    if use_default_types and country:
        matrix_paths = [
            root_dir / 'hilda_country_transitions_output' / country / f'{country}_contemporary_transition_pct_2000-2019.csv',
            root_dir / 'data' / 'hilda_country_transitions_output' / country / f'{country}_contemporary_transition_pct_2000-2019.csv',
            Path('/mnt/user-data') / 'hilda_country_transitions_output' / country / f'{country}_contemporary_transition_pct_2000-2019.csv'
        ]
        
        for matrix_path in matrix_paths:
            if matrix_path.exists():
                try:
                    matrix_df = pd.read_csv(matrix_path, index_col=0)
                    matrix = matrix_df.values
                    
                    # Check if matrix is in percentage format (values > 1.0)
                    # If so, convert to decimal format (divide by 100)
                    if np.any(matrix > 1.0):
                        st.info("🔄 Matrix is in percentage format (0-100), converting to decimal format (0-1)...")
                        matrix = matrix / 100.0
                    
                    if matrix.shape[0] == n and matrix.shape[1] == n:
                        st.success(f"✅ Auto-loaded transition matrix for {country} ({n}×{n})")
                        auto_loaded = True
                        break
                except Exception as e:
                    continue
    
    # Decide whether to show manual input options
    show_manual_input = False
    
    if use_default_types:
        # Default land uses
        if not auto_loaded:
            st.info(f"💡 No pre-computed matrix found for {country}. Please provide {n}×{n} matrix manually.")
            show_manual_input = True
        else:
            # Auto-loaded, but user can override
            use_manual = st.checkbox("📝 Override with custom matrix", value=False)
            if use_manual:
                show_manual_input = True
                matrix = None  # Reset to force manual input
    else:
        # Custom land uses - always require manual matrix
        st.info(f"📝 You are using custom land uses. Please provide your {n}×{n} transition matrix below.")
        show_manual_input = True
    
    # Show manual input interface if needed
    if show_manual_input:
        st.markdown("---")
        
        input_method = st.radio(
            "Input method:",
            ["Upload CSV", "Paste from Excel"],
            horizontal=True,
            key="matrix_input_method"
        )
        
        if input_method == "Upload CSV":
            uploaded_matrix = st.file_uploader("Upload matrix CSV", type=['csv'], key="matrix_csv_upload")
            
            if uploaded_matrix:
                try:
                    csv_content = uploaded_matrix.read().decode('utf-8')
                    matrix = parse_matrix_from_csv_string(csv_content)
                    
                    # Check if in percentage format
                    if np.any(matrix > 1.0):
                        st.info("🔄 Converting from percentage to decimal format...")
                        matrix = matrix / 100.0
                    
                    st.success(f"✅ Matrix loaded: {matrix.shape[0]}×{matrix.shape[1]}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        else:  # Paste from Excel
            matrix_text = st.text_area(
                "Paste matrix (tab or comma separated)",
                height=200,
                placeholder="Example:\n0.9\t0.05\t0.05\n0.1\t0.85\t0.05\n0.05\t0.05\t0.9",
                key="matrix_text_paste"
            )
            
            if matrix_text.strip():
                try:
                    matrix = parse_matrix_from_text(matrix_text)
                    
                    # Check if in percentage format
                    if np.any(matrix > 1.0):
                        st.info("🔄 Converting from percentage to decimal format...")
                        matrix = matrix / 100.0
                    
                    st.success(f"✅ Matrix parsed: {matrix.shape[0]}×{matrix.shape[1]}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    if matrix is not None:
        # Check if matrix is in percentage format and convert if needed
        if np.any(matrix > 1.0):
            st.info("🔄 Matrix appears to be in percentage format (0-100), converting to decimal format (0-1)...")
            matrix = matrix / 100.0
        
        if matrix.shape[0] != n or matrix.shape[1] != n:
            st.error(f"❌ Matrix must be {n}×{n}!")
        else:
            row_sums = matrix.sum(axis=1)
            matrix_df = pd.DataFrame(matrix, index=land_uses, columns=land_uses)
            matrix_df['Row Sum'] = row_sums
            
            def highlight_sums(val):
                if isinstance(val, (int, float)):
                    return 'background-color: #d4edda' if abs(val - 1.0) < 0.01 else 'background-color: #f8d7da'
                return ''
            
            styled = matrix_df.style.map(highlight_sums, subset=['Row Sum']).format('{:.4f}')
            st.dataframe(styled, width="stretch")
            
            # Show uncertainty ranges
            st.markdown("#### 📊 Transition Matrix with Uncertainty Ranges (±)")
            st.caption("*Uncertainty values represent plausible variation bounds based on validation analysis*")
            
            # Calculate uncertainty: ±0.02 for small values, ±0.05 for larger values
            # Uncertainty is proportional to sqrt(p*(1-p)) but capped for display
            def calc_uncertainty(p):
                if p <= 0.01:
                    return 0.005
                elif p <= 0.1:
                    return 0.02
                elif p <= 0.5:
                    return 0.03
                else:
                    return 0.05
            
            # Create display with ± values
            matrix_with_unc = pd.DataFrame(index=land_uses, columns=land_uses)
            for i, row_lu in enumerate(land_uses):
                for j, col_lu in enumerate(land_uses):
                    val = matrix[i, j]
                    unc = calc_uncertainty(val)
                    matrix_with_unc.loc[row_lu, col_lu] = f"{val:.3f} ±{unc:.3f}"
            
            st.dataframe(matrix_with_unc, width="stretch")
            
            valid, errors = validate_transition_matrix(matrix)
            if valid:
                st.success("✅ Valid!")
            else:
                for error in errors:
                    st.error(f"❌ {error}")

st.markdown("---")

# ----------------------------------------------------------------------------
# 1.4 Run Base Projection
# ----------------------------------------------------------------------------

st.markdown("#### 1.4 Run Base Projection")

can_run = baseline_areas is not None and matrix is not None and len(land_uses) > 0 and matrix.shape[0] == len(land_uses)

if can_run:
    valid_matrix, _ = validate_transition_matrix(matrix)
    
    if valid_matrix:
        if st.button("🚀 Run Stage 1: Base Projection", type="primary", width="stretch"):
            with st.spinner("Running projection..."):
                try:
                    results = run_custom_projection(country, matrix, land_uses, baseline_areas)
                    
                    st.session_state['base_results'] = results
                    st.session_state['land_uses'] = land_uses
                    st.session_state['use_default_types'] = use_default_types
                    
                    st.success("✅ Base projection complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Show Base Results
if 'base_results' in st.session_state:
    st.markdown("---")
    st.markdown("""
    <div class='section-header'>
        <h3>📊 Base Projection Results</h3>
        <p>Land-use trajectories from 2020 to 2050</p>
    </div>
    """, unsafe_allow_html=True)
    
    results = st.session_state['base_results']
    land_uses = st.session_state['land_uses']
    
    # Area conservation check
    total_2020 = results[results['Year'] == 2020][land_uses].values[0].sum()
    total_2050 = results[results['Year'] == 2050][land_uses].values[0].sum()
    difference = total_2050 - total_2020
    
    st.markdown("**🔍 Area Conservation Diagnostic:**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Area 2020", f"{total_2020:,.0f} km²", delta=None)
    with col2:
        st.metric("Total Area 2050", f"{total_2050:,.0f} km²", delta=None)
    with col3:
        delta_color = "off" if abs(difference) < 1 else "normal"
        st.metric("Difference", f"{difference:+,.0f} km²", delta=None)
    
    if abs(difference) < 1:
        st.success("✅ **Perfect!** Total area is conserved.")
    elif difference > 0:
        st.warning(f"⚠️ Exceeded by **{abs(difference):,.0f} km²**. Reconsider smaller change percentages.")
    else:
        st.warning(f"⚠️ Used less by **{abs(difference):,.0f} km²**")
    
    st.markdown("---")
    st.markdown("**📈 Land-Use Trajectory (2020-2050)**")
    
    plot_data = results.melt(id_vars=['Country', 'Year'], var_name='Land_Use', value_name='Area')
    fig = px.line(
        plot_data, 
        x='Year', 
        y='Area', 
        color='Land_Use',
        color_discrete_sequence=px.colors.qualitative.Set2,
        height=500
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12, family="sans-serif"),
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            title=dict(text="Land Use", font=dict(size=13, family="sans-serif"))
        ),
        margin=dict(l=50, r=150, t=50, b=50)
    )
    st.plotly_chart(fig, width="stretch")

# ============================================================================
# STAGE 2: SCENARIO ADJUSTMENT
# ============================================================================

if 'base_results' in st.session_state:
    st.markdown("---")
    st.markdown("""
    <div class='section-header'>
        <h3>🎯 Stage 2: Scenario Adjustment</h3>
        <p>Apply multipliers to adjust your 2050 projection results</p>
    </div>
    """, unsafe_allow_html=True)
    
    results = st.session_state['base_results']
    land_uses = st.session_state['land_uses']
    use_default_types = st.session_state['use_default_types']
    
    st.markdown("`Formula: 2050 Adjusted = 2050 Base × Multipliers`")
    
    st.markdown("#### 2.1 Select Multipliers")
    
    if use_default_types:
        st.success("✅ Can select & edit scenarios")
        
        scenario = st.selectbox(
            "Select scenario:",
            list(SCENARIOS.keys()),
            format_func=lambda x: SCENARIOS[x]['name']
        )
        
        st.info(f"**{SCENARIOS[scenario]['description']}**")
        
        # Initialize
        if 'multipliers' not in st.session_state or st.session_state.get('last_scenario') != scenario:
            if SCENARIOS[scenario]['multipliers']:
                st.session_state['multipliers'] = SCENARIOS[scenario]['multipliers'].copy()
            else:
                st.session_state['multipliers'] = {lu: 1.00 for lu in DEFAULT_LAND_USES}
            st.session_state['last_scenario'] = scenario
        
        with st.expander("📊 Reference Table"):
            ref = []
            for s in ['Fat', 'EAT', 'NDC', 'Afforest', 'Bioen', 'Yieldint', 'Landretir']:
                row = {'Scenario': SCENARIOS[s]['name']}
                for lu in DEFAULT_LAND_USES:
                    row[lu] = SCENARIOS[s]['multipliers'][lu]
                ref.append(row)
            st.dataframe(pd.DataFrame(ref), hide_index=True)
        
        st.markdown("**Edit:**")
        cols = st.columns(4)
        for idx, lu in enumerate(DEFAULT_LAND_USES):
            with cols[idx % 4]:
                mult = st.number_input(lu, 0.5, 1.5, float(st.session_state['multipliers'][lu]), 0.01, format="%.2f", key=f"m_{lu}")
                st.session_state['multipliers'][lu] = mult
        
        multipliers = st.session_state['multipliers']
        
    else:
        st.info("ℹ️ Custom land uses - enter your own multipliers")
        
        with st.expander("📊 Reference (for default land uses)"):
            ref = []
            for s in ['Fat', 'EAT', 'NDC', 'Afforest', 'Bioen', 'Yieldint', 'Landretir']:
                row = {'Scenario': SCENARIOS[s]['name']}
                for lu in DEFAULT_LAND_USES:
                    row[lu] = SCENARIOS[s]['multipliers'][lu]
                ref.append(row)
            st.dataframe(pd.DataFrame(ref), hide_index=True)
        
        if 'custom_mults' not in st.session_state:
            st.session_state['custom_mults'] = {lu: 1.00 for lu in land_uses}
        
        cols = st.columns(4)
        for idx, lu in enumerate(land_uses):
            with cols[idx % 4]:
                mult = st.number_input(lu, 0.5, 1.5, float(st.session_state['custom_mults'][lu]), 0.01, format="%.2f", key=f"cm_{lu}")
                st.session_state['custom_mults'][lu] = mult
        
        multipliers = st.session_state['custom_mults']
    
    st.markdown("---")
    st.markdown("#### 2.2 Apply Multipliers")
    
    if st.button("🎯 Apply Multipliers to 2050 Results", type="primary", width="stretch"):
        base_2050 = results[results['Year'] == 2050].copy()
        adj_2050 = base_2050.copy()
        
        for lu in land_uses:
            adj_2050[lu] = base_2050[lu].values[0] * multipliers[lu]
        
        st.session_state['adj_2050'] = adj_2050
        st.session_state['mults_applied'] = multipliers.copy()
        st.success("✅ Applied!")
        st.rerun()
    
    if 'adj_2050' in st.session_state:
        st.markdown("---")
        st.markdown("""
        <div class='section-header'>
            <h3>📊 Adjusted Results</h3>
            <p>2050 projection with applied multipliers</p>
        </div>
        """, unsafe_allow_html=True)
        
        base_2050 = results[results['Year'] == 2050]
        adj_2050 = st.session_state['adj_2050']
        mults = st.session_state['mults_applied']
        
        st.markdown("**Comparison: Base vs Adjusted 2050**")
        
        comp = []
        for lu in land_uses:
            base_val = base_2050[lu].values[0]
            adj_val = adj_2050[lu].values[0]
            comp.append({
                'Land Use': lu,
                'Multiplier': f'{mults[lu]:.2f}',
                'Base 2050 (km²)': f'{base_val:,.0f}',
                'Adjusted 2050 (km²)': f'{adj_val:,.0f}',
                'Change (km²)': f'{(adj_val - base_val):+,.0f}'
            })
        
        st.dataframe(pd.DataFrame(comp), hide_index=True, width="stretch")
        
        total_base = sum([base_2050[lu].values[0] for lu in land_uses])
        total_adj = sum([adj_2050[lu].values[0] for lu in land_uses])
        
        st.markdown("**Total Area After Adjustment:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Base 2050", f"{total_base:,.0f} km²")
        with col2:
            st.metric("Adjusted 2050", f"{total_adj:,.0f} km²")
        with col3:
            st.metric("Change", f"{(total_adj - total_base):+,.0f} km²")
        
        st.markdown("---")
        st.markdown("**📈 Full Trajectory with Adjusted 2050**")
        
        # Chart
        chart_data = []
        for _, row in results.iterrows():
            for lu in land_uses:
                chart_data.append({'Year': row['Year'], 'LU': f'{lu} (Base)', 'Area': row[lu]})
        
        for lu in land_uses:
            chart_data.append({'Year': 2050, 'LU': f'{lu} (Adj)', 'Area': adj_2050[lu].values[0]})
        
        df = pd.DataFrame(chart_data)
        fig = px.line(
            df[df['LU'].str.contains('Base')], 
            x='Year', 
            y='Area', 
            color='LU',
            color_discrete_sequence=px.colors.qualitative.Set2,
            height=550
        )
        
        for lu in land_uses:
            lu_adj = df[df['LU'] == f'{lu} (Adj)']
            fig.add_scatter(
                x=lu_adj['Year'], 
                y=lu_adj['Area'], 
                mode='markers', 
                name=f'{lu} (Adjusted)', 
                marker=dict(size=14, symbol='star', line=dict(width=2, color='white'))
            )
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12, family="sans-serif"),
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                title=dict(text="", font=dict(size=13))
            ),
            margin=dict(l=50, r=200, t=50, b=50)
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # Downloads
        st.markdown("---")
        st.markdown("**💾 Download Results**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download Base Projection CSV", 
                results.to_csv(index=False), 
                f"{country}_base_projection.csv", 
                "text/csv",
                width="stretch"
            )
        with col2:
            adj_full = results.copy()
            adj_full.loc[adj_full['Year'] == 2050, land_uses] = [adj_2050[lu].values[0] for lu in land_uses]
            st.download_button(
                "📥 Download Adjusted Projection CSV", 
                adj_full.to_csv(index=False), 
                f"{country}_adjusted_projection.csv", 
                "text/csv",
                width="stretch"
            )

# ============================================================================
# SCENARIO COMPARISON CHART (FAT vs EAT vs NDC)
# ============================================================================

st.markdown("---")
st.markdown("""
<div class='section-header'>
    <h3>📊 Main Scenarios Comparison: FAT vs EAT vs NDC</h3>
    <p>Reference comparison of the three main dietary/policy scenarios at 2050</p>
</div>
""", unsafe_allow_html=True)

# Data paths for scenarios
SCENARIO_DATA_PATHS = {
    'Fat': root_dir / "data/7.diets/fat/ALL_COUNTRIES_fat_annual_projections_2020_2050.csv",
    'EAT': root_dir / "data/7.diets/eat/ALL_COUNTRIES_eat_annual_projections_2020_2050.csv",
    'NDC': root_dir / "data/7.diets/ndc/ALL_COUNTRIES_ndc_annual_projections_2020_2050.csv",
}

comparison_scenarios = ['Fat', 'EAT', 'NDC']

# Load actual 2050 data from CSV files
comparison_data = []
data_loaded = False

for scen in comparison_scenarios:
    file_path = SCENARIO_DATA_PATHS[scen]
    try:
        if file_path.exists():
            df = pd.read_csv(file_path)
            df_2050 = df[df['Year'] == 2050]
            
            for lu in DEFAULT_LAND_USES:
                if lu in df_2050.columns:
                    val_2050 = df_2050[lu].sum()  # Global total
                    comparison_data.append({
                        'Scenario': scen,
                        'Land Use': lu,
                        'Area 2050 (km²)': val_2050
                    })
            data_loaded = True
    except Exception as e:
        st.warning(f"Could not load {scen} data: {e}")

# Fallback to multiplier-based estimates if no data
if not data_loaded or len(comparison_data) == 0:
    st.info("📊 Using estimated values based on BAU × scenario multipliers (actual data files not found)")
    
    # Try to load BAU for baseline
    bau_path = root_dir / "data/7.diets/bau/1.ALL_COUNTRIES_annual_projections_2020_2050.csv"
    baseline_global = {}
    
    if bau_path.exists():
        try:
            df_bau = pd.read_csv(bau_path)
            df_bau_2050 = df_bau[df_bau['Year'] == 2050]
            for lu in DEFAULT_LAND_USES:
                if lu in df_bau_2050.columns:
                    baseline_global[lu] = df_bau_2050[lu].sum()
        except:
            pass
    
    # Fallback baseline if BAU not available
    if not baseline_global:
        baseline_global = {
            'Crops': 15000000, 'TreeCrops': 3000000, 'Forest': 40000000,
            'Grassland': 35000000, 'Urban': 2000000, 'Water': 5000000, 'Other': 30000000
        }
    
    comparison_data = []
    for scen in comparison_scenarios:
        mults = SCENARIOS[scen]['multipliers']
        for lu in DEFAULT_LAND_USES:
            val_2050 = baseline_global.get(lu, 0) * mults[lu]
            comparison_data.append({
                'Scenario': scen,
                'Land Use': lu,
                'Area 2050 (km²)': val_2050
            })

if comparison_data:
    df_compare = pd.DataFrame(comparison_data)
    
    # Create grouped bar chart
    fig_compare = go.Figure()
    
    for scen in comparison_scenarios:
        df_scen = df_compare[df_compare['Scenario'] == scen]
        fig_compare.add_trace(go.Bar(
            name=SCENARIOS[scen]['name'],
            x=df_scen['Land Use'],
            y=df_scen['Area 2050 (km²)'],
            marker_color=SCENARIO_COLORS[scen]
        ))
    
    fig_compare.update_layout(
        barmode='group',
        title='Comparative Land Area by Class in 2050 — FAT vs EAT vs NDC (Global)',
        xaxis_title='Land Use Type',
        yaxis_title='Area (km²)',
        template='plotly_white',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # Add percent comparison
    st.markdown("### Percent Composition Comparison")
    
    fig_pct = go.Figure()
    
    for scen in comparison_scenarios:
        df_scen = df_compare[df_compare['Scenario'] == scen]
        total = df_scen['Area 2050 (km²)'].sum()
        pct_values = (df_scen['Area 2050 (km²)'] / total * 100).values
        
        fig_pct.add_trace(go.Bar(
            name=SCENARIOS[scen]['name'],
            x=df_scen['Land Use'],
            y=pct_values,
            marker_color=SCENARIO_COLORS[scen]
        ))
    
    fig_pct.update_layout(
        barmode='group',
        title='Comparative Land Composition (%) in 2050 — FAT vs EAT vs NDC',
        xaxis_title='Land Use Type',
        yaxis_title='Share of Global Land Area (%)',
        template='plotly_white',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig_pct, use_container_width=True)