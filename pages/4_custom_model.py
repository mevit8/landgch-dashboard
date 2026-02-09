"""
Custom Model Builder - Two-Stage System
Stage 1: Base Projection (with transition matrix)
Stage 2: Scenario Adjustment (with multipliers)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Import from core module
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
# SCENARIO DEFINITIONS
# ============================================================================

SCENARIOS = {
    'Custom': {
        'name': 'Custom Multipliers',
        'description': 'Enter your own multipliers',
        'multipliers': None
    },
    'Fat': {
        'name': 'High-meat ("Fat" Diet)',
        'description': 'Rich in meat - increased area to pasture and feed crops',
        'multipliers': {'Crops': 1.06, 'TreeCrops': 1.02, 'Forest': 0.97, 'Grassland': 1.12, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95}
    },
    'EAT': {
        'name': 'The Lancet EAT Diet',
        'description': 'Healthy diet - increased vegetables/legumes, reduced pasture',
        'multipliers': {'Crops': 0.90, 'TreeCrops': 0.97, 'Forest': 1.08, 'Grassland': 0.82, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.98}
    },
    'NDC': {
        'name': 'National Determined Contributions',
        'description': 'Forest gain targets, bioenergy crops',
        'multipliers': {'Crops': 1.03, 'TreeCrops': 1.05, 'Forest': 1.07, 'Grassland': 0.98, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95}
    },
    'Afforest': {
        'name': 'Afforestation/Reforestation',
        'description': 'Forest protection policies',
        'multipliers': {'Crops': 1.01, 'TreeCrops': 1.02, 'Forest': 1.07, 'Grassland': 0.99, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95}
    },
    'Bioen': {
        'name': 'Bioenergy Expansion',
        'description': 'Increased biofuel production',
        'multipliers': {'Crops': 1.03, 'TreeCrops': 1.05, 'Forest': 1.03, 'Grassland': 0.97, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.95}
    },
    'Yieldint': {
        'name': 'Yield Intensification',
        'description': 'Higher yields, less land needed',
        'multipliers': {'Crops': 0.90, 'TreeCrops': 0.97, 'Forest': 1.03, 'Grassland': 0.83, 'Urban': 1.00, 'Water': 1.00, 'Other': 0.98}
    },
    'Landretir': {
        'name': 'Land Retirement',
        'description': 'Marginal cropland to forest',
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
    layout="wide"
)

st.title("🔬 Custom Model Builder")

st.markdown("""
**Two-Stage Process:**
1. **Stage 1:** Run base projection with transition matrix → Get 2050 results
2. **Stage 2:** Apply scenario multipliers to adjust 2050 results

---
""")

# ============================================================================
# STAGE 1: BASE PROJECTION
# ============================================================================

st.header("📊 Stage 1: Base Projection")

# ----------------------------------------------------------------------------
# 1.1 Country Selection & Baseline Loading
# ----------------------------------------------------------------------------

st.subheader("1.1 Country & Baseline Data")

country = st.text_input(
    "Country Code (ISO3)",
    value="BRA",
    max_chars=3,
    help="3-letter country code"
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
                
                st.success(f"✅ Loaded baseline for {country} (2020) - Total: {total_area:,.0f} km²")
                
                with st.expander("📊 View 2020 Baseline Data"):
                    baseline_df = pd.DataFrame([baseline_2020])
                    st.dataframe(baseline_df.T.rename(columns={0: 'Area (km²)'}))
                
                baseline_loaded = True
                break
            else:
                st.error(f"❌ No data found for country {country} in year 2020")
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

st.divider()

# ----------------------------------------------------------------------------
# 1.2 Land Use Configuration
# ----------------------------------------------------------------------------

st.subheader("1.2 Land Use Configuration")

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
    
    # Initialize in session state if not present
    if 'custom_num_land_uses' not in st.session_state:
        st.session_state['custom_num_land_uses'] = 4
    if 'custom_land_use_names' not in st.session_state:
        st.session_state['custom_land_use_names'] = []
    if 'custom_baseline_areas' not in st.session_state:
        st.session_state['custom_baseline_areas'] = []
    
    num_land_uses = st.number_input(
        "Number of categories",
        min_value=2,
        max_value=15,
        value=st.session_state['custom_num_land_uses']
    )
    
    # Temporary inputs (not confirmed yet)
    temp_land_uses = []
    temp_baseline_areas = []
    
    st.markdown("**Enter names and baseline areas (2020):**")
    
    cols = st.columns(3)
    for i in range(num_land_uses):
        with cols[i % 3]:
            # Get default value from session state if available
            default_name = st.session_state['custom_land_use_names'][i] if i < len(st.session_state['custom_land_use_names']) else f"LandUse_{i+1}"
            default_area = st.session_state['custom_baseline_areas'][i] if i < len(st.session_state['custom_baseline_areas']) else 1000000.0
            
            name = st.text_input(f"Land Use {i+1} Name", value=default_name, key=f"custom_lu_name_{i}")
            area = st.number_input(f"Baseline (km²)", min_value=0.0, value=default_area, step=100000.0, format="%.0f", key=f"custom_lu_area_{i}")
            temp_land_uses.append(name)
            temp_baseline_areas.append(area)
    
    # Confirm button
    st.markdown("---")
    if st.button("✓ Confirm Land Uses", type="primary", use_container_width=False):
        st.session_state['custom_num_land_uses'] = num_land_uses
        st.session_state['custom_land_use_names'] = temp_land_uses
        st.session_state['custom_baseline_areas'] = temp_baseline_areas
        st.success("✅ Land uses confirmed!")
        st.rerun()
    
    st.caption("💡 Click 'Confirm' after entering/changing land use names to update")
    
    # Use confirmed values from session state
    land_uses = st.session_state['custom_land_use_names'] if st.session_state['custom_land_use_names'] else temp_land_uses
    baseline_areas_list = st.session_state['custom_baseline_areas'] if st.session_state['custom_baseline_areas'] else temp_baseline_areas
    
    baseline_areas = np.array(baseline_areas_list)
    if len(baseline_areas) > 0:
        st.metric("Total Baseline Area", f"{baseline_areas.sum():,.0f} km²")

st.divider()

# ----------------------------------------------------------------------------
# 1.3 Transition Matrix
# ----------------------------------------------------------------------------

st.subheader("1.3 Transition Matrix")

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
            
            styled = matrix_df.style.applymap(highlight_sums, subset=['Row Sum']).format('{:.4f}')
            st.dataframe(styled, use_container_width=True)
            
            valid, errors = validate_transition_matrix(matrix)
            if valid:
                st.success("✅ Valid!")
            else:
                for error in errors:
                    st.error(f"❌ {error}")

st.divider()

# ----------------------------------------------------------------------------
# 1.4 Run Base Projection
# ----------------------------------------------------------------------------

st.subheader("1.4 Run Base Projection")

can_run = baseline_areas is not None and matrix is not None and len(land_uses) > 0 and matrix.shape[0] == len(land_uses)

if can_run:
    valid_matrix, _ = validate_transition_matrix(matrix)
    
    if valid_matrix:
        if st.button("🚀 Run Stage 1", type="primary", use_container_width=True):
            with st.spinner("Running..."):
                try:
                    results = run_custom_projection(country, matrix, land_uses, baseline_areas)
                    
                    st.session_state['base_results'] = results
                    st.session_state['land_uses'] = land_uses
                    st.session_state['use_default_types'] = use_default_types
                    
                    st.success("✅ Done!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# Show Base Results
if 'base_results' in st.session_state:
    st.divider()
    st.subheader("📊 Base Projection Results")
    
    results = st.session_state['base_results']
    land_uses = st.session_state['land_uses']
    
    total_2020 = results[results['Year'] == 2020][land_uses].values[0].sum()
    total_2050 = results[results['Year'] == 2050][land_uses].values[0].sum()
    difference = total_2050 - total_2020
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("2020", f"{total_2020:,.0f} km²")
    with col2:
        st.metric("2050", f"{total_2050:,.0f} km²")
    with col3:
        st.metric("Diff", f"{difference:+,.0f} km²")
    
    if abs(difference) < 1:
        st.success("✅ Area conserved")
    elif difference > 0:
        st.warning(f"⚠️ Exceeded by {abs(difference):,.0f} km². Reconsider smaller change percentages.")
    else:
        st.warning(f"⚠️ Used less by {abs(difference):,.0f} km²")
    
    plot_data = results.melt(id_vars=['Country', 'Year'], var_name='Land_Use', value_name='Area')
    fig = px.line(plot_data, x='Year', y='Area', color='Land_Use', height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# STAGE 2: SCENARIO ADJUSTMENT
# ============================================================================

if 'base_results' in st.session_state:
    st.divider()
    st.header("🎯 Stage 2: Scenario Adjustment")
    
    results = st.session_state['base_results']
    land_uses = st.session_state['land_uses']
    use_default_types = st.session_state['use_default_types']
    
    st.markdown("`2050 Adjusted = 2050 Base × Multipliers`")
    
    st.subheader("2.1 Multipliers")
    
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
    
    st.divider()
    st.subheader("2.2 Apply")
    
    if st.button("🎯 Apply to 2050", type="primary", use_container_width=True):
        base_2050 = results[results['Year'] == 2050].copy()
        adj_2050 = base_2050.copy()
        
        for lu in land_uses:
            adj_2050[lu] = base_2050[lu].values[0] * multipliers[lu]
        
        st.session_state['adj_2050'] = adj_2050
        st.session_state['mults_applied'] = multipliers.copy()
        st.success("✅ Applied!")
        st.rerun()
    
    if 'adj_2050' in st.session_state:
        st.divider()
        st.subheader("📊 Adjusted Results")
        
        base_2050 = results[results['Year'] == 2050]
        adj_2050 = st.session_state['adj_2050']
        mults = st.session_state['mults_applied']
        
        comp = []
        for lu in land_uses:
            base_val = base_2050[lu].values[0]
            adj_val = adj_2050[lu].values[0]
            comp.append({
                'Land Use': lu,
                'Mult': f'{mults[lu]:.2f}',
                'Base 2050': f'{base_val:,.0f}',
                'Adjusted 2050': f'{adj_val:,.0f}',
                'Change': f'{(adj_val - base_val):+,.0f}'
            })
        
        st.dataframe(pd.DataFrame(comp), hide_index=True)
        
        total_base = sum([base_2050[lu].values[0] for lu in land_uses])
        total_adj = sum([adj_2050[lu].values[0] for lu in land_uses])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Base", f"{total_base:,.0f} km²")
        with col2:
            st.metric("Adjusted", f"{total_adj:,.0f} km²")
        with col3:
            st.metric("Δ", f"{(total_adj - total_base):+,.0f} km²")
        
        # Chart
        chart_data = []
        for _, row in results.iterrows():
            for lu in land_uses:
                chart_data.append({'Year': row['Year'], 'LU': f'{lu} (Base)', 'Area': row[lu]})
        
        for lu in land_uses:
            chart_data.append({'Year': 2050, 'LU': f'{lu} (Adj)', 'Area': adj_2050[lu].values[0]})
        
        df = pd.DataFrame(chart_data)
        fig = px.line(df[df['LU'].str.contains('Base')], x='Year', y='Area', color='LU', height=500)
        
        for lu in land_uses:
            lu_adj = df[df['LU'] == f'{lu} (Adj)']
            fig.add_scatter(x=lu_adj['Year'], y=lu_adj['Area'], mode='markers', name=f'{lu} (Adj)', marker=dict(size=12, symbol='star'))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Downloads
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Base CSV", results.to_csv(index=False), f"{country}_base.csv", "text/csv")
        with col2:
            adj_full = results.copy()
            adj_full.loc[adj_full['Year'] == 2050, land_uses] = [adj_2050[lu].values[0] for lu in land_uses]
            st.download_button("📥 Adjusted CSV", adj_full.to_csv(index=False), f"{country}_adjusted.csv", "text/csv")
