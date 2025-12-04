# LandGCH Dashboard

Interactive web application for exploring global land-use projections (2020-2050) based on HILDA+ v2.0 data and Time-varying Markov Chain models.

## Overview

This Streamlit application provides interactive visualization and analysis of land-use change scenarios at the country level. It implements three dietary/policy intervention scenarios compared against a business-as-usual baseline.

## Features

- **Country-level analysis**: Explore projections for 195+ countries
- **Multiple scenarios**: Compare BAU, Fat, EAT-Lancet, and NDC/Afforestation scenarios
- **Interactive visualizations**: Plotly-based charts with filtering and comparison tools
- **Temporal analysis**: Annual projections from 2020 to 2050

## Installation

### Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/landgch-dashboard.git
cd landgch-dashboard

# Install dependencies
pip install -r requirements.txt

# Configure data paths
# Edit DATA_PATHS dictionary in landgch_app.py (line ~100)

# Run application
streamlit run landgch_app.py
```

Application will be available at `http://localhost:8501`

## Data Requirements

The application expects four CSV files containing annual land-use projections:

```
- ALL_COUNTRIES_annual_projections_2020_2050.csv (BAU baseline)
- ALL_COUNTRIES_fat_annual_projections_2020_2050.csv
- ALL_COUNTRIES_eat_annual_projections_2020_2050.csv
- ALL_COUNTRIES_ndc_annual_projections_2020_2050.csv
```

### Required CSV Format

Each file must contain the following columns:
- `Country`: ISO3 country code
- `Year`: Integer (2020-2050)
- `Crops`: Area in km²
- `TreeCrops`: Area in km²
- `Forest`: Area in km²
- `Grassland`: Area in km²
- `Urban`: Area in km²
- `Water`: Area in km²
- `Other`: Area in km²

## Scenarios

| Scenario | Description | Key Parameters |
|----------|-------------|----------------|
| **BAU** | Business As Usual | Historical trend continuation |
| **Fat** | High Meat Diet | Grassland +12%, Forest -3%, Crops +6% |
| **EAT** | EAT-Lancet Diet | Crops -10%, Forest +8%, Grassland -18% |
| **NDC** | Afforestation & Biofuels | Forest +7%, TreeCrops +5%, Crops +3% |

## Application Structure

### Pages

1. **Home**: Model overview, scenario descriptions, global statistics
2. **Country Explorer**: Single-country analysis with scenario comparison
3. **Scenario Comparison**: Multi-country comparative analysis

### Visualization Types

- Time series line charts
- Stacked area charts
- Grouped bar charts
- Summary statistics tables

## Methodology

The projections are based on:

1. **Historical Data**: HILDA+ v2.0 (1960-2020) at 1km² resolution
2. **Forecasting**: Time-varying Markov Chain models
3. **Validation**: Multi-period validation (10/20/30 year windows)
4. **Scenario Application**: Multipliers applied to baseline projections with country-specific constraints

## Development

### Project Structure

```
landgch-dashboard/
├── landgch_app.py          # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore patterns
├── LICENSE                # MIT License
└── README.md              # This file
```

### Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this tool in research, please cite:

```bibtex
@software{landgch_dashboard,
  title = {LandGCH Dashboard: Interactive Land-Use Projection Tool},
  author = {[Author Names]},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/landgch-dashboard}
}
```

## Acknowledgments

- HILDA+ v2.0 for historical land-use data
- Natural Earth for country boundary data
- Streamlit for the application framework
- Plotly for visualization tools

## Contact

[Your contact information]

## Related Publications

[List any related papers or documentation]
