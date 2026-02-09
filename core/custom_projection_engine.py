"""
Custom Land-Use Projection Engine
Accepts user-provided N×N transition matrix and runs Markov projection
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

def validate_transition_matrix(matrix: np.ndarray) -> Tuple[bool, List[str]]:
    """
    Validate transition probability matrix
    
    Args:
        matrix: N×N numpy array
    
    Returns:
        (is_valid, list_of_error_messages)
    """
    errors = []
    
    # Check if square
    if len(matrix.shape) != 2:
        errors.append("Matrix must be 2-dimensional")
        return False, errors
    
    if matrix.shape[0] != matrix.shape[1]:
        errors.append(f"Matrix must be square, got {matrix.shape}")
    
    # Check value range [0, 1]
    if np.any(matrix < 0):
        errors.append("All values must be non-negative (≥ 0)")
    
    if np.any(matrix > 1):
        errors.append("All values must be ≤ 1")
    
    # Check rows sum to 1.0
    row_sums = matrix.sum(axis=1)
    tolerance = 0.01
    bad_rows = []
    
    for i, row_sum in enumerate(row_sums):
        if abs(row_sum - 1.0) > tolerance:
            bad_rows.append(f"Row {i+1}: sums to {row_sum:.4f}")
    
    if bad_rows:
        errors.append(f"Rows must sum to 1.0 (±{tolerance}). Issues: " + ", ".join(bad_rows))
    
    return len(errors) == 0, errors


def project_markov(
    initial_state: np.ndarray,
    transition_matrix: np.ndarray,
    n_years: int = 30
) -> np.ndarray:
    """
    Run deterministic Markov projection
    
    Args:
        initial_state: 1D array of N land-use areas (km²)
        transition_matrix: N×N transition probability matrix
        n_years: Number of years to project (default 30: 2020→2050)
    
    Returns:
        trajectory: (n_years+1) × N array, rows are years
    """
    n = len(initial_state)
    trajectory = np.zeros((n_years + 1, n))
    trajectory[0, :] = initial_state.copy()
    
    state = initial_state.copy()
    
    for t in range(n_years):
        # Markov step: x[t+1] = x[t] · P
        state = state @ transition_matrix
        trajectory[t + 1, :] = state
    
    return trajectory


def run_custom_projection(
    country: str,
    transition_matrix: np.ndarray,
    land_use_names: List[str],
    baseline_2020: np.ndarray,
    start_year: int = 2020,
    n_years: int = 30
) -> pd.DataFrame:
    """
    Complete custom projection workflow
    
    Args:
        country: ISO3 country code (e.g., 'BRA')
        transition_matrix: N×N probability matrix
        land_use_names: List of N land-use category names
        baseline_2020: N baseline areas in km² for start year
        start_year: Starting year (default 2020)
        n_years: Years to project (default 30)
    
    Returns:
        DataFrame with columns: Country, Year, [land_use_names...]
    
    Raises:
        ValueError: If validation fails
    """
    # Validate matrix
    valid, errors = validate_transition_matrix(transition_matrix)
    if not valid:
        raise ValueError(f"Invalid transition matrix: {'; '.join(errors)}")
    
    # Check dimensions match
    n = transition_matrix.shape[0]
    
    if len(land_use_names) != n:
        raise ValueError(f"Need {n} land-use names, got {len(land_use_names)}")
    
    if len(baseline_2020) != n:
        raise ValueError(f"Need {n} baseline values, got {len(baseline_2020)}")
    
    # Check baseline values
    if np.any(baseline_2020 < 0):
        raise ValueError("All baseline values must be non-negative")
    
    if baseline_2020.sum() == 0:
        raise ValueError("Baseline cannot be all zeros")
    
    # Run projection
    trajectory = project_markov(baseline_2020, transition_matrix, n_years)
    
    # Format as DataFrame
    df = pd.DataFrame(trajectory, columns=land_use_names)
    
    # Add metadata columns
    years = list(range(start_year, start_year + n_years + 1))
    df.insert(0, 'Year', years)
    df.insert(0, 'Country', country)
    
    return df


def parse_matrix_from_csv_string(csv_string: str) -> np.ndarray:
    """
    Parse transition matrix from CSV string (uploaded file content)
    
    Args:
        csv_string: String content of CSV file
    
    Returns:
        N×N numpy array
    """
    import io
    
    # Try to read as CSV
    try:
        df = pd.read_csv(io.StringIO(csv_string), index_col=0)
        matrix = df.values.astype(float)
        return matrix
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")


def parse_matrix_from_text(text: str) -> np.ndarray:
    """
    Parse transition matrix from pasted text (tab or comma separated)
    
    Args:
        text: Pasted text from Excel or other source
    
    Returns:
        N×N numpy array
    """
    lines = text.strip().split('\n')
    rows = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try tab-separated first, then comma
        if '\t' in line:
            values = line.split('\t')
        elif ',' in line:
            values = line.split(',')
        else:
            values = line.split()
        
        # Convert to float, skip empty and non-numeric
        row = []
        for v in values:
            v = v.strip()
            if not v:
                continue
            try:
                row.append(float(v))
            except ValueError:
                # Skip non-numeric (headers or row labels)
                continue
        
        # Only add rows with numeric data
        if row:
            rows.append(row)
    
    if not rows:
        raise ValueError("No numeric data found in text")
    
    # Check all rows same length
    row_lengths = [len(r) for r in rows]
    if len(set(row_lengths)) > 1:
        raise ValueError(f"Inconsistent row lengths: {row_lengths}")
    
    matrix = np.array(rows)
    
    return matrix


if __name__ == "__main__":
    # Test with simple 3×3 matrix
    print("Testing custom projection engine...")
    
    # Create test matrix
    test_matrix = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.85, 0.05],
        [0.05, 0.05, 0.9]
    ])
    
    test_names = ['Crops', 'Forest', 'Urban']
    test_baseline = np.array([100000, 200000, 50000])
    
    print("\nTest matrix:")
    print(test_matrix)
    print(f"\nRow sums: {test_matrix.sum(axis=1)}")
    
    # Validate
    valid, errors = validate_transition_matrix(test_matrix)
    print(f"\nValidation: {'✓ Valid' if valid else '✗ Invalid'}")
    if errors:
        for err in errors:
            print(f"  - {err}")
    
    # Run projection
    if valid:
        results = run_custom_projection(
            country='TEST',
            transition_matrix=test_matrix,
            land_use_names=test_names,
            baseline_2020=test_baseline
        )
        
        print(f"\nProjection complete!")
        print(f"Shape: {results.shape}")
        print(f"\nFirst 5 years:")
        print(results.head())
        print(f"\nLast 5 years:")
        print(results.tail())
        
        print(f"\nTotal area conservation check:")
        print(f"2020: {results.iloc[0, 2:].sum():,.0f} km²")
        print(f"2050: {results.iloc[-1, 2:].sum():,.0f} km²")
