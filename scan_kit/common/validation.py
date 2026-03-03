"""Data validation utilities for scan-kit."""

import pandas as pd


def create_valid_mask(df, exclude_values=(-1, -10000)):
    """Create a boolean mask for valid rows.

    Excludes rows where any value is in exclude_values or NaN.

    Args:
        df: DataFrame to validate.
        exclude_values: Tuple of values to exclude. Default (-1, -10000).

    Returns:
        Boolean Series where True indicates valid rows.
    """
    mask = ~df.isna().any(axis=1)
    for val in exclude_values:
        mask = mask & ~(df == val).any(axis=1)
    return mask


def apply_validation(df, exclude_values=(-1, -10000)):
    """Filter DataFrame to valid rows only.

    Args:
        df: DataFrame to validate.
        exclude_values: Tuple of values to exclude. Default (-1, -10000).

    Returns:
        Filtered DataFrame (may be empty).
    """
    mask = create_valid_mask(df, exclude_values)
    return df[mask].copy()
