import pandas as pd


def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Perform basic analysis of a dataset.

    Returns:
        dict: Dataset structure and data-quality information.
    """

    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    missing_values = {
        column: int(count)
        for column, count in df.isnull().sum().items()
        if count > 0
    }

    missing_percentage = {
        column: round((count / len(df)) * 100, 2)
        for column, count in df.isnull().sum().items()
        if count > 0
    }

    unique_counts = {
        column: int(count)
        for column, count in df.nunique(dropna=False).items()
    }

    constant_columns = [
        column
        for column, count in unique_counts.items()
        if count <= 1
    ]

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),

        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },

        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,

        "missing_values": missing_values,
        "missing_percentage": missing_percentage,

        "duplicate_rows": int(df.duplicated().sum()),

        "unique_counts": unique_counts,

        "constant_columns": constant_columns,
    }

def analyze_target(
    df: pd.DataFrame,
    target_column: str
) -> dict:
    """
    Analyze the user-selected target column.

    The target is chosen by the user. This function only validates
    and analyzes the selected target.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist in the dataset."
        )

    target = df[target_column]

    missing_count = int(target.isnull().sum())
    non_missing_count = int(target.notna().sum())
    unique_count = int(target.nunique(dropna=True))

    if non_missing_count == 0:
        problem_type = "undetermined"
    else:
        unique_ratio = unique_count / non_missing_count

        if not pd.api.types.is_numeric_dtype(target):
            problem_type = "classification"

        elif pd.api.types.is_integer_dtype(target):
            if unique_count <= 20 or unique_ratio <= 0.05:
                problem_type = "classification"
            else:
                problem_type = "regression"

        else:
            if unique_ratio <= 0.05:
                problem_type = "classification"
            else:
                problem_type = "regression"

    return {
        "target_column": target_column,
        "data_type": str(target.dtype),
        "unique_values": unique_count,
        "missing_values": missing_count,
        "missing_percentage": round(
            (missing_count / len(df)) * 100, 2
        ),
        "problem_type": problem_type,
    }