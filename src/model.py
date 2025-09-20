import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def explore_data(df: pd.DataFrame):
    """
    Perform basic data exploration, printing key details about the dataset.

    Args:
        df (pd.DataFrame): The DataFrame to explore.
    """
    print("Dataset Shape: ", df.shape)
    print("\nData Types: ")
    print(df.dtypes)
    print("\nMissing Values: ")
    print(df.isnull().sum())
    
    # Corrected column names based on your dataset
    print("\nGender Distribution: ")
    print(df['Gender'].value_counts())
    
    print("\nCourse Distribution: ")
    print(df['Course'].value_counts())
    
    print("\nTuition Fees Up to Date Distribution: ")
    print(df['Tuition fees up to date'].value_counts())
    
    print("\nDebtor Distribution: ")
    print(df['Debtor'].value_counts())


def create_approval_rate(df: pd.DataFrame):
    """
    Create the 'approval_rate' feature based on 'Curricular units 1st sem (approved)' and 'Curricular units 1st sem (enrolled)' columns.

    Args:
        df (pd.DataFrame): The DataFrame with student data.

    Returns:
        pd.DataFrame: DataFrame with the 'approval_rate' column.
    """
    df['approval_rate'] = df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (enrolled)']
    return df


def create_performance_score(df: pd.DataFrame):
    """
    Create the 'performance_score' feature based on 'Curricular units 1st sem (approved)' and 'Curricular units 1st sem (evaluations)' columns.

    Args:
        df (pd.DataFrame): The DataFrame with student data.

    Returns:
        pd.DataFrame: DataFrame with the 'performance_score' column.
    """
    df['performance_score'] = df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (evaluations)']
    return df


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features needed for the analysis.

    Args:
        df (pd.DataFrame): The DataFrame with student data.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # TODO: Add engineered features by calling the functions to create 'approval_rate' and 'performance_score'
    df = create_approval_rate(df)
    df = create_performance_score(df)
    
    # TODO: Print the first few rows of the DataFrame to verify that the new columns are correctly added
    print("\nData with new columns: ")
    print(df[['Course', 'Gender', 'approval_rate', 'performance_score']].head())
    
    return df


# --- Main Execution Block ---
if __name__ == '__main__':
    # Path to the dataset (make sure it's correct)
    file_path = 'dataset.csv'
    
    # Step 1: Load the dataset
    df = load_data(file_path)
    
    # Step 2: Explore the dataset to understand its structure and distribution
    explore_data(df)
    
    # Step 3: Create engineered features for the model
    df = create_engineered_features(df)
