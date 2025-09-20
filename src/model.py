import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- Ensemble Learning & Model Optimization Functions ---

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier.
    
    TODO: 
        - Initialize and train a RandomForestClassifier using the training data.
        - Ensure you set the random_state for reproducibility.
    """
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    """
    Train a GradientBoostingClassifier.
    
    TODO: 
        - Initialize and train a GradientBoostingClassifier using the training data.
        - Ensure you set the random_state for reproducibility.
    """
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    return gb_model

def apply_k_fold_cross_validation(model, X_train: pd.DataFrame, y_train: pd.Series, cv_splits: int = 5) -> float:
    """
    Apply K-fold cross-validation and return the average accuracy.
    
    TODO: 
        - Implement K-fold cross-validation for model validation.
        - Ensure you shuffle the dataset and set the random_state for reproducibility.
        - Use `cross_val_score()` to compute accuracy scores.
    """
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    return cv_scores.mean()

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Evaluate the model using accuracy, precision, recall, and confusion matrix.
    
    TODO: 
        - Predict values using the trained model with `model.predict()`.
        - Calculate accuracy, precision, recall, and confusion matrix.
        - Ensure you use `precision_score` and `recall_score` with the `average='weighted'` option.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, cm

# --- Feature Selection, Handling Missing Values, Encoding Categorical Features, and Data Splitting ---

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant features for the analysis.
        
    TODO: 
        - Modify this function to select only the relevant columns for your analysis.
        - Make sure the features are related to student dropout prediction.
    """
    features = ['Age at enrollment', 'Gender', 'Debtor', 'Tuition fees up to date', 'Curricular units 1st sem (approved)']
    X = df[features]
    return X

def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        X (pd.DataFrame): The input feature matrix.
        
    Returns:
        pd.DataFrame: The feature matrix with missing values handled.
        
    TODO:
        - Fill missing values in categorical columns with 'Unknown'.
        - Fill missing values in numerical columns with the median of that column.
    """
    X = X.copy()

    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    X.loc[:, categorical_cols] = X[categorical_cols].fillna('Unknown')

    numerical_cols = ['Age at enrollment', 'Curricular units 1st sem (approved)']
    X.loc[:, numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())

    return X

def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features into numeric format using one-hot encoding.
        
    TODO:
        - Apply one-hot encoding to the categorical columns like 'Gender', 'Debtor', and 'Tuition fees up to date'.
        - Use `pd.get_dummies()` to perform one-hot encoding and drop the first column to avoid the dummy variable trap.
    """
    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    return X

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> tuple:
    """
    Split the dataset into training and testing sets.
        
    TODO:
        - Use `train_test_split()` to split the data into training and testing sets.
        - Set the random_state for reproducibility.
        - Ensure the test size is appropriate (usually 30% for testing and 70% for training).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
    
# --- Main Execution Block ---
if __name__ == '__main__':
    # Load the dataset 
    file_path = 'dataset.csv'
    df = pd.read_csv(file_path)

    # Step 1: Select relevant features for the analysis
    features = select_features(df)

    # Step 2: Handle missing values
    features = features.copy()
    categorical_cols = ['Gender', 'Debtor', 'Tuition fees up to date']
    features.loc[:, categorical_cols] = features[categorical_cols].fillna('Unknown')
    numerical_cols = ['Age at enrollment', 'Curricular units 1st sem (approved)']
    features.loc[:, numerical_cols] = features[numerical_cols].fillna(features[numerical_cols].median())

    # Step 3: One-hot encode categorical columns
    features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

    # Step 4: Set the target variable 'Target'
    y = df['Target']  

    # Step 5: Split data into training and testing sets (70-30 split)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)

    # Step 6: Train and evaluate Random Forest model
    rf_model = train_random_forest(X_train, y_train)

    # Apply K-fold cross-validation for Random Forest
    rf_cv_score = apply_k_fold_cross_validation(rf_model, X_train, y_train)
    print("Random Forest Cross-validation Accuracy:", rf_cv_score)

    # Evaluate Random Forest model
    rf_accuracy, rf_precision, rf_recall, rf_cm = evaluate_model(rf_model, X_test, y_test)
    print("\nRandom Forest Model Accuracy:", rf_accuracy)
    print("Random Forest Model Precision:", rf_precision)
    print("Random Forest Model Recall:", rf_recall)
    print("Random Forest Confusion Matrix:\n", rf_cm)

    # Step 7: Train and evaluate Gradient Boosting model
    gb_model = train_gradient_boosting(X_train, y_train)

    # Apply K-fold cross-validation for Gradient Boosting
    gb_cv_score = apply_k_fold_cross_validation(gb_model, X_train, y_train)
    print("Gradient Boosting Cross-validation Accuracy:", gb_cv_score)

    # Evaluate Gradient Boosting model
    gb_accuracy, gb_precision, gb_recall, gb_cm = evaluate_model(gb_model, X_test, y_test)
    print("\nGradient Boosting Model Accuracy:", gb_accuracy)
    print("Gradient Boosting Model Precision:", gb_precision)
    print("Gradient Boosting Model Recall:", gb_recall)
    print("Gradient Boosting Confusion Matrix:\n", gb_cm)
