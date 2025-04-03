from kfp.dsl import component, pipeline, Input, Output, Dataset, Model
import kfp

# Pipeline Component-1: Data Loading & Processing
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def load_churn_data(drop_missing_vals: bool, churn_dataset: Output[Dataset]):
    import pandas as pd
    # Load Customer Churn dataset
    df = pd.read_csv("https://raw.githubusercontent.com/MLOPS-test/test-scripts/refs/heads/main/mlops-ast10/Churn_Modeling.csv")

    # Define target variable and features
    X = df[["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary"]].copy()
    y = df[["Exited"]]

    # Handling category labels present in `Geography` and `Gender` columns
    # Create dictionaries to map categorical values to numberic labels. OR Use LabelEncoder
    geography_mapping = {'France': 0, 'Spain': 1, 'Germany': 2}
    gender_mapping = {'Female': 0, 'Male': 1}

    # Map categorical values to numbers using respective dictionaries
    X['Geography'] = X['Geography'].map(geography_mapping)
    X['Gender'] = X['Gender'].map(gender_mapping)

    transformed_df = X.copy()
    transformed_df['Exited'] = y

    if drop_missing_vals:
        transformed_df = transformed_df.dropna()

    with open(churn_dataset.path, 'w') as file:
        transformed_df.to_csv(file, index=False)


# Pipeline Component-2: Train-Test Split
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def train_test_split_churn(
    input_churn_dataset: Input[Dataset],
    X_train: Output[Dataset],
    X_test: Output[Dataset],
    y_train: Output[Dataset],
    y_test: Output[Dataset],
    test_size: float,
    random_state: int,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # YOUR CODE HERE to split the dataset into training & testing set and save them as CSV files



# Pipeline Component-3: Model Training
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def train_churn_model(
    X_train: Input[Dataset],
    y_train: Input[Dataset],
    model_output: Output[Model],
    n_estimators: int,
    random_state: int,
):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import pickle

    # YOUR CODE HERE to load train the model on training set, and save the model



# Pipeline Component-4: Model Evaluation
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def evaluate_churn_model(
    X_test: Input[Dataset],
    y_test: Input[Dataset],
    model_path: Input[Model],
    metrics_output: Output[Dataset]
):
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pickle

    # YOUR CODE HERE to check the model performance using different metrics and save them



# Pipeline Definition
@pipeline(name="customer-churn-pipeline", description="Pipeline for Customer Churn Prediction")
def customer_churn_pipeline(
    drop_missing_vals: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
):
    # YOUR CODE HERE to connect the pipeline components and direct their inputs and outputs



# Compile Pipeline
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(customer_churn_pipeline, 'customer_churn_pipeline_v1.yaml')
