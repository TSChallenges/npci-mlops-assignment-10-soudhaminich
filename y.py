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

    # Load the dataset
    churn_df = pd.read_csv(input_churn_dataset.path)

    # Features and target variable
    X = churn_df.drop(columns=["Exited"])
    y = churn_df["Exited"]

    # Split the data
    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Save the datasets as CSV
    with open(X_train.path, 'w') as f:
        X_train_data.to_csv(f, index=False)
    with open(X_test.path, 'w') as f:
        X_test_data.to_csv(f, index=False)
    with open(y_train.path, 'w') as f:
        y_train_data.to_csv(f, index=False)
    with open(y_test.path, 'w') as f:
        y_test_data.to_csv(f, index=False)


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

    # Load the training data
    X_train_data = pd.read_csv(X_train.path)
    y_train_data = pd.read_csv(y_train.path)

    # Train the model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_data, y_train_data)

    # Save the trained model
    with open(model_output.path, 'wb') as f:
        pickle.dump(model, f)


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

    # Load the test data
    X_test_data = pd.read_csv(X_test.path)
    y_test_data = pd.read_csv(y_test.path)

    # Load the trained model
    with open(model_path.path, 'rb') as f:
        model = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X_test_data)

    # Calculate metrics
    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred)
    recall = recall_score(y_test_data, y_pred)
    f1 = f1_score(y_test_data, y_pred)

    # Save the metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    metrics_df = pd.DataFrame([metrics])

    with open(metrics_output.path, 'w') as file:
        metrics_df.to_csv(file, index=False)


# Pipeline Definition
@pipeline(name="customer-churn-pipeline", description="Pipeline for Customer Churn Prediction")
def customer_churn_pipeline(
    drop_missing_vals: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
):
    churn_data = load_churn_data(drop_missing_vals=drop_missing_vals)
    X_train, X_test, y_train, y_test = train_test_split_churn(
        input_churn_dataset=churn_data,
        test_size=test_size,
        random_state=random_state
    )
    model = train_churn_model(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        random_state=random_state
    )
    evaluate_churn_model(
        X_test=X_test,
        y_test=y_test,
        model_path=model,
    )


# Compile Pipeline
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(customer_churn_pipeline, 'customer_churn_pipeline_v1.yaml')
