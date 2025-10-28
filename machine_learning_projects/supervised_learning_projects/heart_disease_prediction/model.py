import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
import os


# --- Configuration --- #
DATASET_PATH = "fedesoriano/heart-failure-prediction"
DATA_FILE = "/heart.csv"
TEST_SIZE = 0.1
RANDOM_STATE = 42
MAX_ITER = 1000
OUTPUT_DIR = "./heart_disease_prediction/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Functions --- #

def load_data(dataset_path: str, data_file: str) -> pd.DataFrame:
    print("Loading dataset...")
    path = kagglehub.dataset_download(dataset_path)
    data = pd.read_csv(path + data_file)
    print(f"Dataset loaded. Shape: {data.shape}")
    return data


def explore_data(data: pd.DataFrame):
    print("\nData Exploration:")
    print(f"Data size: {data.shape}")
    print("First 5 rows:\n", data.head())
    print("Columns:", data.columns.tolist())


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    print("\nEncoding categorical columns using one-hot encoding...")
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    print(f"Columns after encoding: {data.columns.tolist()}")
    return data


def visualize_target_distribution(data: pd.DataFrame):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='HeartDisease', data=data)
    plt.title("Distribution of Heart Disease")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'target_distribution.png'), dpi=300)
    plt.close()


def visualize_correlation_heatmap(data: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=300)
    plt.close()


def train_model(X_train, Y_train):
    model = LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)
    model.fit(X_train, Y_train)
    return model


def cross_validate_model(model, X, Y, folds=5):
    """Perform cross-validation and print metrics."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    print(f"\nPerforming {folds}-Fold Cross-Validation (Accuracy and Log Loss)...")

    accuracy_scores = cross_val_score(model, X, Y, cv=skf, scoring='accuracy')
    print(f"{folds}-Fold CV Accuracy: {accuracy_scores.mean():.4f}")
    print(f"Standard Deviation of Accuracy scores: {accuracy_scores.std():.4f}")

    logloss_scores = cross_val_score(model, X, Y, cv=skf, scoring='neg_log_loss')
    print(f"{folds}-Fold CV Log Loss: {-logloss_scores.mean():.4f}")
    print(f"Standard Deviation of Log Loss scores: {logloss_scores.std():.4f}")


def evaluate_model(model, X_test, Y_test, X_columns):
    Y_pred = model.predict(X_test)
    Y_pred_proba = model.predict_proba(X_test)  # Probabilities needed for log loss

    # Metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    loss = log_loss(Y_test, Y_pred_proba)  # Log Loss
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Log Loss: {loss:.4f}")
    print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

    # Confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # Predicted vs Actual (scatter)
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(Y_test)), Y_test.values, label='Actual', alpha=0.7)
    plt.scatter(range(len(Y_pred)), Y_pred, label='Predicted', alpha=0.7, color='red', marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("Class (0/1)")
    plt.title("Predicted vs Actual Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'predicted_vs_actual.png'), dpi=300)
    plt.close()

    # Feature importance
    coefficients = pd.DataFrame({'Feature': X_columns, 'Coefficient': model.coef_[0]})
    coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coefficients)
    plt.title("Feature Importance (Logistic Regression)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300)
    plt.close()


# --- Main Execution --- #

def main():
    print("Starting Heart Disease Prediction Project")

    # Load & Explore
    data = load_data(DATASET_PATH, DATA_FILE)
    explore_data(data)
    visualize_target_distribution(data)

    # Preprocess & visualize correlation
    data = preprocess_data(data)
    visualize_correlation_heatmap(data)

    # Features & target
    X = data.drop('HeartDisease', axis=1)
    Y = data['HeartDisease']

    # --- Scaling --- #
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Cross-Validation --- #
    model_cv = LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)
    cross_validate_model(model_cv, X_scaled, Y, folds=5)

    # --- Train/Test Split for final evaluation --- #
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    print("\nTraining Final Model...")

    # --- Train final model --- #
    final_model = train_model(X_train, Y_train)

    # --- Evaluate final model --- #
    evaluate_model(final_model, X_test, Y_test, X.columns)

    print("\n✅ Project complete. Check visualizations folder for outputs.")


if __name__ == "__main__":
    main()
