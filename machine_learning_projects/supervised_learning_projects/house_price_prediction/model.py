import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

# Display large numbers normally
pd.set_option('display.float_format', lambda x: '%.0f' % x)

# --- Configuration --- #
DATASET_PATH = "shree1992/housedata"
DATA_FILE = "/data.csv"
TEST_SIZE = 0.1
RANDOM_STATE = 42
OUTPUT_DIR = "./visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    print("\nSummary statistics:\n", data.describe())


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    print("\nEncoding categorical columns...")
    encoder = LabelEncoder()
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col])
    print(f"Encoded columns: {list(categorical_cols)}")
    return data


def visualize_target_distribution(data: pd.DataFrame, target_column: str):
    plt.figure(figsize=(7, 4))
    sns.histplot(data[target_column], kde=True, bins=30)
    plt.title(f"Distribution of {target_column}")
    plt.xlabel(target_column)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'target_distribution.png'), dpi=300)
    plt.close()


def visualize_correlation_heatmap(data: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=300)
    plt.close()


def train_model(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model


def evaluate_model(model, X_test, Y_test, X_columns):
    Y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    print(f"\nMSE: {mse:.4f}, R²: {r2:.4f}")

    # Predicted vs Actual
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=Y_test, y=Y_pred, alpha=0.7)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual Values')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'predicted_vs_actual.png'), dpi=300)
    plt.close()

    # Residuals (optional)
    residuals = Y_test - Y_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title('Residual Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'residual_distribution.png'), dpi=300)
    plt.close()


def main():
    print("Starting House Price Prediction Project")

    # Load and explore data
    data = load_data(DATASET_PATH, DATA_FILE)
    explore_data(data)

    target_column = 'price' if 'price' in data.columns else data.columns[-1]
    visualize_target_distribution(data, target_column)

    # Preprocess
    data = preprocess_data(data)
    visualize_correlation_heatmap(data)

    # Split features and target
    X = data.drop(target_column, axis=1)
    Y = data[target_column]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = train_model(X_train_scaled, Y_train)

    # Evaluate
    evaluate_model(model, X_test_scaled, Y_test, X.columns)

    print("✅ Project complete. Check 'visualizations' folder for plots.")


if __name__ == "__main__":
    main()
