import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import numpy as np

# Display numbers without scientific notation
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


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    print("\nEncoding categorical columns...")
    encoder = LabelEncoder()
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col])
    print(f"Encoded columns: {list(categorical_cols)}")

    # Drop irrelevant columns
    drop_cols = ['street', 'country', 'date']
    dropped = [col for col in drop_cols if col in data.columns]
    data = data.drop(columns=dropped)
    print(f"\nDropped columns: {dropped}")

    return data


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    print("\nFeature Engineering...")
    # House age
    data['house_age'] = 2025 - data['yr_built']
    # Renovation age (0 if never renovated)
    data['renovation_age'] = 2025 - data['yr_renovated'].replace(0, 2025)
    # Living space ratio
    data['living_ratio'] = data['sqft_living'] / data['sqft_lot']
    # Bathrooms per bedroom
    data['bath_per_bed'] = data['bathrooms'] / (data['bedrooms'].replace(0, 1))
    # Living area per total room
    data['living_per_room'] = data['sqft_living'] / (data['bedrooms'] + data['bathrooms'] + 1)
    # Living area per floor
    data['living_per_floor'] = data['sqft_living'] / (data['floors'].replace(0, 1))
    # Basement area ratio
    data['basement_ratio'] = data['sqft_basement'] / (data['sqft_living'] + 1)
    # Rooms per living area
    data['room_density'] = (data['bedrooms'] + data['bathrooms']) / (data['sqft_living'] + 1)
    print("Engineered features: ['house_age', 'renovation_age', 'living_ratio', 'bath_per_bed', 'living_per_room', 'basement_ratio', 'room_density']")
    return data


def scale_target(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    print(f"\nScaling target ({target_column}) using log transformation...")
    # Log-transform the target
    data['price_scaled'] = np.log1p(data[target_column])

    # Show first 5 prices before and after scaling in rows & columns
    display_df = pd.DataFrame({
        f'Original {target_column}': data[target_column].head(),
        'Scaled Price (log1p)': data['price_scaled'].head()
    })
    print("\nSample prices before and after scaling:")
    print(display_df.to_string(index=False))

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
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=300)
    plt.close()


def train_model(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model


def evaluate_model(model, X_test, Y_test):
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

    # Residuals
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
    visualize_target_distribution(data, 'price')

    # Preprocess and feature engineering
    data = preprocess_data(data)
    data = feature_engineering(data)
    data = scale_target(data, 'price')
    visualize_correlation_heatmap(data)

    # Split features and target
    X = data.drop(['price', 'price_scaled'], axis=1)
    Y = data['price_scaled']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = train_model(X_train_scaled, Y_train)

    # Evaluate
    evaluate_model(model, X_test_scaled, Y_test)

    print("✅ Project complete. Check 'visualizations' folder for plots.")


if __name__ == "__main__":
    main()
