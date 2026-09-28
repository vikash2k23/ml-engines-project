import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import ssl

# Bypass SSL certificate verification for dataset download
ssl._create_default_https_context = ssl._create_unverified_context


def run_real_estate_pipeline():
    print("1. Fetching California Housing Dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    print("2. Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. Building Scikit-Learn Pipeline...")
    # Pipeline sequentially applies a list of transforms and a final estimator
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print("4. Training Model (this might take a few seconds)...")
    # This automatically scales the data, then trains the Random Forest
    pipeline.fit(X_train, y_train)

    print("5. Evaluating Model...")
    # The pipeline automatically scales the test data using the logic it learned from the training data
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    print("6. Saving the trained pipeline...")
    # Save the pipeline structure and weights so it can be deployed later
    os.makedirs(os.path.join(os.path.dirname(__file__), '../models'), exist_ok=True)
    save_path = os.path.join(os.path.dirname(__file__), '../models/real_estate_pipeline.pkl')
    joblib.dump(pipeline, save_path)
    print(f"Pipeline saved to {save_path}")


def interactive_prediction():
    # Construct the path to your saved model
    model_path = os.path.join(os.path.dirname(__file__), '../models/real_estate_pipeline.pkl')

    if not os.path.exists(model_path):
        print("\nError: Model not found. Please run the training pipeline first to generate the .pkl file.")
        return

    print("\n--- Real Estate Value Predictor ---")
    print("Enter the following details for the neighborhood (press Enter to use typical default values):")

    try:
        # Prompting for user input with fallback defaults
        med_inc = float(input("Median Income (in $10k, e.g., 3.5 for $35k) [3.5]: ") or 3.5)
        house_age = float(input("Average House Age (in years) [20.0]: ") or 20.0)
        ave_rooms = float(input("Average Rooms per Dwelling [5.0]: ") or 5.0)
        ave_bedrms = float(input("Average Bedrooms per Dwelling [1.0]: ") or 1.0)
        population = float(input("Block Population [1000.0]: ") or 1000.0)
        ave_occup = float(input("Average Occupancy per Household [3.0]: ") or 3.0)
        latitude = float(input("Latitude [35.0]: ") or 35.0)
        longitude = float(input("Longitude [-119.0]: ") or -119.0)

        # Create a Pandas DataFrame with the exact column names the model was trained on
        input_data = pd.DataFrame(
            [[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]],
            columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        )

        print("\nLoading trained model and calculating...")
        pipeline = joblib.load(model_path)
        prediction = pipeline.predict(input_data)[0]

        # The California housing dataset targets are expressed in hundreds of thousands of dollars
        predicted_value = prediction * 100000

        print("=====================================================")
        print(f"  ESTIMATED HOUSE VALUE: ${predicted_value:,.2f}")
        print("=====================================================\n")

    except ValueError:
        print("\nInvalid input. Please enter numeric values only.")

if __name__ == "__main__":
    run_real_estate_pipeline()