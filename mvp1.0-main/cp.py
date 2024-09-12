import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error


def load_and_preprocess_data(filenames):
    """Loads and preprocesses CSV files, attempting to automatically detect date format.

    Args:
        filenames (list): List of filenames to load.

    Returns:
        pd.DataFrame: The combined and preprocessed DataFrame.
    """

    all_dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)

        # Try multiple date formats
        possible_formats = ['%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d']
        for format in possible_formats:
            try:
                df['DATE'] = pd.to_datetime(df['DATE'], format=format)
                break  # Parsing successful, stop trying other formats
            except ValueError:
                pass  # Try the next format if parsing fails

        if 'DATE' not in df.columns:
            raise ValueError(f"Failed to parse date in file '{filename}' using any of the provided formats.")

        df.set_index('DATE', inplace=True)

        # Identify the target column (assuming it's numerical)
        target_column = df.select_dtypes(include=[np.number]).columns[0]

        # Create lagged features
        for lag in range(1, 4):
            df[f'lag_{lag}'] = df[target_column].shift(lag)

        # Drop rows with missing values
        df.dropna(inplace=True)

        all_dfs.append(df)

    combined_df = pd.concat(all_dfs)
    return combined_df


def clean_target_variable(df, target_column):
    """
    Removes rows with infinite or NaN values in the target column.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        target_column (str): Name of the target column.

    Returns:
        pd.DataFrame: The cleaned DataFrame with missing target values removed.
    """

    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(subset=[target_column], inplace=True)
    return df


csv_files = ['WPU10170502.csv', 'PPIACO.csv', 'WPS081.csv', 'WPU083.csv']
df = load_and_preprocess_data(csv_files)
target_column = df.select_dtypes(include=[np.number]).columns[0]
df = clean_target_variable(df, target_column)

X = df[[f'lag_{i}' for i in range(1, 4)]]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

parameters = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 500, 1000]
}

model = xgb.XGBRegressor(objective='reg:squarederror')
grid_search = GridSearchCV(model, parameters, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
