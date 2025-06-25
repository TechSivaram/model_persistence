import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle  # Import the pickle module
import os      # To clean up the saved file

# --- Configuration for the model file ---
model_filename = 'linear_regression_model_single_file.pkl'

print("--- ML Model Persistence: All in One File ---")

# --- 1. Data Preparation (from previous steps) ---
print("\n--- 1. Preparing Data ---")
try:
    df_students = pd.read_csv('student_scores.csv')
except FileNotFoundError:
    print("Creating dummy student_scores.csv for demo...")
    student_content = """StudentID,Name,Math,Science,English,Gender,HoursStudied
1,Alice,85,90,78,Female,5
2,Bob,70,65,75,Male,3
3,Charlie,92,88,95,Male,8
4,David,78,80,70,Male,4
5,Eve,95,92,89,Female,7
6,Frank,60,55,62,Male,2
7,Grace,88,91,85,Female,6
8,Heidi,72,70,73,Female,4
9,Ivan,80,82,85,Male,6
10,Jane,90,90,92,Female,8
"""
    with open('student_scores.csv', 'w') as f:
        f.write(student_content)
    df_students = pd.read_csv('student_scores.csv')

df_students['AverageScore'] = df_students[['Math', 'Science', 'English']].mean(axis=1)

X = df_students[['HoursStudied']] # Features
y = df_students['AverageScore']   # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data prepared. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# --- 2. Model Training ---
print("\n--- 2. Training Model ---")
trained_model = LinearRegression() # Create a fresh model instance
trained_model.fit(X_train, y_train) # Train the model

print("Model training complete.")
print(f"Trained model coefficients: {trained_model.coef_}")
print(f"Trained model intercept: {trained_model.intercept_}")

# --- 3. Save the Trained Model using pickle.dump() ---
print(f"\n--- 3. Saving Model to '{model_filename}' ---")
try:
    with open(model_filename, 'wb') as file: # 'wb' means write binary
        pickle.dump(trained_model, file)
    print("Model successfully saved using pickle.dump().")
except Exception as e:
    print(f"Error saving model: {e}")

# --- 4. Load the Model using pickle.load() ---
print(f"\n--- 4. Loading Model from '{model_filename}' ---")
loaded_model = None
try:
    with open(model_filename, 'rb') as file: # 'rb' means read binary
        loaded_model = pickle.load(file)
    print("Model successfully loaded using pickle.load().")
except FileNotFoundError:
    print(f"Error: Model file '{model_filename}' not found.")
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")

# --- 5. Use the Loaded Model for Prediction ---
print("\n--- 5. Using the Loaded Model for Prediction ---")
if loaded_model:
    # Create a new data point for prediction
    new_hours_studied = 6
    new_data_for_prediction = pd.DataFrame([[new_hours_studied]], columns=['HoursStudied'])

    # Make prediction using the loaded model
    predicted_score = loaded_model.predict(new_data_for_prediction)

    print(f"Prediction for {new_hours_studied} hours studied (using loaded model): {predicted_score[0]:.2f}")
else:
    print("Model was not loaded, cannot make prediction.")

# --- Optional: Clean up the generated model file ---
# print(f"\n--- Optional: Cleaning up '{model_filename}' ---")
# if os.path.exists(model_filename):
#     os.remove(model_filename)
#     print(f"Removed '{model_filename}'.")

print("\n--- End of Demonstration ---")