import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
try:
    df = pd.read_excel('bank_churn_data.xlsx', engine='openpyxl')  # Use openpyxl engine for reading Excel files
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit(1)

# Check the columns of the DataFrame to ensure the columns exist
print("Columns in the dataset:", df.columns)

# Strip any leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# Feature columns and target
try:
    X = df[['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
    y = df['Exited']  # Target column (Exited - churn)
except KeyError as e:
    print(f"Column not found: {e}")
    exit(1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Set a higher max_iter if the model doesn't converge
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])  # Specify labels to avoid warnings

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Function to take input from the user
def get_user_input():
    try:
        print("Please enter the following details:")
        age = float(input("Age: "))
        tenure = float(input("Tenure (in years): "))
        balance = float(input("Account Balance: "))
        num_of_products = int(input("Number of products: "))
        has_crcard = int(input("Has credit card (1 for Yes, 0 for No): "))
        is_active_member = int(input("Is active member (1 for Yes, 0 for No): "))
        estimated_salary = float(input("Estimated Salary: "))
        
        # Prepare the input data for prediction (user data as a DataFrame)
        user_data = pd.DataFrame([[age, tenure, balance, num_of_products, has_crcard, is_active_member, estimated_salary]],
                                 columns=['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'])
        
        return user_data
    except ValueError:
        print("Invalid input. Please enter the correct values.")
        return None

# Get user input
user_data = get_user_input()

if user_data is not None:
    # Scale the user input data using the same scaler fitted to the training data
    user_data_scaled = scaler.transform(user_data)

    # Predict churn for the user input
    user_prediction = model.predict(user_data_scaled)

    # Output the prediction
    if user_prediction == 1:
        print("The customer is likely to churn.")
    else:
        print("The customer is likely to stay.")
