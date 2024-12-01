from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

# Function to simulate the model training and return the model and scaler
def load_model():
    # Simulating the training process with a small dataset
    df = pd.DataFrame({
        'Age': [20, 25, 30, 35, 40],
        'Tenure': [1, 2, 3, 4, 5],
        'Balance': [1000, 2000, 3000, 4000, 5000],
        'NumOfProducts': [1, 2, 3, 4, 5],
        'HasCrCard': [1, 1, 0, 1, 0],
        'IsActiveMember': [1, 0, 1, 0, 1],
        'EstimatedSalary': [50000, 60000, 70000, 80000, 90000],
        'Exited': [0, 0, 1, 0, 1]
    })

    # Features and target variable
    X = df[['Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
    y = df['Exited']

    # Scaling the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Training the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return model, scaler

# Load the model and scaler
model, scaler = load_model()

# HTTP Request Handler class
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    # Handle GET requests
    def do_GET(self):
        print(f"Request path: {self.path}")  # Log the request path
        if self.path == '/':
            # Redirect to the HTML page
            self.send_response(302)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            # Serve the index.html file
            self.serve_file('index.html')
        else:
            # Return 404 for other paths
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Page not found")

    # Helper function to serve an HTML file
    def serve_file(self, filename):
        """Helper function to serve HTML files"""
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.send_response(404)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"File not found")
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"File not found")

    # Handle POST requests for predictions
    def do_POST(self):
        if self.path == '/predict':
            # Read and parse the incoming JSON data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            # Prepare the input data for prediction
            user_data = np.array([[data['age'], data['tenure'], data['balance'], data['num_of_products'],
                                   data['has_crcard'], data['is_active_member'], data['estimated_salary']]])

            # Scale the input data using the same scaler
            user_data_scaled = scaler.transform(user_data)

            # Make the prediction
            prediction = model.predict(user_data_scaled)

            # Respond with the prediction result in JSON format
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # Send the prediction result (0 for "No Churn", 1 for "Churn")
            response = {'prediction': int(prediction[0])}
            self.wfile.write(json.dumps(response).encode())

# Set up the HTTP server to listen on port 5000
server_address = ('', 5000)  # Run on port 5000
httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)

print("Server started on port 5000")
httpd.serve_forever()
