import numpy as np
import joblib
import tkinter as tk

# Load saved models and preprocessing tools
model = joblib.load('catboost_optimized.pkl')
scaler = joblib.load('scaler.pkl')

# Prediction function
def predict_model(inputs):
    # Prepare input data
    data = np.array([inputs])
    
    # Data preprocessing
    data_scaled = scaler.transform(data)
    
    # Predict
    prediction = model.predict(data_scaled)
    return prediction[0]

# Create the main window
root = tk.Tk()
root.title("ML Prediction for Extraction Yield")
root.geometry("1000x650")
root.configure(bg="#FFF8DC")

# Title label
title_label = tk.Label(root, text="ML Prediction for Extraction Yield", font=("Helvetica", 28, "bold"), fg="#0D47A1", bg="#FFF8DC")
title_label.grid(row=0, column=0, columnspan=2, pady=20)

# Create labeled entry fields for input parameters
def create_labeled_entry(parent, text, row):
    label = tk.Label(parent, text=text, font=("Helvetica", 16), fg="#333333", bg=parent["bg"])
    label.grid(row=row, column=0, padx=20, pady=10, sticky=tk.W)
    
    entry = ttk.Entry(parent, width=30, font=("Helvetica", 14))
    entry.grid(row=row, column=1, padx=20, pady=10)
    
    return entry

# Input fields
inputs = {}
input_params = [
    "Adsorbent dosage (mg):",
    "Grinding time (min):",
    "DES dosage (%):",
    "Extractant volume (mL):",
    "Extraction time (min):"
]

for i, param in enumerate(input_params):
    inputs[param] = create_labeled_entry(root, param, i + 1)

# Predict button handler
def handle_predict():
    try:
        input_values = [float(inputs[param].get()) for param in input_params]
        prediction = predict_model(input_values)
        result_label.config(text=f'Predicted Extraction Yield: {prediction:.2f} mg/g', fg="#2E7D32")
    except ValueError:
        result_label.config(text="Please enter valid numeric values.", fg="red")

# Predict button
style = ttk.Style()
style.configure('Custom.TButton', font=("Helvetica", 16), padding=10)

predict_button = ttk.Button(root, text="Predict", command=handle_predict, style='Custom.TButton')
predict_button.grid(row=len(input_params) + 1, column=0, columnspan=2, pady=30)

# Result label
result_label = tk.Label(root, text="Prediction will appear here", font=("Helvetica", 18, "bold"), fg="#1B5E20", bg="#FFF8DC")
result_label.grid(row=len(input_params) + 2, column=0, columnspan=2, pady=15)

# Run the Tkinter main loop
root.mainloop()
