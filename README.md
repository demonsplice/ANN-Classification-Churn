# ANN-Classification-Churn
This project predicts bank customer churn using machine learning. The dataset includes customer demographics, account info, and churn status. Data is preprocessed (encoding, scaling), then split for training/testing. An artificial neural network is trained to classify churn, and TensorBoard is used for model evaluation and visualization.
# Customer Churn Prediction

This project predicts whether a bank customer will churn (leave the bank) using machine learning and deep learning techniques. It uses the [Churn_Modelling.csv](Churn_Modelling.csv) dataset, which contains customer demographics, account information, and churn status.

## Features

- Data preprocessing: encoding categorical variables, feature scaling
- Model training: Artificial Neural Network (ANN) using TensorFlow/Keras
- Model evaluation: Accuracy and loss tracking with TensorBoard
- Prediction: Jupyter notebook and Streamlit web app for real-time predictions

## Dataset

The dataset includes the following columns:
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (target variable: 1 = churned, 0 = stayed)

## How It Works

1. **Preprocessing:**  
   - Drop unnecessary columns (`RowNumber`, `CustomerId`, `Surname`)
   - Encode `Gender` with LabelEncoder
   - One-hot encode `Geography`
   - Scale features with StandardScaler

2. **Model Training:**  
   - Split data into train/test sets
   - Train an ANN on the processed data
   - Save the model and preprocessing objects

3. **Prediction:**  
   - Load the model and encoders
   - Preprocess new input in the same way as training data
   - Predict churn probability and class

4. **Web App:**  
   - Use Streamlit (`app.py`) to provide a user-friendly interface for predictions

## Usage

### Jupyter Notebook

Run `experiments.ipynb` to preprocess data, train the model, and evaluate performance.

### Streamlit App

1. Install requirements:
    ```
    pip install -r req.txt
    ```
2. Run the app:
    ```
    streamlit run app.py
    ```
3. Enter customer details in the web interface to get churn predictions.

### Prediction Script

Use `pred.ipynb` to test predictions on custom input samples.

## Project Structure

```
├── Churn_Modelling.csv
├── experiments.ipynb
├── pred.ipynb
├── app.py
├── req.txt
├── model.h5
├── label_encoder_gender.pkl
├── One_hot_encoder_geo_.pkl
├── scaler.pkl
```

## Requirements

- Python 3.8–3.11
- pandas
- scikit-learn
- tensorflow
- streamlit
- matplotlib

Install all dependencies with:
```
pip install -r req.txt
```

## License

This project is for educational purposes.
