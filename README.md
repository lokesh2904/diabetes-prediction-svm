# Diabetes Prediction using SVM

## Overview
This project implements a Support Vector Machine (SVM) classifier to predict diabetes based on health-related parameters. The dataset used for training and testing is the **PIMA Indians Diabetes Dataset**.

## Features & Technologies Used
- **Python** for model development
- **NumPy, Pandas** for data preprocessing
- **Scikit-learn** for model building and evaluation
- **Matplotlib, Seaborn** for data visualization
- **Joblib** for model saving

## Dataset
The dataset contains the following features:
- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (Target Variable: 0 = No Diabetes, 1 = Diabetes)

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/diabetes-prediction-svm.git
   cd diabetes-prediction-svm
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the model training script:
   ```sh
   python diabetes_svm.py
   ```

## Model Performance
The SVM model achieved **85% accuracy** in predicting diabetes by optimizing feature selection and hyperparameters.

## File Structure
```
├── diabetes.csv           # Dataset file
├── diabetes_svm.py        # Model training and evaluation script
├── svm_diabetes_model.pkl # Trained SVM model
├── scaler.pkl             # Scaler for input normalization
├── requirements.txt       # Dependencies list
├── README.md              # Project documentation
```

## Contributing
Feel free to contribute by improving the model, adding visualizations, or trying different ML algorithms!

## License
This project is open-source and available under the MIT License.

