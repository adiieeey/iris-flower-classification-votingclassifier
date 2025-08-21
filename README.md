# Iris Flower Classification using Voting Classifier

This project implements an **Iris Flower Classification** model using a **Voting Classifier** that combines multiple machine learning algorithms for better performance.  
The Iris dataset is a classic dataset in machine learning, widely used for pattern recognition tasks.

---

## 📂 Project Structure
```
├── eg1.py             # Main script to train and test the model
├── iris_model.pkl     # Saved trained model (Voting Classifier)
```

---

## 🚀 Features
- Uses the **Iris dataset** for training and testing.  
- Implements a **Voting Classifier** that combines multiple ML models.  
- Trains and saves the model as a `.pkl` file.  
- Can be extended for deployment or further experimentation.  

---

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/adiieeey/iris-flower-classification-votingclassifier.git
   cd iris-flower-classification-votingclassifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is not available, manually install `scikit-learn`, `pandas`, and `numpy`.)*

---

## ▶️ Usage

Run the main script:
```bash
python eg1.py
```

This will:
- Load the Iris dataset.  
- Train the Voting Classifier.  
- Save the trained model as `iris_model.pkl`.  

---

## 📊 Models Used
The Voting Classifier combines multiple algorithms:
- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  

*(Modify `eg1.py` to experiment with more models.)*

---

## 📈 Results
The model achieves high accuracy on the Iris dataset by combining multiple classifiers instead of relying on a single one.

---

## 📌 Future Improvements
- Add visualization for decision boundaries.  
- Deploy the model with Flask/Django or Streamlit.  
- Experiment with more ensemble techniques like Bagging/Boosting.  

  
