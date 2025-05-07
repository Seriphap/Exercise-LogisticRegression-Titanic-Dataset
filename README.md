# Exercise-LogisticRegression-Titanic-Dataset
<p><font size="4">&nbsp;&nbsp;&nbsp;View in Steamlit (Web Application):https://exerciselogisticregressiontitanicdataset-cmpbbggazqzs7dsqxddnz.streamlit.app/</p> 
<br>

## Step1: Data cleansing and features selection
<p><font size="4">&nbsp;&nbsp;&nbsp;Dataset: https://drive.google.com/file/d/1w1XQ8RzDDOLrUpBoE9oqRSPLO6QILEog/view?usp=drive_link </p>
  
- Sample Raw Data
<img src="https://github.com/user-attachments/assets/78b9cf70-9168-4f49-822c-386d333fa2df" width="100%">
<br>
<br>

- Feastures selection
<p><font size="4">&nbsp;&nbsp;&nbsp;Select features for training model and One-hot encoding for categorical data </p>
<img src="https://github.com/user-attachments/assets/b19406d2-beec-4190-aa5e-cc7f63d53f7b" width="50%">
<br>
<br
  
- Sample Data Frame for training model and testing model
<img src="https://github.com/user-attachments/assets/f4fe54b0-8668-43a9-b872-70a26e5e859c" width="50%">

## Step2: Apply Logistic Regresstion for training and testing model
```python
#Split the data into training and testing sets
X_train_LG, X_test_LG, y_train_LG, y_test_LG = train_test_split(X, y, test_size=0.2, random_state=42)

#Create a Logistic Regression model
model = LogisticRegression(max_iter=10000)  # Increase iterations if necessary
```
```python
#Make predictions on the train set
y_predict_train_LG = model.predict(X_train_LG)
print("Classification Report for Training Set:")
print('Predicted train values (Head):', y_predict_train_LG[:10])  # First 10 predicted values
print('Predicted train values (Tail):', y_predict_train_LG[-10:])  # Last 10 predicted values
```
```python
#Make predictions on the test set
y_predict_test_LG = model.predict(X_test_LG)
print("\nClassification Report for Test Dataset:")
print('Predicted test values (Head):', y_predict_test_LG[:10])  # First 10 predicted values
print('Predicted test values (Tail):', y_predict_test_LG[-10:])  # Last 10 predicted values
```
## Step3: Evaluate the model's performance

| Test | Train |
|--------|---------|
| <img src="https://github.com/user-attachments/assets/05c3f015-96ba-447f-95c0-c7fe54c5a6db" width="500"> | <img src="https://github.com/user-attachments/assets/48bd0879-6604-42b4-a14b-1ea916e2d56a" width="500">|

<img src="https://github.com/user-attachments/assets/49e3da9f-13ca-4dc0-b7ca-7d054b9d757b" width="100%">




