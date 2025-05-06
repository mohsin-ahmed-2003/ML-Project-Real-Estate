# ML-Project-Real_Estate_prediction
A Machine Learning project that will predict the price of house based on its Latitude, Longitude , etc.,

**Problem Statement** <br>
This project focused on real estate price prediction using machine learning in Python. ​ <br>
The core problem addressed is the need for accurate and reliable estimations of property prices to aid decision-making for various stakeholders in the real estate market, including buyers, sellers, investors, and professionals. ​ <br>
By leveraging machine learning techniques and Python programming, project aims to guide readers through the process of building a predictive model that can forecast real estate values, thus empowering informed transactions and strategic planning within the property sector. <br>

**Proposed Solution** <br>
For a real estate price prediction system using machine learning in Python, we can propose a solution that involves: <br>
**1. Data Collection:** <br>
	Gathering historical and current real estate data. <br>
  E.g., House age, Distance to the nearest MRT station, Number of convenience stores, Latitude, Longitude, House price of unit area. <br>
**2. Data Pre-processing:** <br>
	Cleaning and transforming the data, creating relevant features for the model. <br>
  We can use function like drop() and in encoding(e.g., one-hot encoding, binary encoding) <br>
**3. Model Selection:** <br>
	Exploring and choosing an appropriate machine learning algorithm. <br>
  E.g., linear regression, Decision Tree. <br>
**4. Model Training:** <br>
	Training the selected model on the prepared data. <br>
**5. Evaluation:** <br>
	Assessing the model's performance using relevant metrics. <br>

# Library required to build the model: 
<br>
**1. Pandas:** It is used for data manipulation and analysis. <br>
**2. NumPy:** It is used for numerical computations on array. <br>
**3. Scikit-learn (sklearn):** It provides various algorithms for regression (like Linear Regression, Random Forest Regressor, Gradient Boosting Regressors, etc.), tools for data preprocessing (like scaling, encoding) and evaluation metrics (like MSE, RMSE, R-squared). <br>

# Algorithm Selection:
 <br>
 
**1. Linear Regression Algorithm:** It assumes a linear relationship between the features (e.g., House age, Latitude, Longitude and the target variable (price)  <br>
**2. Random Forest Regression:** It is used to address overfitting issue, this often leads to more accurate and robust predictions for real estate prices. <br>
 <br>
# Steps to Follow:

**1. Library to import** <br>
   import pandas as pd <br>
   import matplotlib.pyplot as plt <br>
   import seaborn as sns <br>
   from sklearn.model_selection import train_test_split <br>
   from sklearn.linear_model import LinearRegression <br>
   from sklearn.tree import RandomForestRegressor <br>
   from sklearn.tree import DecisionTreeRegressor <br>
   from sklearn.metrics import mean_squared_error <br>
   from sklearn.metrics import mean_absolute_error <br>
   from sklearn.metrics import r2_score <br>

**2. Load the dataset**  <br>
  df = pd.read_csv("Downloads/Real_Estate.csv") <br>

**3. **Top 5 row of csv file**** <br>
   df.head() <br>

**4.** **check if it contain any null value** <br>
   df.info()  <br>

# 5. View Visualization: <br>
**Histogram** <br>
   **Set the aesthetic style of the plots** <br>
  sns.set_style("whitegrid") <br>
  
**Create histograms for the numerical columns** <br>
  fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12)) <br>
  fig.suptitle('Histograms of Real Estate Data', fontsize=16) <br>
  
  cols = ['House age', 'Distance to the nearest MRT station', 'Number of convenience stores',
          'Latitude', 'Longitude', 'House price of unit area']
   <br>
  for i, col in enumerate(cols): <br>
      sns.histplot(df[col], kde=True, ax=axes[i//2, i%2]) <br>
      axes[i//2, i%2].set_title(col) <br>
      axes[i//2, i%2].set_xlabel('') <br>
      axes[i//2, i%2].set_ylabel('') <br>
   <br>
  plt.tight_layout(rect=[0, 0.03, 1, 0.95]) <br>
  plt.show() <br>
  ![image](https://github.com/user-attachments/assets/01066947-2f88-4e72-8035-2de81bf17ef1)
 <br>


**Scatter plot**  <br>
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10)) <br>
fig.suptitle('Scatter Plots with House Price of Unit Area', fontsize=16) <br>

 **Scatter plot for each variable against the house price** <br>
sns.scatterplot(df, x='House age', y='House price of unit area', ax=axes[0, 0]) <br>
sns.scatterplot(df, x='Distance to the nearest MRT station', y='House price of unit area', ax=axes[0, 1]) <br>
sns.scatterplot(df, x='Number of convenience stores', y='House price of unit area', ax=axes[1, 0]) <br>
sns.scatterplot(df, x='Latitude', y='House price of unit area', ax=axes[1, 1]) <br>
![image](https://github.com/user-attachments/assets/7765bc6c-4078-4f21-9fd7-24704cd6f1e9)
 <br>


plt.tight_layout(rect=[0, 0.03, 1, 0.95]) <br>
plt.show() <br>

**6. Correlation matrix**  <br>
  correlation_matrix = ndf.corr() <br>
  correlation_matrix <br>

  correlation = df[['House age', 'House price of unit area']].corr() <br>
  print('Correlation of House age & House price of unit area: \n' , correlation) <br>

**7. Plotting the correlation matrix** <br>
  plt.figure(figsize=(10, 6)) <br>
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5) <br>
  plt.title('Correlation Matrix') <br>
  plt.show() <br>
  ![image](https://github.com/user-attachments/assets/242d57f3-a171-46c5-ab2a-f11cbdd503c9)

 <br>
 
# 8. Implement model for each Model that we have imported 
<br>
   Selecting features and target variable <br>
  features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude'] <br>
  target = 'House price of unit area' <br>
  
  X = df[features] <br>
  y = df[target] <br>
  
  # Splitting the dataset into training and testing sets
   <br>
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50) <br>
  
  # Model initialization
   <br>
  model = LinearRegression() **# Change it if you use DecisionTreeRegressor() or RandomForestRegressor** <br>
  
  # Training the model 
  <br>
  model.fit(X_train, y_train) <br>

**9. Calculate the Mean Squared Error of data** <br>
  mse = mean_squared_error(y_test, y_pred) <br>
  print('The Mean Squared Error of data is : ', mse) <br>

**10. Calculate the Root Mean Squared Error of data** <br>
  rmse = mse**0.5 <br>
  print('The Root Mean Squared Error of data is : ', rmse) <br>

**11. Calculate the Mean Absolute Error of data** <br>
  mae = mean_absolute_error(y_test, y_pred) <br>
  print(f"Mean Absolute Error: {mae}") <br>

**12. Calculate R-Squared of data** <br>
  r2 = r2_score(y_test, y_pred) <br>
  print(f"R-squared: {r2}") <br>

**13. Visualization: Actual vs. Predicted values** <br>
  plt.figure(figsize=(8, 6)) <br>
  plt.scatter(y_test, y_pred, alpha=0.5) <br>
  plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) <br>
  plt.xlabel('Actual') <br>
  plt.ylabel('Predicted') <br>
  plt.title('Actual vs. Predicted House Prices') <br>
  plt.show() <br>
  ![image](https://github.com/user-attachments/assets/62f27585-efc3-4be1-8a24-0b2d04f32280)
 <br>

**14. Predict the price of new house by Provide a data of house** <br>
  new_property = pd.DataFrame({ <br>
    'Distance to the nearest MRT station': [150], <br>
    'Number of convenience stores': [8], <br>
    'Latitude': [24.985], <br>
    'Longitude': [121.541], <br>
    'House age': [10] <br>
}) <br>
y_pred = model.predict(new_property) <br>
print("Predicted price for the new property: ",y_pred) <br>

Predicted price for the new property:  **[38.59834007]**

 **NOTE: FOR OTHER ALGORITHM REFER THE CODE OF THIS PROJECT**

**CONCLUSION** <br>
In this project, the model makes reasonably accurate predictions for a significant portion of the test set that provide a accurate price for House relevant to ‘House age , House age, Latitude, Longitude, etc. <br>
The Actual and predicted house prices shows a “Positive correlation” , that indicate that model tends to predict higher prices for houses with higher actual prices. <br>
The quantitative evaluation metrics like MSE indicates the average squared difference between the actual and predicted house prices , where RMSE indicates the same units of house prices.
