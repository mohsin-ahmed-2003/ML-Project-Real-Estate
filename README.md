# ML-Project-Real_Estate_prediction
A Machine Learning project that will predict the price of house based on its Latitude, Longitude , etc.,

**Problem Statement**
This project focused on real estate price prediction using machine learning in Python. ​
The core problem addressed is the need for accurate and reliable estimations of property prices to aid decision-making for various stakeholders in the real estate market, including buyers, sellers, investors, and professionals. ​
By leveraging machine learning techniques and Python programming, project aims to guide readers through the process of building a predictive model that can forecast real estate values, thus empowering informed transactions and strategic planning within the property sector.

**Proposed Solution**
For a real estate price prediction system using machine learning in Python, we can propose a solution that involves:
**1. Data Collection:**
	Gathering historical and current real estate data.
  E.g., House age, Distance to the nearest MRT station, Number of convenience stores, Latitude, Longitude, House price of unit area.
**2. Data Pre-processing:**
	Cleaning and transforming the data, creating relevant features for the model.
  We can use function like drop() and in encoding(e.g., one-hot encoding, binary encoding)
**3. Model Selection:**
	Exploring and choosing an appropriate machine learning algorithm.
  E.g., linear regression, Decision Tree.
**4. Model Training:**
	Training the selected model on the prepared data.
**5. Evaluation:**
	Assessing the model's performance using relevant metrics.

**Library required to build the model:**
**1. Pandas:** It is used for data manipulation and analysis.
**2. NumPy:** It is used for numerical computations on array.
**3. Scikit-learn (sklearn):** It provides various algorithms for regression (like Linear Regression, Random Forest Regressor, Gradient Boosting Regressors, etc.), tools for data preprocessing (like scaling, encoding) and evaluation metrics (like MSE, RMSE, R-squared).

**Algorithm Selection:**
**1. Linear Regression Algorithm:** It assumes a linear relationship between the features (e.g., House age, Latitude, Longitude and the target variable (price)
**2. Random Forest Regression:** It is used to address overfitting issue, this often leads to more accurate and robust predictions for real estate prices.

**Steps to Follow:**

**1. Library to import**
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.tree import RandomForestRegressor
   from sklearn.tree import DecisionTreeRegressor
   from sklearn.metrics import mean_squared_error
   from sklearn.metrics import mean_absolute_error
   from sklearn.metrics import r2_score

**2. Load the dataset**
  df = pd.read_csv("Downloads/Real_Estate.csv")

**3. **Top 5 row of csv file****
   df.head()

**4.** **check if it contain any null value**
   df.info() 

**5. View Visualization:**
**Histogram**
  # Set the aesthetic style of the plots
  sns.set_style("whitegrid")
  
  # Create **histograms** for the numerical columns
  fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
  fig.suptitle('Histograms of Real Estate Data', fontsize=16)
  
  cols = ['House age', 'Distance to the nearest MRT station', 'Number of convenience stores',
          'Latitude', 'Longitude', 'House price of unit area']
  
  for i, col in enumerate(cols):
      sns.histplot(df[col], kde=True, ax=axes[i//2, i%2])
      axes[i//2, i%2].set_title(col)
      axes[i//2, i%2].set_xlabel('')
      axes[i//2, i%2].set_ylabel('')
  
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.show()
  ![image](https://github.com/user-attachments/assets/01066947-2f88-4e72-8035-2de81bf17ef1)



**Scatter plot**
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle('Scatter Plots with House Price of Unit Area', fontsize=16)

# Scatter plot for each variable against the house price
sns.scatterplot(df, x='House age', y='House price of unit area', ax=axes[0, 0])
sns.scatterplot(df, x='Distance to the nearest MRT station', y='House price of unit area', ax=axes[0, 1])
sns.scatterplot(df, x='Number of convenience stores', y='House price of unit area', ax=axes[1, 0])
sns.scatterplot(df, x='Latitude', y='House price of unit area', ax=axes[1, 1])
![image](https://github.com/user-attachments/assets/7765bc6c-4078-4f21-9fd7-24704cd6f1e9)



plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

**6. Correlation matrix**
  correlation_matrix = ndf.corr()
  correlation_matrix

  correlation = df[['House age', 'House price of unit area']].corr()
  print('Correlation of House age & House price of unit area: \n' , correlation)

**7. Plotting the correlation matrix**
  plt.figure(figsize=(10, 6))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
  plt.title('Correlation Matrix')
  plt.show()
  ![image](https://github.com/user-attachments/assets/242d57f3-a171-46c5-ab2a-f11cbdd503c9)


8. Implement model for each Model that we have imported
  # Selecting features and target variable
  features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
  target = 'House price of unit area'
  
  X = df[features]
  y = df[target]
  
  # Splitting the dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
  
  # Model initialization
  model = LinearRegression() **# Change it if you use DecisionTreeRegressor() or RandomForestRegressor**
  
  # Training the model
  model.fit(X_train, y_train)

**9. Calculate the Mean Squared Error of data**
  mse = mean_squared_error(y_test, y_pred)
  print('The Mean Squared Error of data is : ', mse)

**10. Calculate the Root Mean Squared Error of data**
  rmse = mse**0.5
  print('The Root Mean Squared Error of data is : ', rmse)

**11. Calculate the Mean Absolute Error of data**
  mae = mean_absolute_error(y_test, y_pred)
  print(f"Mean Absolute Error: {mae}")

**12. Calculate R-Squared of data**
  r2 = r2_score(y_test, y_pred)
  print(f"R-squared: {r2}")

**13. Visualization: Actual vs. Predicted values**
  plt.figure(figsize=(8, 6))
  plt.scatter(y_test, y_pred, alpha=0.5)
  plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
  plt.xlabel('Actual')
  plt.ylabel('Predicted')
  plt.title('Actual vs. Predicted House Prices')
  plt.show()
  ![image](https://github.com/user-attachments/assets/62f27585-efc3-4be1-8a24-0b2d04f32280)


**14. Predict the price of new house by Provide a data of house**
  new_property = pd.DataFrame({
    'Distance to the nearest MRT station': [150],
    'Number of convenience stores': [8],
    'Latitude': [24.985],
    'Longitude': [121.541],
    'House age': [10]
})
y_pred = model.predict(new_property)
print("Predicted price for the new property: ",y_pred)

Predicted price for the new property:  **[38.59834007]**

 **NOTE: FOR OTHER ALGORITHM REFER THE CODE OF THIS PROJECT**

**CONCLUSION**
In this project, the model makes reasonably accurate predictions for a significant portion of the test set that provide a accurate price for House relevant to ‘House age , House age, Latitude, Longitude, etc.
The Actual and predicted house prices shows a “Positive correlation” , that indicate that model tends to predict higher prices for houses with higher actual prices.
The quantitative evaluation metrics like MSE indicates the average squared difference between the actual and predicted house prices , where RMSE indicates the same units of house prices.
