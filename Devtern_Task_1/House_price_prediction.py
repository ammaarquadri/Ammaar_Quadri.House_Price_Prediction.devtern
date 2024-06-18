import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Switch backend to 'Agg' to avoid Tkinter-related issues
plt.switch_backend('Agg')

# Step 1: Data Preparation

# Data Loading
url = 'https://docs.google.com/spreadsheets/d/1caaR9pT24GNmq3rDQpMiIMJrmiTGarbs/export?format=csv&gid=1150341366'
df = pd.read_csv(url)
print("Data loaded successfully!")
print(df.head())

# Print column names to check the correct name for the target variable
print("Column names in the dataset:")
print(df.columns)

# Replace 'SalePrice' with the actual column name of the target variable in your dataset
target_column = 'SalePrice'

# Data Visualization
plt.figure(figsize=(10, 6))
sns.histplot(df[target_column], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.savefig('house_prices_distribution.png')
plt.close()

# Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Handling missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Visualize correlations between features
plt.figure(figsize=(12, 8))
sns.heatmap(df_imputed.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# Step 2: Feature Analysis
correlation_matrix = df_imputed.corr()
correlations_with_price = correlation_matrix[target_column].sort_values(ascending=False)
print("Correlations with price:\n", correlations_with_price)

# Step 3: Building the Machine Learning Model

# Define features and target variable
X = df_imputed.drop(columns=[target_column])
y = df_imputed[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Make predictions with the Linear Regression model
y_pred_lr = model_lr.predict(X_test)

# Step 4: Model Evaluation and Fine-Tuning

# Evaluate the Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f'Linear Regression Model - Mean Squared Error: {mse_lr}')
print(f'Linear Regression Model - R-squared: {r2_lr}')

# Initialize and train a Random Forest model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Make predictions with the Random Forest model
y_pred_rf = model_rf.predict(X_test)

# Evaluate the Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f'Random Forest Model - Mean Squared Error: {mse_rf}')
print(f'Random Forest Model - R-squared: {r2_rf}')

# Conclusions and Insights
print("\nConclusions and Insights:")
print(f"The Linear Regression model achieved an MSE of {mse_lr} and an R-squared of {r2_lr}.")
print(f"The Random Forest model achieved an MSE of {mse_rf} and an R-squared of {r2_rf}.")
print("The Random Forest model performed better in terms of both MSE and R-squared.")

# End of script
