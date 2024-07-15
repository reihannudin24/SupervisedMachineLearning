#
# # import libraries
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 1. Loading and Dsiplaying 0_Data : The dataset is loaded, and the first few rows are displayed.
# df = pd.read_csv('../0_Data/diagnosing_diabetes_dataset.csv')
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
#
# # print(df.columns)
# # print(df.head)
#
# # 2. Checking 0_Data Types : The data types of the columns are printed
# typeData = df.dtypes.value_counts()
# # print(typeData)
#
# feature_integer = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Age', 'Outcome']]
# feature_float = df[['BMI', 'DiabetesPedigreeFunction']]
#
# # print(f"Integer Features : {feature_integer}")
# # print(f"Float Features : {feature_float}")
#
# # Handling missing and Duplicate values : Missing and duplicate values are checked and printed.
# checkNull = df.isna().sum()
# checkDuplicate = df.duplicated().sum()
#
# # print(checkNull)
# # print(checkDuplicate)
#
# # Linear regression digunakan untuk memodelkan hubungan antara satu variabel dependen (respon) dengan satu atau lebih variabel independen (prediktor). Tujuan utamanya adalah untuk
#
# # filter data for outcomes diabetes
# diabetes_df = df[df['Outcome'] == 1]
# not_diabetes_df = df[df['Outcome'] == 0]
#
# # print("Total 0_Data : ", df.size)
# # print("Total Diabetes 0_Data : ", diabetes_df.size)
# # print("Total Not Diabetes 0_Data : ", not_diabetes_df.size)
#
# # Result = Total 0_Data :  6912, Total Diabetes 0_Data :  2412, Total Not Diabetes 0_Data :  4500
# # lets visualize
# labels = ['Positive', 'Negative']
# sizes = [len(diabetes_df), len(not_diabetes_df)]
# colors = ['blue', 'cyan']
# explode = (0.1, 0)
# plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.title('Positive And Negative Diabetes')
# plt.axis('equal')
# plt.show()
#
# outcomes = df['Outcome']
#
#
# # i want make correlation betwen oucome and all point :
#
# # Pregnacies X Outcomes ===============================
# pregnanciesOutcomes = df[['Pregnancies' , 'Outcome']]
# pregnanciesUniques = df['Pregnancies'].unique()
# # [ 6  1  8  0  5  3 10  2  4  7  9 11 13 15 17 12 14]
#
# for pregnant in pregnanciesUniques:
#     subset = df[df['Pregnancies'] == pregnant]
#     outcomes_counts = subset['Outcome'].value_counts(normalize=True)
#
#     outcomes = df['Outcome']
#     outcome_counts = subset['Outcome'].value_counts()
#
#     outcome_count_positive = outcome_counts.get(1, 0)  # Jika tidak ada nilai 1, kembalikan 0
#     outcome_count_negative = outcome_counts.get(0, 0)  # Jika tidak ada nilai 0, kembalikan 0
#
#     # print(f"Pregrancies : {pregnant}")
#     # print(f"OutcomeCountPositive  : {outcome_count_positive}")
#     # print(f"OutcomeCountNegative : {outcome_count_negative}")
#     # print(outcomes_counts)
#
# # jika status lebih sama dengan dari Pregrancies : 7
# # maka kemungkinan sesorang untuk diabetes 55% lebih besar
# # find correlation
# Pregnat_correlation = pregnanciesOutcomes['Pregnancies'].corr(pregnanciesOutcomes['Outcome'])
# print(Pregnat_correlation)
# # Correlation between Pregrancies X Outcomes : 0.2218981530339867
#
#
# # Glucose X Outcomes ===============================
# glucoseOutcomes = df[['Glucose', 'Outcome']]
# glucoseUniques = df['Glucose'].unique()
# outcomeUniques = df['Outcome'].unique()
#
# # Deine validate for Glucose
# glucose_thresholds = [50, 80, 100, 120, 150]
#
# for threshold in glucose_thresholds:
#
#     # make subset data base value of Glucose > threshold
#     subset = df[df['Glucose'] > threshold]
#
#     # count total positive and negative
#     outcome_counts = subset['Outcome'].value_counts()
#
#     outcome_count_positive = outcome_counts.get(1, 0)  # Jika tidak ada nilai 1, kembalikan 0
#     outcome_count_negative = outcome_counts.get(0, 0)  # Jika tidak ada nilai 0, kembalikan 0
#
#     print("====================")
#
#     print(f"Glucose > {threshold}")
#     print(f"OutcomeCountPositive : {outcome_count_positive}")
#     print(f"OutcomeCountNegative : {outcome_count_negative}")
#     print(outcome_counts)
#
# # summary glucose
# # > 50 : 0 = 496, 1 = 266
# # > 80 : 0 = 457, 1 = 264
# # > 100 : 0 = 306, 1 = 248
# # > 120 : 0 = 195, 1 = 154
# # > 150 : 0 = 105, 1 = 35
#
# Glucose_correlation = glucoseOutcomes['Glucose'].corr(glucoseOutcomes['Outcome'])
# print(Glucose_correlation)
#
#
# # BloodPressure X Outcomes ===============================
# bloodPressureOutcomes = df[['BloodPressure', 'Outcome']]
# bloodPressureUniques = df['Bloo']
# glucoseUniques = df['Glucose'].unique()
# outcomeUniques = df['Outcome'].unique()
#
# # SkinThickness X Outcomes ===============================
# # Insulin X Outcomes ===============================
# # BMI X Outcomes ===============================
# # DiabetesPedigreeFunction X Outcomes ===============================
# # Age X Outcomes ===============================


#  ===============




# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# show data
df = pd.read_csv('../customer_churn_dataset.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# print(df.columns)

# All columns
# ['CustomerID', 'Age', 'Gender', 'Tenure', 'Usage Frequency',
#        'Support Calls', 'Payment Delay', 'Subscription Type',
#        'Contract Length', 'Total Spend', 'Last Interaction', 'Churn']

# Tujuannya
# 1. untuk memprediksi berapa banyak orang yang menyudahi/memutuskan berlangganan berdasarkan : (age, subscribe type )
# 2. untuk memprediksi massa waktu rata rata customer berlangganan berdasarkan tipe berlangganan  total pengeluaran

# renaming label column has spaces will be underscore
df.columns = df.columns.str.lower().str.replace(' ', '_')
# renaming column names
df = df.rename(columns={'customerid' : 'customer_id'})
print(df.head())

# knowing type data
typeData = df.dtypes.value_counts()
print(typeData)

# check data theres a null data
theresNull = df.isna().sum()
print(theresNull)

# check data theres a duplicate data
theresDuplicate = df.duplicated().sum()
print(theresDuplicate)

# right now data is clean

print("============== EDA ================\n")
# now let's do EDA (Exploring and Processing data)
# EDA memiliki tujuan untuk memahami distribusi antar variable dengan variable lainnya

# now i want to make a label based their type data, in this case have 2 typedata : Numerical(Int64) and Object

# filter data for chur
churned_df = df[df['churn'] == 1]
not_churned_df = df[df['churn'] == 0]
print("Total data : ", df.size)
print("Total data has churned : ", churned_df.size)
print("Total data has no churned : ", not_churned_df.size)

labels = ['Churned', 'Not Churned']
sizes = [len(churned_df), len(not_churned_df)]
colors = ['red', 'green']
explode = (0.1, 0)  # Memisahkan potongan 'Churned'

# show pie chart percent of data has chrun and not churn
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Customer Churn 0_Data')
plt.axis('equal')
plt.show()

# i want get data from subsciption_type
subs_basic_df = df[df['subscription_type'] == 'Basic']
subs_std_df = df[df['subscription_type'] == 'Standard']
subs_prem_df = df[df['subscription_type'] == 'Premium']

# show bar chart
plt.bar(['Basic', 'Standard', 'Premium'], [len(subs_basic_df), len(subs_std_df), len(subs_prem_df)], color=['blue', 'red', 'green'])
plt.title('Customer Type Subscription')
plt.xlabel("Type Subscription")
plt.xlabel("Number of Customer")
plt.show()


# Function to process data based on age group
def process_age_group(data, age_limit, age_condition='less'):
    if age_condition == 'less':
        filtered_data = data[data['age'] < age_limit]
    elif age_condition == 'greater':
        filtered_data = data[data['age'] > age_limit]
    else:
        raise ValueError("age_condition must be 'less' or 'greater'")

    unique_subscription_types = filtered_data['subscription_type'].unique()
    unique_contract_length = filtered_data['contract_length'].unique()
    total_spend = filtered_data['total_spend']

    total_list = []
    total_month = []
    total_annual = []
    total_quarterly = []

    for subscription_type in unique_subscription_types:
        for contract_length in unique_contract_length:
            filtered_by_time = filtered_data[
                (filtered_data['subscription_type'] == subscription_type) &
                (filtered_data['contract_length'] == contract_length)
            ]
            print(f"Subscription type {subscription_type} - Contract length {contract_length} : {len(filtered_by_time)}")
            if contract_length == 'Monthly':
                total_month.append(len(filtered_by_time))
            elif contract_length == 'Annual':
                total_annual.append(len(filtered_by_time))
            elif contract_length == 'Quarterly':
                total_quarterly.append(len(filtered_by_time))
            else:
                print("Nothing")

    # Calculate the sums
    sum_month = sum(total_month)
    sum_annual = sum(total_annual)
    sum_quarterly = sum(total_quarterly)

    # Append the sums to total_list
    total_list.extend([sum_month, sum_annual, sum_quarterly])

    print(unique_subscription_types)
    print(unique_contract_length)  # Print all totals as a single list
    print(f"Total Monthly : {sum(total_month)}")
    print(f"Total Annual : {sum(total_annual)}")
    print(f"Total Quarterly : {sum(total_quarterly)}")
    print(f"Total 0_Data : {sum(total_month) + sum(total_annual) + sum(total_quarterly)}")
    print(f"Total list : {total_list}")
    print(f"Average spend : {np.average(total_spend)}")

    plt.bar(['Monthly', 'Annual', 'Quarterly'], total_list, color=['blue', 'green', 'red'])
    plt.title(f'Total contract length by type subs age {age_condition} {age_limit}')
    plt.xlabel("Contract length")
    plt.ylabel("Count")
    plt.show()

# Function to analyze the age group between 25 and 35, you can create a similar function or modify this one.
def process_age_range(data, age_min, age_max):
    filtered_data = data[(data['age'] >= age_min) & (data['age'] < age_max)]
    unique_subscription_types = filtered_data['subscription_type'].unique()
    unique_contract_length = filtered_data['contract_length'].unique()
    total_spend = filtered_data['total_spend']

    total_list = []
    total_month = []
    total_annual = []
    total_quarterly = []

    for subscription_type in unique_subscription_types:
        for contract_length in unique_contract_length:
            filtered_by_time = filtered_data[
                (filtered_data['subscription_type'] == subscription_type) &
                (filtered_data['contract_length'] == contract_length)
            ]
            print(f"Subscription type {subscription_type} - Contract length {contract_length} : {len(filtered_by_time)}")
            if contract_length == 'Monthly':
                total_month.append(len(filtered_by_time))
            elif contract_length == 'Annual':
                total_annual.append(len(filtered_by_time))
            elif contract_length == 'Quarterly':
                total_quarterly.append(len(filtered_by_time))
            else:
                print("Nothing")

    # Calculate the sums
    sum_month = sum(total_month)
    sum_annual = sum(total_annual)
    sum_quarterly = sum(total_quarterly)

    # Append the sums to total_list
    total_list.extend([sum_month, sum_annual, sum_quarterly])

    print(unique_subscription_types)
    print(unique_contract_length)  # Print all totals as a single list
    print(f"Total Monthly : {sum(total_month)}")
    print(f"Total Annual : {sum(total_annual)}")
    print(f"Total Quarterly : {sum(total_quarterly)}")
    print(f"Total 0_Data : {sum(total_month) + sum(total_annual) + sum(total_quarterly)}")
    print(f"Total list : {total_list}")
    print(f"Average spend : {np.average(total_spend)}")

    plt.bar(['Monthly', 'Annual', 'Quarterly'], total_list, color=['blue', 'green', 'red'])
    plt.title(f'Total contract length by type subs age between {age_min} and {age_max}')
    plt.xlabel("Contract length")
    plt.ylabel("Count")
    plt.show()


# print("========== Age less 25  ==========")
process_age_group(df, 25, 'less')
# print("========== Age between 25 - 35  ==========")
process_age_range(df, 25, 35)
# print("========== Age greater 35  ==========")
process_age_group(df, 35, 'greater')


print("============== Set Training data ================\n")

# Example preprocessing pipeline
numeric_features = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'total_spend', 'last_interaction']
categorical_features = ['gender', 'subscription_type', 'contract_length']

# Define transformers for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define your model pipeline including preprocessing and regressor
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Load your dataset (assuming it's in a DataFrame named df)
# Replace 'df' with your actual DataFrame variable name
# Example: df = pd.read_csv('your_dataset.csv')

# Separate features and target variable
X = df.drop(columns=['customer_id', 'churn'])
y = df['churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model pipeline on training data
model_pipeline.fit(X_train, y_train)

# Extimating regression coefficients and intercepts
regressor = model_pipeline.named_steps['regressor']
coefficients = regressor.coef_
intercept = regressor.intercept_

print(f'Regression coefficients: {coefficients}')
print(f'Intercept: {intercept}')

# Predicting on the test set
y_pred = model_pipeline.predict(X_test)

# Evaluation of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Model validation and tuning using Ridge and Lasso Regression
ridge = Ridge()
lasso = Lasso()

param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
lasso_cv = GridSearchCV(lasso, param_grid, cv=5)

# Ensure X_train is passed through the preprocessor
X_train_processed = model_pipeline.named_steps['preprocessor'].transform(X_train)

ridge_cv.fit(X_train_processed, y_train)
lasso_cv.fit(X_train_processed, y_train)

# Print best parameters found by GridSearchCV
print(f"Best parameters for Ridge: {ridge_cv.best_params_}")
print(f"Best parameters for Lasso: {lasso_cv.best_params_}")

# Predicting on the test set using Ridge and Lasso
X_test_processed = model_pipeline.named_steps['preprocessor'].transform(X_test)

ridge_pred = ridge_cv.predict(X_test_processed)
lasso_pred = lasso_cv.predict(X_test_processed)

# Evaluate Ridge and Lasso on the test set
ridge_mse = mean_squared_error(y_test, ridge_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)

ridge_r2 = r2_score(y_test, ridge_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

ridge_mae = mean_absolute_error(y_test, ridge_pred)
lasso_mae = mean_absolute_error(y_test, lasso_pred)

print("\nRidge Regression:")
print(f"Mean Squared Error: {ridge_mse}")
print(f"R^2 Score: {ridge_r2}")
print(f"Mean Absolute Error: {ridge_mae}")

print("\nLasso Regression:")
print(f"Mean Squared Error: {lasso_mse}")
print(f"R^2 Score: {lasso_r2}")
print(f"Mean Absolute Error: {lasso_mae}")


# Final model selection and interpretation
final_model = ridge_cv if ridge_r2 > lasso_r2 else lasso_cv
final_pred = final_model.predict(X_test_processed)
#
# # Interpretation of results
coefficients = final_model.best_estimator_.coef_
intercept = final_model.best_estimator_.intercept_
feature_names = preprocessor.transformers_[0][1].get_feature_names_out()
#
print("Model Coefficients : ")
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature} : {coef}")
print(f"Intercept : {intercept}")

# Visualizing actual vs predicted
plt.scatter(y_test, final_pred, alpha=0.5)
plt.plot([0, 1], [0, 1], '--', color='red')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()




