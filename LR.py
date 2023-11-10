import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
df_train = pd.read_csv('fraudTrain.csv')
df_test = pd.read_csv('fraudTest.csv')

# Drop unnecessary columns
columns_to_drop = [ 'merchant', 'category', 'first', 'last', 'street', 'city', 'state', 'job', 'trans_num', 'trans_date_trans_time', 'dob', 'city_pop', 'unix_time']

print('Columns dropped: ', len(columns_to_drop))

df_train = df_train.drop(columns_to_drop, axis=1)
df_test = df_test.drop(columns_to_drop, axis=1)

df_train['gender'] = df_train['gender'].apply(lambda x : 1 if x=='M' else 0)
df_test['gender'] = df_test['gender'].apply(lambda x : 1 if x=='M' else 0)

print(df_train.head())
print(df_test.head())

df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

# df_train['trans_date_trans_time']=pd.to_datetime(df_train['trans_date_trans_time'])
# df_train['trans_date']=df_train['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
# df_train['trans_date']=pd.to_datetime(df_train['trans_date'])    
# df_train['dob']=pd.to_datetime(df_train['dob'])

# df_test['trans_date_trans_time']=pd.to_datetime(df_test['trans_date_trans_time'])
# df_test['trans_date']=df_test['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
# df_test['trans_date']=pd.to_datetime(df_test['trans_date'])
# df_test['dob']=pd.to_datetime(df_test['dob'])

X_train = df_train.drop('is_fraud', axis=1)
y_train = df_train['is_fraud']
X_test = df_test.drop('is_fraud', axis=1)
y_test = df_test['is_fraud']

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

df_train_predict = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df_train_predict.to_csv('LR.csv')

print(df_train_predict)

# Evaluate the model on the testing data
score = lr.score(X_test, y_test)
print('-----------------------------------')
print(f"Model score: {score}")
print('Train Score: ', lr.score(X_train, y_train))
print('Intercept: ', lr.intercept_)
print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
print('R2 Score: ', r2_score(y_test, y_pred))
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))
print('Accuracy Score: ', accuracy_score(y_test, y_pred.round()))
print('-----------------------------------')


# plt.scatter(y_test, y_pred, color='grey', marker='o')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Actual vs Predicted')
# plt.show()
