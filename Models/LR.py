
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


# Load the data
train_df = pd.read_csv('./Data/fraudTrain.csv')
test_df = pd.read_csv('./Data/fraudTest.csv')

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

train_df.groupby('is_fraud', group_keys=False).apply(lambda x: x.sample(frac=.5))
test_df.groupby('is_fraud', group_keys=False).apply(lambda x: x.sample(frac=.5))


# Drop the columns that are not needed
columns_to_drop = ['first', 'last', 'street', 'city', 'state', 'job', 'trans_num', 'dob', 'city_pop', 'unix_time']

print('Columns dropped: ', (columns_to_drop))

train_df = train_df.drop(columns_to_drop, axis=1)
test_df = test_df.drop(columns_to_drop, axis=1)


# Fix the date column

# Convert the date-time column to a datetime object
train_df['date'] = pd.to_datetime(train_df['trans_date_trans_time'])

# Extract the parts of the date
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['hour'] = train_df['date'].dt.hour
train_df['minute'] = train_df['date'].dt.minute
train_df['second'] = train_df['date'].dt.second

# Now you can drop the original date-time column
train_df = train_df.drop(['date', 'trans_date_trans_time'], axis=1)

# Convert the date-time column to a datetime object
test_df['date'] = pd.to_datetime(test_df['trans_date_trans_time'])

# Extract the parts of the date
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['hour'] = test_df['date'].dt.hour
test_df['minute'] = test_df['date'].dt.minute
test_df['second'] = test_df['date'].dt.second

# Now you can drop the original date-time column
test_df = test_df.drop(['date', 'trans_date_trans_time'], axis=1)


# Convert the categorical columns to numerical values

train_df['merchant'] = train_df['merchant'].astype('category')
train_df['merchant'] = train_df['merchant'].cat.codes

test_df['merchant'] = test_df['merchant'].astype('category')
test_df['merchant'] = test_df['merchant'].cat.codes

train_df['category'] = train_df['category'].astype('category')
train_df['category'] = train_df['category'].cat.codes

test_df['category'] = test_df['category'].astype('category')
test_df['category'] = test_df['category'].cat.codes


train_df['gender'].replace(['F', 'M'], [0, 1], inplace=True)
test_df['gender'].replace(['F', 'M'], [0, 1], inplace=True)

# train_df.head().to_csv('../Data/cleanedTrain.csv', index=False)
# test_df.head().to_csv('../Data/cleanedTest.csv', index=False)
len(train_df.columns), len(test_df.columns)


# split the data into X and y

X_train = train_df.drop(['is_fraud'], axis=1)
y_train = train_df['is_fraud']
X_test = test_df.drop(['is_fraud'], axis=1)
y_test = test_df['is_fraud']

print(X_train.head())
print(y_train)
print(X_test.head())
print(y_test)

model = LinearRegression()

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print(df)

# Metrics
print('Linear Regression')
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Mean Absolute Percentage Error:', mean_absolute_percentage_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))


