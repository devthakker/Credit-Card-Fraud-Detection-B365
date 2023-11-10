
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the data
train_df = pd.read_csv('fraudTrain.csv')
test_df = pd.read_csv('fraudTest.csv')

# Preprocess the data
le = LabelEncoder()

train_df['category'] = le.fit_transform(train_df['category'])
test_df['category'] = le.transform(test_df['category'])

# Fit and transform the column in both train and test dataset
train_df['merchant'] = le.fit_transform(train_df['merchant'])
test_df['merchant'] = le.transform(test_df['merchant'])

columns_to_drop = ['first', 'last', 'street', 'city', 'state', 'job', 'trans_num', 'dob', 'city_pop', 'unix_time', 'gender']

print('Columns dropped: ', len(columns_to_drop))

train_df = train_df.drop(columns_to_drop, axis=1)
test_df = test_df.drop(columns_to_drop, axis=1)



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

# Split the data into features and target
X_train = train_df.drop(['is_fraud'], axis=1)
y_train = train_df['is_fraud']
X_test = test_df.drop(['is_fraud'], axis=1)
y_test = test_df['is_fraud']

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Print out all metrics
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
