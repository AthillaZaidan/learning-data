import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train_file_path = 'data/train.csv'
test_file_path = 'data/test.csv'


train_data = pd.read_csv(train_file_path) 
test_data = pd.read_csv(test_file_path)

y = train_data["Survived"] # prediction target
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

titanic_model = RandomForestClassifier(random_state= 1)
titanic_model.fit (train_X, train_y)

titanic_pred = titanic_model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': titanic_pred})
output.to_csv('submission.csv', index=False)

print("success")