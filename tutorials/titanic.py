

from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas as pd

df = sns.load_dataset("titanic")

y = df["survived"]

features = ["pclass", "sex", "sibsp", "parch"]
X = pd.get_dummies(df[features])
X_test = pd.get_dummies(df[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
hyperparams = model.get_params()
# acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

output = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")