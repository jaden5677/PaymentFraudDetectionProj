import sklearn

from sklearn import datasets

iris_data = datasets.load_iris()
digits = datasets.load_digits()

print(iris_data.data[0]) # Feature values for first sample
print(iris_data.target[0]) # Target value for first

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(iris_data.data)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(iris_data.data)

import matplotlib.pyplot as plt
plt.scatter(iris_data.data[:, 0], iris_data.data[:, 1], c=iris_data.target)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(scaled_data, iris_data.target)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(scaled_data, iris_data.target)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_data, iris_data.target)

from sklearn.metrics import classification_report

print(classification_report(y_test, model.predict(X_test)))

from sklearn.model_selection import GridSearchCV

params = {'C': [0.1, 1, 10]}
grid_search = GridSearchCV(model, params, cv=5)
grid_search.fit(scaled_data, iris_data.target)

from sklearn.model_selection import cross_val_score

cross_val_scores = cross_val_score(model, scaled_data, iris_data.target, cv=5)
cross_val_scores

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(scaled_data, iris_data.target)

from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[('lr', model), ('rf', random_forest)])
voting_clf.fit(scaled_data, iris_data.target)

# Train base models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

rf = RandomForestClassifier()
svc = SVC()

rf.fit(X_train, y_train)
svc.fit(X_train, y_train)

# Make predictions to train meta-model
rf_predictions = rf.predict(X_test)
svc_predictions = svc.predict(X_test)

# Create dataset for meta-model
blender = np.vstack((rf_predictions, svc_predictions)).T
blender_target = y_test

# Fit meta-model on predictions
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb.fit(blender, blender_target)

# Make final predictions
final_predictions = gb.predict(blender)

