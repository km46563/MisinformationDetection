import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("preprocessed_data.csv")

data = data.dropna(subset=["text", "label"])
print(data.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.25
)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

print(accuracy_score(y_train, tree.predict(X_train)))
print(accuracy_score(y_test, tree.predict(X_test)))
