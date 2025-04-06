import matplotlib
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

matplotlib.use('TkAgg')

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plt.figure(figsize=(6, 5))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm')
plt.title("Data make_moons")
plt.show()

tree_configs = [
    ('entropy', 2),
    ('gini', 3),
    ('entropy', 5)
]

trees = []

for criterion, depth in tree_configs:
    dtc = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    trees.append((dtc, f'{criterion}, depth={depth}'))

plt.figure(figsize=(20, 5))

for i, (dtc, title) in enumerate(trees):
    plt.subplot(1, 3, i + 1)
    plot_tree(dtc, filled=True, fontsize=8)
    plt.title(f"Tree: {title}")

plt.tight_layout()
plt.show()

n_trees_list = [5, 20, 100]
rf_models = []

for n in n_trees_list:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_models.append((rf, n))

plt.figure(figsize=(20, 6))

for i, (rf, n) in enumerate(rf_models):
    plt.subplot(1, 3, i + 1)
    estimator = rf.estimators_[0]
    plot_tree(
        estimator,
        filled=True,
        max_depth=3,
        fontsize=8
    )
    plt.title(f"Random Forest with {n} trees — tree №1")

plt.tight_layout()
plt.show()

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)


svm = SVC(kernel='rbf', gamma='scale')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)


X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_decision_regions(X_combined, y_combined, clf=log_reg, legend=2)
plt.title("Logistic Regression")

plt.subplot(1, 2, 2)
plot_decision_regions(X_combined, y_combined, clf=svm, legend=2)
plt.title("Support Vector Machine")

plt.tight_layout()
plt.show()

voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_reg),
        ('svc', svm),
        ('rf', rf_models[0][0])
    ],
    voting='hard'
)

voting_clf.fit(X_train, y_train)

y_pred_voting = voting_clf.predict(X_test)


plt.figure(figsize=(6, 5))
plot_decision_regions(np.vstack((X_train, X_test)), np.hstack((y_train, y_test)), clf=voting_clf, legend=2)
plt.title("Voting Classifier")
plt.show()


def evaluate_classifier(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    }


best_tree = max(trees, key=lambda x: f1_score(y_test, x[0].predict(X_test)))
best_tree_model = best_tree[0]

best_forest = max(rf_models, key=lambda x: f1_score(y_test, x[0].predict(X_test)))
best_forest_model = best_forest[0]

results = []

results.append(evaluate_classifier("Decision Tree", best_tree_model, X_test, y_test))
results.append(evaluate_classifier("Random Forest", best_forest_model, X_test, y_test))
results.append(evaluate_classifier("Logistic Regression", log_reg, X_test, y_test))
results.append(evaluate_classifier("SVM", svm, X_test, y_test))
results.append(evaluate_classifier("Voting Classifier", voting_clf, X_test, y_test))

df_results = pd.DataFrame(results)
df_results.set_index('Model', inplace=True)
df_results = df_results.round(2)

plt.figure(figsize=(10, 6))
df_results.plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.title("Accuracy, Precision, Recall, F1-score")
plt.ylabel("Score")
plt.ylim(0.7, 1.0)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
