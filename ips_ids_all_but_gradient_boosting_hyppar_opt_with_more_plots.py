import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve


file_path = "/home/und3rd06012/Downloads/combined.txt"

initialization_columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

columns = initialization_columns + ['target']


combined_data = pd.read_csv(file_path, names=columns)


combined_data = combined_data.sample(frac=0.5, random_state=42)


combined_data.dropna(axis='columns', inplace=True)
combined_data.drop(['service'], axis=1, inplace=True)

pmap = {'icmp': 0, 'tcp': 1, 'udp': 2}
combined_data['protocol_type'] = combined_data['protocol_type'].map(pmap)

fmap = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'SH': 5, 'S1': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9, 'OTH': 10}
combined_data['flag'] = combined_data['flag'].map(fmap)

X = combined_data.drop(['target'], axis=1)
y = combined_data[['target']]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y if y.nunique().iloc[0] > 1 else None)


plt.figure(figsize=(12, 5))
ax = sns.countplot(x=y.values.ravel(), hue=y.values.ravel(), palette="coolwarm", legend=False)
plt.title("Class Distribution in Dataset")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.show()


best_models = {
    "Gaussian Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Optimized Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42),
    "Optimized Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42),
    "Gradient Boosting (Default)": GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=42),
}

results = []
for name, model in best_models.items():
    model.fit(X_train, y_train.values.ravel())
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    results.append([name, train_acc, test_acc])
    print(f"{name} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")


results_df = pd.DataFrame(results, columns=["Model", "Train Accuracy", "Test Accuracy"])
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Test Accuracy", hue="Model", data=results_df, palette="viridis", legend=False)
plt.xticks(rotation=45, ha="right")
plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1)
plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


y_pred_rf = best_models["Optimized Random Forest"].predict(X_test)
plot_confusion_matrix(y_test, y_pred_rf, classes=np.unique(y))


rf_feature_importances = best_models["Optimized Random Forest"].feature_importances_
feature_names = combined_data.drop(['target'], axis=1).columns
feat_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": rf_feature_importances})
feat_importance_df = feat_importance_df.sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", hue="Feature", data=feat_importance_df, palette="coolwarm", legend=False)
plt.title("Top 10 Important Features (Random Forest)")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.show()


plt.figure(figsize=(8, 6))
for name, model in best_models.items():
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]  # Get probabilities
        fpr, tpr, _ = roc_curve(y_test, y_probs, pos_label=np.unique(y)[1])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


plt.figure(figsize=(8, 6))
for name, model in best_models.items():
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_probs, pos_label=np.unique(y)[1])
        plt.plot(recall, precision, label=name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
