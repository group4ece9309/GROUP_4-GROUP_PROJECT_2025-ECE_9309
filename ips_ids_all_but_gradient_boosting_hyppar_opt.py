import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,
                                                    stratify=y if y.nunique().iloc[0] > 1 else None)


def objective_dt(trial):
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 5)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 3)

    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    return accuracy_score(y_test, model.predict(X_test))



def objective_rf(trial):
    n_estimators = trial.suggest_int("n_estimators", 30, 50)
    max_depth = trial.suggest_int("max_depth", 5, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 5)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 3)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train.values.ravel())
    return accuracy_score(y_test, model.predict(X_test))



def objective_gnb(trial):
    var_smoothing = trial.suggest_float("var_smoothing", 1e-9, 1e-6, log=True)

    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train, y_train.values.ravel())
    return accuracy_score(y_test, model.predict(X_test))



def objective_lr(trial):
    C = trial.suggest_float("C", 0.01, 10, log=True)
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])

    model = LogisticRegression(C=C, solver=solver, max_iter=1000)
    model.fit(X_train, y_train.values.ravel())
    return accuracy_score(y_test, model.predict(X_test))


# Run Optuna Optimization (Minimal Computation)
study_dt = optuna.create_study(direction="maximize")
study_dt.optimize(objective_dt, n_trials=3)

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=3)

study_gnb = optuna.create_study(direction="maximize")
study_gnb.optimize(objective_gnb, n_trials=3)

study_lr = optuna.create_study(direction="maximize")
study_lr.optimize(objective_lr, n_trials=3)


best_models = {
    "Gaussian Naive Bayes": GaussianNB(**study_gnb.best_params),
    "Logistic Regression": LogisticRegression(**study_lr.best_params, max_iter=1000),
    "Optimized Decision Tree": DecisionTreeClassifier(**study_dt.best_params, random_state=42),
    "Optimized Random Forest": RandomForestClassifier(**study_rf.best_params, random_state=42),
    "Gradient Boosting (Default)": GradientBoostingClassifier(  # No tuning, uses predefined settings
        n_estimators=50,
        learning_rate=0.1,
        max_depth=2,
        random_state=42
    ),
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
