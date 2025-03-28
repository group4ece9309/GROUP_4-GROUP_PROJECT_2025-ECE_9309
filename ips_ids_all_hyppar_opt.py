import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

with open("/home/und3rd06012/Downloads/kdd.names", 'r') as f:
    _ = f.read()

col_raw = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

cols = [c.strip() for c in col_raw if c.strip()]
cols.append("target")

attack_map = {
    'normal': 'normal', 'anomaly': 'anomaly', 'back': 'dos', 'buffer_overflow': 'u2r', 'ftp_write': 'r2l',
    'guess_passwd': 'r2l', 'imap': 'r2l', 'ipsweep': 'probe', 'land': 'dos', 'loadmodule': 'u2r',
    'multihop': 'r2l', 'neptune': 'dos', 'nmap': 'probe', 'perl': 'u2r', 'phf': 'r2l', 'pod': 'dos',
    'portsweep': 'probe', 'rootkit': 'u2r', 'satan': 'probe', 'smurf': 'dos', 'spy': 'r2l',
    'teardrop': 'dos', 'warezclient': 'r2l', 'warezmaster': 'r2l'
}

df = pd.read_csv("/home/und3rd06012/Downloads/combined.txt", names=cols)
df["label"] = df["target"].apply(lambda x: attack_map.get(x.strip(), "unknown"))

num_cols = df._get_numeric_data().columns
cat_cols = list(set(df.columns) - set(num_cols))
cat_cols.remove("target")
cat_cols.remove("label")

df.dropna(axis=1, inplace=True)
selected = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
df_corr = df[selected]

plt.figure(figsize=(30, 22))
sns.heatmap(df_corr.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG').set_title('Correlation Heatmap', fontdict={'fontsize': 22}, pad=12)
plt.savefig('heatmap.png', dpi=800, bbox_inches='tight')
plt.show()

drop_cols = [
    'num_root', 'srv_serror_rate', 'srv_rerror_rate', 'dst_host_srv_serror_rate',
    'dst_host_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'dst_host_same_srv_rate'
]
df.drop(drop_cols, axis=1, inplace=True)

proto_map = {'icmp': 0, 'tcp': 1, 'udp': 2}
flag_map = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'SH': 5, 'S1': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9, 'OTH': 10}
df["protocol_type"] = df["protocol_type"].map(proto_map)
df["flag"] = df["flag"].map(flag_map)

df.drop(["service", "target"], axis=1, inplace=True)

y = df["label"]
X = df.drop(["label"], axis=1)

scaler_minmax = MinMaxScaler()
X = scaler_minmax.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model_gnb = GaussianNB()
model_gnb.fit(X_train, y_train)

model_dtc = DecisionTreeClassifier(criterion="entropy", max_depth=4)
model_dtc.fit(X_train, y_train)

model_rfc = RandomForestClassifier(n_estimators=30)
model_rfc.fit(X_train, y_train)

model_lr = LogisticRegression(max_iter=1200000)
model_lr.fit(X_train, y_train)

scaler_std = StandardScaler()
X_train_scaled = scaler_std.fit_transform(X_train)
X_test_scaled = scaler_std.transform(X_test)

model_gbc = GradientBoostingClassifier(random_state=0, n_estimators=100, learning_rate=0.1, max_depth=3)
model_gbc.fit(X_train_scaled, y_train)
y_train_gbc = model_gbc.predict(X_train_scaled)
y_test_gbc = model_gbc.predict(X_test_scaled)

acc_train = [
    model_gnb.score(X_train, y_train),
    model_dtc.score(X_train, y_train),
    model_rfc.score(X_train, y_train),
    model_lr.score(X_train, y_train),
    accuracy_score(y_train, y_train_gbc)
]

acc_test = [
    model_gnb.score(X_test, y_test),
    model_dtc.score(X_test, y_test),
    model_rfc.score(X_test, y_test),
    model_lr.score(X_test, y_test),
    accuracy_score(y_test, y_test_gbc)
]

print("Train Accuracy:")
for name, acc in zip(['GNB', 'DTC', 'RFC', 'LRC', 'GBC'], acc_train):
    print(f"{name}: {acc:.4f}")

print("\nTest Accuracy:")
for name, acc in zip(['GNB', 'DTC', 'RFC', 'LRC', 'GBC'], acc_test):
    print(f"{name}: {acc:.4f}")

model_names = ['GNB', 'DTC', 'RFC', 'LRC', 'GBC']

plt.figure(figsize=(15, 3))
plt.bar(model_names, acc_train)
plt.title("Train Accuracy")
plt.ylim(0.8, 1.01)
plt.show()

plt.figure(figsize=(15, 3))
plt.bar(model_names, acc_test)
plt.title("Test Accuracy")
plt.ylim(0.8, 1.01)
plt.show()
