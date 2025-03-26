import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

file_path = "/home/und3rd06012/Downloads/kdd.names"

with open(file_path, 'r') as file:
    file_contents = file.read()

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

columns = []
for c in initialization_columns:
    if c.strip():  # Check if the column name is not an empty string
        columns.append(c.strip())

columns.append('target')

attack_types = dict([
    ('normal', 'normal'), ('anomaly', 'anomaly'), ('back', 'dos'),
    ('buffer_overflow', 'u2r'), ('ftp_write', 'r2l'), ('guess_passwd', 'r2l'),
    ('imap', 'r2l'), ('ipsweep', 'probe'), ('land', 'dos'), ('loadmodule', 'u2r'),
    ('multihop', 'r2l'), ('neptune', 'dos'), ('nmap', 'probe'), ('perl', 'u2r'),
    ('phf', 'r2l'), ('pod', 'dos'), ('portsweep', 'probe'), ('rootkit', 'u2r'),
    ('satan', 'probe'), ('smurf', 'dos'), ('spy', 'r2l'), ('teardrop', 'dos'),
    ('warezclient', 'r2l'), ('warezmaster', 'r2l')
])


combined_data = pd.read_csv("/home/und3rd06012/Downloads/combined.txt", names = columns)

combined_data['Attack Type'] = combined_data['target'].apply(lambda r: attack_types.get(r.strip(), 'unknown'))

number_of_columns = combined_data._get_numeric_data().columns

cate_cols = list(set(combined_data.columns)-set(number_of_columns))
cate_cols.remove('target')
cate_cols.remove('Attack Type')

combined_data = combined_data.dropna(axis='columns')

selected_columns = []
for col in combined_data.columns:
    if pd.api.types.is_numeric_dtype(combined_data[col]) and combined_data[col].nunique() > 1:
        selected_columns.append(col)

excluded_columns_combined_data = combined_data[selected_columns]

fig, ax = plt.subplots(figsize =(30, 22))
heatmap = sns.heatmap(excluded_columns_combined_data.corr(), ax=ax, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':22}, pad=12);
plt.savefig('heatmap.png', dpi=800, bbox_inches='tight')
plt.show()

combined_data.drop('num_root', axis = 1, inplace = True)
combined_data.drop('srv_serror_rate', axis = 1, inplace = True)
combined_data.drop('srv_rerror_rate', axis = 1, inplace = True)
combined_data.drop('dst_host_srv_serror_rate', axis = 1, inplace = True)
combined_data.drop('dst_host_serror_rate', axis = 1, inplace = True)
combined_data.drop('dst_host_rerror_rate', axis = 1, inplace = True)
combined_data.drop('dst_host_srv_rerror_rate', axis = 1, inplace = True)
combined_data.drop('dst_host_same_srv_rate', axis = 1, inplace = True)

pmap = {'icmp':0, 'tcp':1, 'udp':2}
combined_data['protocol_type'] = combined_data['protocol_type'].map(pmap)

fmap = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10}
combined_data['flag'] = combined_data['flag'].map(fmap)

combined_data.drop('service', axis = 1, inplace = True)
combined_data = combined_data.drop(['target', ], axis = 1)
#print(combined_data.shape)

y = combined_data[['Attack Type']]
X = combined_data.drop(['Attack Type', ], axis = 1)

sc = MinMaxScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
#print(X_train.shape, X_test.shape)
#print(y_train.shape, y_test.shape)

GNB = GaussianNB()
GNB.fit(X_train, y_train.values.ravel())
y_test_pred = GNB.predict(X_train)
print("Gaussian Naive Bayes(train score):", GNB.score(X_train, y_train))
print("Gaussian Naive Bayes (test score):", GNB.score(X_test, y_test))

DTC = DecisionTreeClassifier(criterion ="entropy", max_depth = 4)
DTC.fit(X_train, y_train.values.ravel())
y_test_pred = DTC.predict(X_train)
print("Decision Tree Classifier (train score):", DTC.score(X_train, y_train))
print("Decision Tree Classifier (test score):", DTC.score(X_test, y_test))

RFC = RandomForestClassifier(n_estimators = 30)
RFC.fit(X_train, y_train.values.ravel())
y_test_pred = RFC.predict(X_train)
print("Random Forest Classifier (train score):", RFC.score(X_train, y_train))
print("Random Forest Classifier (test score):", RFC.score(X_test, y_test))

LR = LogisticRegression(max_iter = 1200000)
LR.fit(X_train, y_train.values.ravel())
y_test_pred = LR.predict(X_train)
print("Logistic Forest (train score):", LR.score(X_train, y_train))
print("Logistic Forest (test score)", LR.score(X_test, y_test))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

GBC = GradientBoostingClassifier(random_state=0, n_estimators=100, learning_rate=0.1, max_depth=3)

GBC.fit(X_train_scaled, y_train.values.ravel())

y_train_pred = GBC.predict(X_train_scaled)
y_test_pred = GBC.predict(X_test_scaled)

print("Gradient Boost Classifier (train accuracy):", accuracy_score(y_train, y_train_pred))
print("Gradient Boost Classifier (test accuracy):", accuracy_score(y_test, y_test_pred))


names = ['GNB', 'DTC', 'RFC', 'LRC', 'GBC']
values = [87.951, 99.058, 99.997, 99.352, 99.793]
f = plt.figure(figsize =(15, 3), num = 10)

plt.subplot(131)
plt.bar(names, values)
plt.show()
plt.close(10)

names = ['NB', 'DT', 'RF', 'LR', 'GB']
values = [87.903, 99.052, 99.969, 99.352, 99.771]
f = plt.figure(figsize =(15, 3), num = 10)

plt.subplot(131)
plt.bar(names, values)
plt.show()
plt.close(10)

names = ['NB', 'DT', 'RF', 'SVM', 'LR', 'GB']
values = [1.54329, 0.14877, 0.199471, 126.50875, 0.09605, 2.95039]
f = plt.figure(figsize =(15, 3), num = 10)

plt.subplot(131)
plt.bar(names, values)
plt.show()

plt.close(10)