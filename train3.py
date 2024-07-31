import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import joblib

# Load data
df = pd.read_csv("Train.csv")
df.drop('id', axis=1, inplace=True)
# Define lists for label encoding and missing values
col_to_le = []
col_to_miss = []

def eda(x, df, miss, le):
    if df[x].dtype == 'object':
        le.append(x)
    if df[x].isna().sum() > 0:
        miss.append(x)

# Identify columns to handle
for i in df.columns:
    eda(i, df, col_to_miss, col_to_le)

# Handle missing values
for i in col_to_miss:
    if df[i].isna().sum() > (0.8 * df.shape[0]):
        df.drop(i, axis=1, inplace=True)
        col_to_miss.remove(i)
    else:
        # Custom handling for specific columns
        if i == 'customer_age':
            df[i].fillna(df[i].median(), inplace=True)
        elif i == 'marital':
            df[i].fillna('missing', inplace=True)
        elif i == 'balance':
            df[i].fillna(0, inplace=True)
        elif i == 'personal_loan':
            df[i].fillna('no', inplace=True)
        elif i == 'last_contact_duration':
            df[i].fillna(0, inplace=True)
        elif i == 'num_contacts_in_campaign':
            df[i].fillna(0, inplace=True)
            df[i] = df[i].round().astype(int)

# Drop unnecessary columns
df_n = df.copy()
df_n.drop(['month', 'prev_campaign_outcome', 'day_of_month'], axis=1, inplace=True)
pt = PowerTransformer(method='yeo-johnson')

for col in ['customer_age', 'balance', 'last_contact_duration']:
    if abs(df_n[col].skew()) > 0.8:
        df_n[col] = pt.fit_transform(df_n[col].values.reshape(-1, 1))
        with open(f'{col}_skew.pkl', 'wb') as f:
            pickle.dump(pt, f)
    print(f"Skewness for column {col} is {df_n[col].skew()}")
# Label encode specific columns
le = LabelEncoder()
for col in ['default', 'housing_loan', 'personal_loan']:
    df_n[col] = le.fit_transform(df_n[col])
    with open(f'{col}_le.pkl', 'wb') as f:
        pickle.dump(le, f)

# Ordinal encode specific columns
oe = OrdinalEncoder()
for col in ['job_type', 'marital', 'education', 'communication_type']:
    df_n[col] = oe.fit_transform(df_n[col].values.reshape(-1, 1))
    df_n[col] = df_n[col].astype(int)
    with open(f'{col}_oe.pkl', 'wb') as f:
        pickle.dump(oe, f)
df_n.drop([
        'num_contacts_in_campaign',
       'num_contacts_prev_campaign'],axis=1,inplace=True)
df_n.drop('term_deposit_subscribed',axis=1,inplace=True)        
scaler = StandardScaler()
df_std = scaler.fit_transform(df_n)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

joblib.dump(df_std, 'df_std.joblib')
pca = PCA(n_components=7)
scores_pca = pca.fit_transform(df_std)
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)
scores_pca_db = pd.DataFrame(scores_pca)
db = DBSCAN(eps=2, min_samples=5).fit(scores_pca_db)
labels = db.labels_
scores_pca_db['Cluster'] = labels

with open('db_model.pkl', 'wb') as f:
    pickle.dump(db, f)
silhouette_avg = silhouette_score(scores_pca_db, labels)
db_score = davies_bouldin_score(scores_pca_db, labels)
ch_score = calinski_harabasz_score(scores_pca_db, labels)

print(f'Silhouette Score: {silhouette_avg}')
print(f'Davies-Bouldin Score: {db_score}')
print(f'Calinski-Harabasz Score: {ch_score}')
df_new = pd.concat([df_n, scores_pca_db['Cluster']], axis=1)
for col in df_new.drop(['Cluster'], axis=1):
    grid = sns.FacetGrid(df_new, col='Cluster')
    grid.map(sns.histplot, col)
plt.show()
