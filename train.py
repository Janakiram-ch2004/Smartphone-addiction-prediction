import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("smartphone_usage.csv")
df
print(df.shape)
df.describe()
df.drop(['transaction_id','user_id'],axis=1,inplace=True,errors='ignore')
print(df.isna().sum())
df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(df.shape)

num_col=df.select_dtypes(include=('int64','float64'))
num_col

cat_col=df.select_dtypes(include=('object'))
cat_col

for col in num_col:
  plt.figure(figsize=(5,6))
  sns.histplot(df[col],kde=True)
  #plt.show()

  num_col

  corri=df.corr(numeric_only=True)
corri

plt.figure(figsize=(10, 6))
sns.heatmap(corri, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
#plt.show

from sklearn.impute import SimpleImputer
si=SimpleImputer(strategy='most_frequent')
df[['addiction_level']]=si.fit_transform(df[['addiction_level']])

df.info()

print(df['addicted_label'].value_counts())

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,f_classif
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score ,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier as RandomForest

X = df.drop(['addicted_label','addiction_level'], axis=1)
y = df['addicted_label']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
                                               
catagorical_col=X_train.select_dtypes(include=('object'))

saved_encoder={}
for col in catagorical_col:
  le=LabelEncoder()
  X_train[col]=le.fit_transform(X_train[col])
  X_test[col]=le.transform(X_test[col])
  saved_encoder[col]=le
                         
original_features=X_train.columns

st =StandardScaler()
X_train = st.fit_transform(X_train)
X_test = st.transform(X_test)

selector=SelectKBest(f_classif,k=10)
X_train=selector.fit_transform(X_train,y_train)
X_test=selector.transform(X_test)

selected_features=selector.get_support(indices=True)
selected_feature_names=original_features[selected_features]
print(selected_feature_names)

smote=SMOTE(random_state=42)
X_train_resampled,y_train_resampled=smote.fit_resample(X_train,y_train)

model=RandomForest()
model.fit(X_train_resampled,y_train_resampled)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
classification=classification_report(y_test,y_pred)

print(accuracy)

print(classification)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(X_train_resampled,y_train_resampled)

y_pred2=model.predict(X_test)
accuracy2=accuracy_score(y_test,y_pred2)
confusion2=confusion_matrix(y_test,y_pred2)
classification2=classification_report(y_test,y_pred2)

print(accuracy2)

print(classification2)


import joblib

print("Model saving starts")


joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(saved_encoder, 'label_encoders.pkl')
joblib.dump(st, 'standard_scaler.pkl')
joblib.dump(selector, 'feature_selector.pkl')
joblib.dump(si, 'simple_imputer.pkl')

print("all files saved.")







