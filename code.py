import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("C:/Users/moham/Downloads/train_LZdllcl.csv")
from sklearn import preprocessing
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

encoder.fit(df['department'])
df['department'] = encoder.transform(df['department'])

encoder.fit(df['region'])
df['region'] = encoder.transform(df['region'])

encoder.fit(df['education'])
df['education'] = encoder.transform(df['education'])

encoder.fit(df['gender'])
df['gender'] = encoder.transform(df['gender'])

encoder.fit(df['recruitment_channel'])
df['recruitment_channel'] = encoder.transform(df['recruitment_channel'])
print(df.head())

x = df.iloc[:,:-1]

print(x.head())

y = df.iloc[:,-1]


print(y.head())

cat = ['region','gender']
xf = pd.DataFrame(x.drop(columns=cat))

X_train, X_test, y_train, y_test = train_test_split(xf, y, test_size=0.2, random_state=42)

print(X_train.head())

model = xgb.XGBClassifier()

model.fit(X_train, y_train)

pred=model.predict(X_test)
print(pred)
cm = confusion_matrix(y_test,pred)
print(cm)
print(accuracy_score(y_test,pred))



df = pd.read_csv("C:/Users/moham/Downloads/test_2umaH9m.csv")

encoder = preprocessing.LabelEncoder()

encoder.fit(df['department'])
df['department'] = encoder.transform(df['department'])

encoder.fit(df['region'])
df['region'] = encoder.transform(df['region'])

encoder.fit(df['education'])
df['education'] = encoder.transform(df['education'])

encoder.fit(df['gender'])
df['gender'] = encoder.transform(df['gender'])

encoder.fit(df['recruitment_channel'])
df['recruitment_channel'] = encoder.transform(df['recruitment_channel'])
print(df.head())



cat = ['region','gender']
xf = pd.DataFrame(df.drop(columns=cat))
print("Test: ",xf)

pred=model.predict(xf)
import numpy as np
import pandas as pd
prediction = pd.DataFrame(pred, columns=['is_promoted']).to_csv('prediction.csv')

df1 = pd.read_csv("prediction.csv")
df2 = df1[['predictions']].copy()
print(df2.head())

df3 = df[['employee_id']].copy()
print(df3.head())

both = df3.join(df2)
print(both)

both.to_csv("predictions3.csv", index=False)