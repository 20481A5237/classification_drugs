import pickle

import form as form
import pandas as pd
import sklearn
from sklearn.metrics import classification_report
#form sklearn.preprocessing
#import StandardScaler
#sc=StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv("drug200.csv")
df.replace({'Sex':{'F':0,'M':1}},inplace=True)
df.replace({'BP':{'HIGH':2,'NORMAL':1,'LOW':0}},inplace=True)
df.replace({'Cholesterol':{'HIGH':1,'NORMAL':0}},inplace = True)
#tx=pd.get_dummies(tr_x, columns = ['Destination'])
#df1=pd.get_dummies(df,columns=['Cholesterol'])
#df1=pd.get_dummies(df,columns=['Drug'])
#df.replace({'Drug':{'DrugY':0,'drugA':1,'drugB':2,'drugC':3,'drugX':4}},inplace=True)

tr=df[['Age','Sex','BP','Cholesterol','Na_to_K']]
te=df[['Drug']]

from sklearn.model_selection import train_test_split
r_x,e_x,r_y,e_y=train_test_split(tr,te,test_size=0.25)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(tr,te)
acc=rf.score(e_x,e_y)
y_pred=rf.predict(e_x)
print(classification_report(e_y,y_pred))

#pickles
pickle.dump(rf,open("model.pkl","wb"))


