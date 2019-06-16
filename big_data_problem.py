
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , classification_report
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[2]:


train = pd.read_csv("train.csv")


# In[3]:


building_ownership = pd.read_csv("building_ownership_Use.csv")


# In[4]:


building_structure = pd.read_csv("building_structure.csv")


# In[5]:


test = pd.read_csv("test.csv")


# In[6]:


result = pd.merge(building_structure, building_ownership, on='building_id')
data=pd.merge(train,result,on="building_id")


# In[7]:


result = pd.merge(building_structure, building_ownership, on='building_id')
test=pd.merge(test,result,on="building_id")


# In[8]:


v=pd.get_dummies(data['area_assesed'])
data=pd.concat([data,v],axis=1)
v=pd.get_dummies(test['area_assesed'])
test=pd.concat([test,v],axis=1)


# In[9]:


v=pd.get_dummies(data['land_surface_condition'])
data=pd.concat([data,v],axis=1)
v=pd.get_dummies(test['land_surface_condition'])
test=pd.concat([test,v],axis=1)


# In[10]:


v=pd.get_dummies(data['foundation_type'])
data=pd.concat([data,v],axis=1)
v=pd.get_dummies(test['foundation_type'])
test=pd.concat([test,v],axis=1)


# In[11]:


v=pd.get_dummies(data['roof_type'])
data=pd.concat([data,v],axis=1)
v=pd.get_dummies(test['roof_type'])
test=pd.concat([test,v],axis=1)


# In[12]:


v=pd.get_dummies(data['condition_post_eq'])
data=pd.concat([data,v],axis=1)
v=pd.get_dummies(test['condition_post_eq'])
test=pd.concat([test,v],axis=1)


# In[13]:


v=pd.get_dummies(data['ground_floor_type'])
data=pd.concat([data,v],axis=1)
v=pd.get_dummies(test['ground_floor_type'])
test=pd.concat([test,v],axis=1)


# In[14]:


v=pd.get_dummies(data['other_floor_type'])
data=pd.concat([data,v],axis=1)
v=pd.get_dummies(test['other_floor_type'])
test=pd.concat([test,v],axis=1)


# In[15]:


v=pd.get_dummies(data['position'])
data=pd.concat([data,v],axis=1)
v=pd.get_dummies(test['position'])
test=pd.concat([test,v],axis=1)


# In[16]:


v=pd.get_dummies(data['plan_configuration'])
data=pd.concat([data,v],axis=1)
v=pd.get_dummies(test['plan_configuration'])
test=pd.concat([test,v],axis=1)


# In[17]:


data = data.drop(['building_id','has_geotechnical_risk_liquefaction','has_geotechnical_risk_other','has_secondary_use_health_post','has_secondary_use_gov_office','has_secondary_use_use_police','legal_ownership_status','plan_configuration','position','other_floor_type','ground_floor_type','condition_post_eq','roof_type','foundation_type','land_surface_condition','count_families','area_assesed','has_repair_started'],axis=1)


# In[18]:


test = test.drop(['building_id','has_geotechnical_risk_liquefaction','has_geotechnical_risk_other','has_secondary_use_health_post','has_secondary_use_gov_office','has_secondary_use_use_police','legal_ownership_status','plan_configuration','position','other_floor_type','ground_floor_type','condition_post_eq','roof_type','foundation_type','land_surface_condition','area_assesed','area_assesed','has_repair_started','count_families'],axis=1)


# In[19]:


data['damage']=data.damage_grade.map({'Grade 5':5,'Grade 4':4,'Grade 3':3,'Grade 2':2,'Grade 1':1})


# In[20]:


data = data.drop(["damage_grade"],axis=1)


# In[21]:


x = data.iloc[:,:-1]
y= data.iloc[:,-1]


# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.7,random_state = 0)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score as cs


# In[27]:


knn = KNeighborsClassifier(n_neighbors = 18 )
score1=cs(knn,x,y,cv=10,scoring='accuracy')
score1


# In[28]:


score1.max()


# In[29]:



classifier_knn = KNeighborsClassifier(n_neighbors = 13 , metric = "minkowski",p=2)
classifier_knn.fit(x_train,y_train)


# In[ ]:


y_pred_knn = classifier_knn.predict(test)
accuracy_knn_diabetes = accuracy_score(y_pred_knn,test)

