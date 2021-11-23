#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("marketing_campaign.csv")
data.head(50)


# In[3]:


data['Purchases'] = data['NumDealsPurchases'] + data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']
data 


# In[4]:


data['Expenses'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']
data


# In[5]:


data['Campaign'] = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + data['AcceptedCmp4'] + data['AcceptedCmp5']
data.head(50)


# In[6]:


data=data.drop("MntWines", axis=1)
data=data.drop("MntFruits", axis=1)
data=data.drop("MntMeatProducts", axis=1)
data=data.drop("MntFishProducts", axis=1)
data=data.drop("MntSweetProducts", axis=1)
data=data.drop("MntGoldProds", axis=1)
data=data.drop("AcceptedCmp1", axis=1)
data=data.drop("AcceptedCmp2", axis=1)
data=data.drop("AcceptedCmp3", axis=1)
data=data.drop("AcceptedCmp4", axis=1)
data=data.drop("AcceptedCmp5", axis=1)


# In[7]:


data


# In[8]:


data=data.drop("ID", axis=1)
data=data.drop("NumWebVisitsMonth", axis=1)
data=data.drop("NumDealsPurchases", axis=1)
data=data.drop("NumWebPurchases", axis=1)
data=data.drop("NumCatalogPurchases", axis=1)
data=data.drop("Z_CostContact", axis=1)
data=data.drop("Z_Revenue", axis=1)
data=data.drop("NumStorePurchases", axis=1)
data=data.drop("Dt_Customer", axis=1)


# In[9]:


data['Income'] = data['Income'].replace(np.NaN, data['Income'].mean())
data=data.assign(Incomes=pd.cut(data['Income'], 
                               bins=[ 0, 25000, 50000,100000,666666], 
                               labels=['Below 25000', 'Income 25000-50000 ', 'Income 50000-100000 ','Above 100000']))
data=data.drop("Income", axis=1)


# In[10]:


data['Expenses'] = data['Expenses'].replace(np.NaN, data['Expenses'].mean())
data=data.assign(Expense=pd.cut(data['Expenses'], 
                               bins=[ 0, 500, 1000, 2525], 
                               labels=['Below 500', 'Expense 500-1000 ','Above 1000']))
data=data.drop("Expenses", axis=1)


# In[11]:


data['Year_Birth'] = data['Year_Birth'].replace(np.NaN, data['Year_Birth'].mean())
data[['Year_Birth']]=data['Year_Birth']
data=data.assign(DOB=pd.cut(data['Year_Birth'], 
                               bins=[ 0, 1959, 1977, 1996], 
                               labels=['Below 1959', 'DOB 1959-1977', 'DOB 1977-1996']))
data=data.drop("Year_Birth", axis=1)


# In[12]:


data['Marital_Status'] = data['Marital_Status'].replace(['Married', 'Together'], 'relationship')
data['Marital_Status'] = data['Marital_Status'].replace(['Single', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'], 'single')

data['Education'] = data['Education'].replace(['2n Cycle', 'Basic'], 'Basic')
data['Education'] = data['Education'].replace(['Graduation', 'Master'], 'Graduated')
data['Education'] = data['Education'].replace(['PhD'], 'PHD')


# In[13]:


data.head(5)


# In[14]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[15]:


data['Education']= label_encoder.fit_transform(data['Education'])
data['Marital_Status']= label_encoder.fit_transform(data['Marital_Status'])
data['Incomes']= label_encoder.fit_transform(data['Incomes'])
data['DOB']= label_encoder.fit_transform(data['DOB'])
data['Expense']= label_encoder.fit_transform(data['Expense'])


# In[16]:


data.head(5)


# In[17]:


from sklearn.preprocessing import normalize
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
data_scaled.head()


# In[18]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[19]:


hc=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage="ward") 


# In[20]:


y_hc=hc.fit_predict(data_scaled)
clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[21]:


data['Cluster_id']=hc.labels_


# In[22]:


data.groupby("Cluster_id").agg(['mean']).reset_index()


# In[23]:


data


# In[24]:


X = data.drop("Cluster_id", axis=1)
y = data.Cluster_id
X.shape, y.shape


# In[25]:


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size = 0.2, random_state = 10)


# In[26]:


from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(max_depth=4, random_state = 10) 
model.fit(x_train, y_train)


# In[27]:


from sklearn.metrics import accuracy_score
pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)


# In[28]:


pred_train = model.predict(x_train)
accuracy_score(y_train,pred_train)


# In[29]:


# saving the model 
import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()


# In[33]:


get_ipython().run_cell_magic('writefile', 'mod.py', ' \nimport pickle\nimport streamlit as st\n \n# loading the trained model\npickle_in = open(\'classifier.pkl\', \'rb\') \nclassifier = pickle.load(pickle_in)\n \n@st.cache()\n\n# defining the function which will make the prediction using the data which the user inputs \ndef prediction(Education, Marital_Status, DOB,Incomes,Kidhome,Teenhome,Purchases,Expense,Recency,Campaign,Complain, Response):   \n \n    # Pre-processing user input    \n    \n    if Education == "Basic":\n        Education = 0\n        \n    elif Education == "Graduated":\n        Education = 1\n        \n    elif Education == "PHD":\n        Education = 2\n#*****************************************#      \n    if Marital_Status == "Single":\n        Marital_Status = 0\n    \n    elif Marital_Status == "Relationship":\n        Marital_Status = 1\n#*****************************************#        \n    if Incomes == "Below 25000":\n        Incomes = 1\n    \n    elif Incomes == "Income 25000-50000":\n        Incomes = 2\n        \n    elif Incomes == "Income 50000-100000":\n        Incomes = 3\n        \n    elif Incomes == "Above 100000":\n        Incomes = 0\n        \n\n#*****************************************# \n\n    if Campaign == "Accepted 0 Campaign":\n        Campaign = 0\n    \n    elif Campaign == "Accepted 1 Campaign":\n        Campaign = 1\n        \n    elif Campaign == "Accepted 2 Campaign":\n        Campaign = 2\n        \n    elif Campaign == "Accepted 3 Campaign":\n        Campaign = 3  \n        \n    elif Campaign == "Accepted 4 Campaign":\n        Campaign = 4\n        \n\n\n#*****************************************#    \n    if Response == "YES":\n        Response = 1\n    \n    elif Response == "NO":\n        Response = 0\n\n#*****************************************#      \n    if Complain == "YES":\n        Complain = 1\n    \n    elif Complain == "NO":\n        Complain = 0\n  #*****************************************#     \n\n  #*****************************************# \n        \n        \n    prediction = classifier.predict( \n        [[Education, Marital_Status,DOB, Incomes, Kidhome,Teenhome,Purchases,Expense,Recency,Campaign,Complain, Response]])\n            \n    if prediction == 0:\n        pred = \'cluster 0\'\n   \n    elif prediction == 1:\n        pred = \'cluster 1\'\n    \n    elif prediction == 2:\n        pred = \'cluster 2\'\n    return pred\n   \n      \n  \n# this is the main function in which we define our webpage  \ndef main():       \n    # front end elements of the web page \n    html_temp = """ \n    <div style ="background-color:Orange;padding:13px"> \n    <h1 style ="color:black;text-align:center;">Model Deployment</h1> \n    </div> \n    """\n      \n    # display the front end aspect\n    st.markdown(html_temp, unsafe_allow_html = True) \n    \n    # following lines create boxes in which user can enter data required to make prediction \n    \n    Education = st.selectbox("Education",("Basic","Graduated","PHD"))\n    \n    Marital_Status = st.radio("Marital_Status: ", (\'Single\', \'Relationship\'))\n    if (Marital_Status == \'Single\'):\n        st.success("Single")\n    elif (Marital_Status == \'Relationship\'):\n        st.success("Relationship")\n    \n    DOB = st.slider("Select DOB", 1930, 2021)\n    st.text(\'Selected: {}\'.format(DOB)) \n    \n    Incomes = st.selectbox("Incomes",("Below 25000", "Income 25000-50000", "Income 50000-100000","Above 100000")) \n   \n    Kidhome = st.text_input("Kidhome")\n    \n    Teenhome = st.text_input("Teenhome") \n    \n    Purchases= st.slider("NUmber of Purchase Made", 0, 50)\n    st.text(\'Selected: {}\'.format(Purchases)) \n    \n    Expense = st.slider("Select Monthly Expense", 0, 3000)\n    st.text(\'Selected: {}\'.format(Expense)) \n    \n    Recency= st.slider("last Purchase", 0, 100)\n    st.text(\'Selected: {}\'.format(Recency)) \n\n    Campaign =st.selectbox("Campaign",("Accepted 0 Campaign","Accepted 1 Campaign","Accepted 2 Campaign","Accepted 3 Campaign","Accepted 4 Campaign"))\n    \n    Complain = st.selectbox("Complain",("YES","NO"))\n    \n    Response = st.selectbox("Accepted the offer in the last campaign",("YES","NO"))\n    \n    result =""\n          \n    # when \'Predict\' is clicked, make the prediction and store it \n    if st.button("Predict"): \n        result = prediction(Education, Marital_Status,DOB, Incomes, Kidhome,Teenhome,Purchases,Expense,Recency, Campaign,Complain, Response) \n        st.success(\'Common cluster is {}\'.format(result))\n   \n     \nif __name__==\'__main__\': \n    main()')


# In[ ]:





# In[ ]:




