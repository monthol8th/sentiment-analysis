#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import sys
# !{sys.executable} -m pip install pythainlp emoji


# In[2]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
sns.set(style="darkgrid")


# In[3]:


from pythainlp.tokenize import word_tokenize


# In[4]:


df = pd.read_csv("sentiment.tsv",sep='\t')


# In[5]:


df['token']=df['text'].apply(lambda x:word_tokenize(x))


# In[6]:


from pythainlp.spell import correct
import re


# In[7]:


import emoji

split_emoji = emoji.get_emoji_regexp()


# In[8]:


df['token']=df['token'].apply(lambda x: [a for y in x for a in split_emoji.split(y) if not bool(re.search(r'^\s*$',a))])


# In[9]:


df['token'].loc[8]


# In[10]:


# df['token']=df['token'].apply(lambda x: [t for t in x if (not bool(re.search(r'^([ก-ฮ])\1*$',t)))] if len(x)>1 else )


# In[11]:


# df[df['token'].apply(lambda x:len(x)<=1)]


# In[12]:


# df[df['token'].apply(lambda x:len(x)<=1)]


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


train,test = train_test_split(df,test_size=0.15,random_state=69)


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[16]:


import identity_tokenizer

tfidf = TfidfVectorizer(tokenizer=identity_tokenizer.identity_tokenizer,lowercase=False,max_df=0.9,min_df=10,ngram_range=(1,2))


# In[17]:


X_train = tfidf.fit_transform(train['token'].values.tolist()).toarray()


# In[18]:


y_train = train['label'].values


# In[19]:


X_train.shape


# In[20]:


X_test = tfidf.transform(test['token'].values.tolist()).toarray()
y_test = test['label'].values


# In[21]:


from sklearn.metrics import accuracy_score


# In[22]:


from sklearn.metrics import f1_score


# In[23]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

#fit logistic regression models
params = {
    'C':[0.5,1,1.5,2,2.5,3],
    'penalty':['l1','l2']
}
classifier = GridSearchCV(LogisticRegression(solver='liblinear',multi_class='ovr',random_state=69),param_grid=params,scoring='f1_weighted',cv=3)
classifier.fit(X_train,y_train)


# In[24]:


classifier.best_params_


# In[25]:


y_pred = classifier.predict(X_test)


# In[26]:


print("acc:",accuracy_score(y_test,y_pred))
print("f1:",f1_score(y_test,y_pred,average='weighted'))


# In[27]:


from sklearn.metrics import precision_recall_fscore_support

print(precision_recall_fscore_support(y_test,y_pred,average='weighted'))


# In[28]:


from sklearn.metrics import confusion_matrix

mat = confusion_matrix(y_test,y_pred,labels=[-1,0,1])


# In[29]:


sns.heatmap(mat.astype(int), annot=True,fmt='d',xticklabels=[-1,0,1],yticklabels=[-1,0,1])


# In[30]:


print(precision_recall_fscore_support(y_test,y_pred))


# In[32]:


import pickle

with open('model.pkl', 'wb') as model_file:
    pickle.dump(classifier,model_file)

with open('vectorize.pkl', 'wb') as vectorize_file:
    pickle.dump(tfidf,vectorize_file)


# In[33]:


test['predict'] = y_pred


# In[34]:


test[['text','label','predict']].to_csv('solution.tsv',sep='\t',index=None)


# In[ ]:




