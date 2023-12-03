#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


credits=pd.read_csv(r"C:\Users\ASUS\Downloads\tmdb_5000_credits.csv")


# In[3]:


movies=pd.read_csv(r"C:\Users\ASUS\Downloads\tmdb_5000_movies.csv")


# In[4]:


credits


# In[5]:


movies


# In[6]:


df=movies.merge(credits,on='title')


# In[7]:


df


# In[8]:


df.shape


# In[9]:


movies.shape


# In[10]:


credits.shape


# In[11]:


df.columns


# In[12]:


df.head(1)


# In[13]:


df.head(1)['crew'].values


# In[14]:


df.columns


# In[15]:


df['genres'].head(5).values


# In[16]:


df.info()


# In[17]:


df['homepage'].sample(2).values


# Required Columns for Recommendation 

# In[18]:


df=df[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[19]:


df


# In[20]:


df.head(1).values


# Preprocessing

# In[21]:


df.isnull().sum()


# In[22]:


droped=df[df['overview'].isnull()==True]


# In[23]:


df=df.dropna()


# In[24]:


df.duplicated().any()


# In[25]:


import json


# In[26]:


def convert(x):
    try:
        p=np.array(x)
        p=json.loads(p[0])
        g=[y['name'] for y in p]
        b=','.join(g)
        return b
    
    except json.JSONDecodeError:
        return None


# In[27]:


x=convert(df['genres'].head(1).values)


# In[28]:


type(x)


# In[29]:


import ast


# In[30]:


def convert2(x):
    list=[]
    for i in ast.literal_eval(x):
        list.append(i['name'])
    return list


# In[31]:


df['genres']=df['genres'].apply(convert2)


# In[32]:


df


# In[33]:


df['keywords'].sample(1).values


# In[34]:


def convert3(x):
    list=[]
    for i in ast.literal_eval(x):
        list.append(i['name'])
    return list


# In[35]:


df['keywords']=df['keywords'].apply(convert3)


# In[36]:


df


# In[37]:


df['keywords'].sample(1).values


# In[38]:


df['cast'].sample().values


# In[39]:


def convert4(x):
    list=[]
    c=0
    for i in ast.literal_eval(x):
        if c!=5:
            list.append(i['name'])
            c+=1
        else:
            break
    return list
        
        


# In[40]:


df['cast']=df['cast'].apply(convert4)


# In[41]:


df


# In[42]:


df['cast'].sample(1).values


# In[43]:


def convert5(x):
    list=[]
    for i in ast.literal_eval(x):
        if i['job']=='Director':
            list.append(i['name'])
            break
    return list


# In[44]:


def convert6(x):
    try:
        list_of_dicts = ast.literal_eval(x)
        director_dict = next((item for item in list_of_dicts if item.get('job') == 'Director'), None)
        director_name = director_dict.get('name') if director_dict else None
        return director_name
    except (SyntaxError, ValueError):
        return None


# In[45]:


df['crew']=df['crew'].apply(convert5)


# In[46]:


df


# In[47]:


df.rename(columns={'crew':'director'},inplace=True)


# In[48]:


df


# In[49]:


df['overview'].head(1).values


# In[50]:


df['overview']=df['overview'].apply(lambda x: x.split())


# In[51]:


df


# In[52]:


df['genres']=df['genres'].apply(lambda x: [i.replace(' ','') for i in x])


# In[53]:


df


# In[54]:


df['cast']=df['cast'].apply(lambda x: [i.replace(' ','') for i in x])


# In[55]:


df


# In[56]:


df['keywords']=df['keywords'].apply(lambda x: [i.replace(' ','') for i in x])


# In[57]:


df


# In[58]:


df['director']=df['director'].apply(lambda x: [i.replace(' ','') for i in x])


# In[59]:


df


# In[60]:


df['tags']=df['overview']+df['genres']+df['keywords']+df['cast']+df['director']


# In[61]:


df


# In[62]:


ndf=df[['movie_id','title','tags']]


# In[63]:


ndf


# In[64]:


ndf['tags']=ndf['tags'].apply(lambda x: ' '.join(x))


# In[65]:


ndf


# In[66]:


ndf['tags'][0]


# In[67]:


ndf['tags']=ndf['tags'].apply(lambda x: x.lower())


# In[68]:


ndf.tags[0]


# Vectorization #BAGOFWORDS 

# In[69]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  


# In[70]:


s=stopwords.words('english')


# In[71]:


ndf['tags']=ndf['tags'].apply(lambda x: word_tokenize(x))


# In[72]:


ndf


# In[73]:


def srem(x):
    list=[]
    for i in x:
        if i not in s:
            list.append(i)
    return list


# In[74]:


ndf['tags']=ndf['tags'].apply(srem)


# In[75]:


ndf


# In[76]:


ndf['tags'][0]


# In[77]:


import string  


# In[78]:


p=string.punctuation


# In[79]:


p


# In[80]:


def rempunc(x):
    list=[]
    for i in x:
        if i not in p:
            list.append(i)
    return list


# In[81]:


ndf['tags']=ndf['tags'].apply(rempunc)


# In[82]:


ndf


# In[83]:


def l(x):
    list=[]
    for i in x:
        list.append(i.lower())
    return list
        
ndf['tags']=ndf['tags'].apply(l)


# In[84]:


ndf


# In[85]:


ndf['tags']=ndf['tags'].apply(lambda x: ' '.join(x))


# In[86]:


ndf


# In[87]:


from nltk.stem.porter import PorterStemmer


# In[88]:


p=PorterStemmer()


# In[89]:


def stem(x):
    list=[]
    for i in x.split():
        list.append(p.stem(i))
    return ' '.join(list)


# In[90]:


ndf['tags']=ndf['tags'].apply(stem)


# In[91]:


ndf


# In[92]:


from sklearn.feature_extraction.text import CountVectorizer


# In[93]:


cv=CountVectorizer(max_features=5000)


# In[94]:


vectors=cv.fit_transform(ndf['tags']).toarray()


# In[95]:


cv.get_feature_names_out()


# In[96]:


vectors


# In[97]:


from sklearn.metrics .pairwise import cosine_similarity


# In[98]:


similarity=cosine_similarity(vectors)


# In[99]:


similarity.shape


# In[101]:


index=ndf[ndf['title']=='Spectre'].index[0]


# In[102]:


distances=similarity[index]


# In[110]:


sorted(list(enumerate(similarity[index])),reverse=True,key=lambda x : x[1])[1:6]


# In[124]:


def rec(x):
    index=ndf[ndf['title']==x].index[0]
    distances=similarity[index]
    l=sorted(list(enumerate(similarity[index])),reverse=True,key=lambda x : x[1])[1:6]
    
    for i in l:
        print(ndf.iloc[i[0]].title)


# In[125]:


rec('Spectre')


# In[127]:


rec('Batman')


# In[128]:


rec('Superman')


# In[ ]:




