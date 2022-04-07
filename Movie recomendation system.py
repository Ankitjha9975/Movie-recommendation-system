#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[ ]:





# # Get the data

# In[3]:


coloumns_names = ['user_id','item_id','rating','timestamp']
# as the file is tsv(tab seperated value) we use sep function to use it under csv
df = pd.read_csv('ml-100k/u.data.data',sep='\t',names=coloumns_names)   


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df['user_id'].nunique()      # this gives the unique the idea about unique users 


# In[7]:


df['item_id'].nunique()     # this gives the unique the idea about unique movies


# In[8]:


# as the file is '\' seperated we use sep function to use it under csv
movies_titles = pd.read_csv('ml-100k/u.item.item',delimiter='\|',header=None)


# In[9]:


movies_titles.head()


# In[10]:


movies_titles.shape


# In[11]:


movies_titles = movies_titles[[0,1]]
movies_titles.columns = ['item_id','title']


# In[12]:


movies_titles.head()


# In[13]:


#Now we will merge the both dataframes


# In[14]:


df = pd.merge(df,movies_titles,on='item_id')


# In[15]:


df.head()


# In[16]:


df.tail()


# In[ ]:





# In[ ]:





# # Exploratory Data Analysis

# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


# In[18]:


df.groupby('title').mean()['rating'].sort_values(ascending=False).head()


# In[19]:


df.groupby('title').count()['rating'].sort_values(ascending=False).head()


# In[ ]:





# In[20]:


ratings =pd.DataFrame(df.groupby('title').mean()['rating']) 


# In[21]:


ratings.head ()


# In[22]:


ratings['num of rating'] = pd.DataFrame(df.groupby('title').count()['rating'])


# In[23]:


ratings


# In[24]:


ratings.sort_values(by='rating',ascending=False)


# In[25]:


#As the above movies are just rated by few persons we will discard such movies


# In[26]:


plt.figure(figsize=(10,6))
plt.hist(ratings['num of rating'],bins=70)
plt.show()


# In[27]:


plt.hist(ratings['rating'],bins=70)
plt.show()


# In[28]:


#Above fig shows that it is a normal distrubation


# In[29]:


sns.jointplot(x='rating',y='num of rating',data=ratings,alpha=0.5)


# In[ ]:





# In[ ]:





# # Creating Movie Recommendation

# In[30]:


df.head()


# In[31]:


moviemat = df.pivot_table(index='user_id',columns='title',values='rating')


# In[32]:


moviemat.head()


# In[33]:


ratings.sort_values('num of rating',ascending=False).head()


# In[34]:


starwars_user_ratings = moviemat['Star Wars (1977)']
starwars_user_ratings.head()


# In[35]:


#Now we will co-relate the indevidual movie rating with whole moviemat


# In[36]:


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)


# In[37]:


similar_to_starwars


# In[38]:


corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])


# In[39]:


corr_starwars.dropna(inplace=True)


# In[40]:


corr_starwars.head()


# In[41]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[42]:


corr_starwars = corr_starwars.join(ratings['num of rating'])
corr_starwars.head()


# In[43]:


corr_starwars[corr_starwars['num of rating']>100].sort_values('Correlation',ascending=False)


# In[ ]:





# # Predict Function

# In[44]:


def predict_movies(movie_name):
    movie_user_rating = moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_rating)
    
    corr_movie = pd.DataFrame(similar_to_movie,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    
    corr_movie =corr_movie.join(ratings['num of rating'])
    predictions = corr_movie[corr_movie['num of rating']>100].sort_values('Correlation',ascending=False)
    
    return predictions
    


# In[45]:


predict_movies('Titanic (1997)')


# In[46]:


predict_movies('Aliens (1986)')


# In[47]:


predict_movies('2001: A Space Odyssey (1968)')


# In[48]:


predict_movies('Man of the Year (1995)')


# In[49]:


predict_movies('Star Wars (1977)')


# In[ ]:





# In[ ]:





# In[50]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




