import tkinter
from tkinter import *


import sys
sys.path.append("/usr/lib/python2.7/dist-packages")
import numpy as np
import pandas as pb
df = pb.read_csv("gsm.csv")
df.head


# In[2]:


df.columns


# In[3]:


df_cols=["index","label","text"]
df.columns = df_cols


# In[4]:


df.head


# In[5]:


df.groupby('label').describe()


# In[6]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.text,df.label, test_size=0.25)


# In[7]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

model = MultinomialNB()
v=CountVectorizer()


clf = Pipeline([
     ('vectorizer',CountVectorizer()),
    ('rb',MultinomialNB())
])


# In[8]:


clf.fit(x_train, y_train)


# In[9]:


clf.score(x_test,y_test)







root=Tk()
root.title("Mood predictor")
e= Entry(root,width=100)
e.pack()
e.insert(0,"")




myLabel=Label(root,text="click")

def myclick():
    messages =["mi "]
    
    mi = e.get()
    print(messages[0])
    messages[0]=mi


    print(messages[0])

    messages_count = v.fit_transform(messages)
    clf.predict(messages)
    mood_num=int(np.asarray(clf.predict(messages)))
    print(mood_num)
   
    
    if(mood_num == 4):
         print("Positive")
         myLabel.config(text = "Positive")
         
    else:
         print("Negetive")
         myLabel.config(text = "Negative")

    myLabel.pack()
    




myButton = Button(root,text="Enter your" , command=myclick)
myButton.pack()


root.mainloop()
