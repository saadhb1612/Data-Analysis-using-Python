#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math as m


# In[2]:


a = 6;b=7.2;c=8.1;d=9


# In[3]:


x = (((a*b)-m.sqrt((d+c)*a))/(a+(m.sqrt(b)/c)))


# In[4]:


x


# In[2]:


X = [3,4,22,67,1,'old',67,32]


# In[3]:


min(X)


# In[3]:


max(X)


# In[6]:


X = [1,6,-7, 4,9,-5,4]


# In[8]:


for i, n in enumerate(X):
    if n<0:
        print('found at'+str(i))


# In[9]:


import numpy


# In[10]:


numpy.__version__


# In[11]:


X=[1,2,3]
Y=[5,6,7]


# In[12]:


X+Y


# In[13]:


import numpy as np


# In[22]:


X1=np.array([1,2,3])
Y1=np.array([5,6,7])


# In[23]:


X1+Y1


# In[24]:


X1.mean()


# In[25]:


X1.sum()


# In[26]:


sum(X1)


# In[28]:


np.mean(X1)


# In[29]:


np.random.randint(1, 5, size=(4, 4))


# In[32]:


N=5
np.random.random((3,4))


# In[33]:


import pandas as pd


# In[34]:


df = pd.read_csv('C:\\Users\\ce\\BigDataAnalytics\\dataset\\titanic_train.csv')


# In[35]:


df.head()


# In[36]:


df.info()


# In[41]:


X = np.array([1,4,5,8,9,3,67,23,90])


# In[42]:


X.mean()


# In[43]:


sum(X)/len(X)


# In[57]:


A = np.array([ [1,4,6,7],
               [7,3,8,10],
               [8,3,4,9],
               [9,1,6,11]])


# In[46]:


A


# In[47]:


A[1,1]


# In[48]:


A[3,2]


# In[49]:


A[0:]


# In[50]:


A[:,0]


# In[51]:


A[2,:]


# In[52]:


A[2,1:]


# In[53]:


A


# In[54]:


A[1:,1:]


# In[56]:


A[0:3,1:]


# In[58]:


A


# In[59]:


A[1:3,0:3]


# In[62]:


x = np.random.randint(10, 42, size=(5, 4))


# In[77]:


y = np.random.randint(1, 22, size=(4, 3))


# In[75]:


x.shape


# In[76]:


y.shape


# In[66]:


x+y


# In[73]:


x*y


# In[78]:


np.dot(x,y)


# # Name: ABC

# # Problem-1

# In[2]:


M = int(input('MTN='))
E = int(input('End='))
for i in range(1, M):
    print(M, '*',i,'=',M*i)


# In[3]:


import numpy as np


# In[10]:


x= np.array([10,2,4,8,9,12,22,89,14])


# In[11]:


temp = 0


# In[12]:


for i in range(len(x)):
    temp = temp+x[i]
print(temp)


# In[14]:


x.mean()


# In[15]:


x


# In[16]:


temp =0
for i in range(len(x)):
    temp = temp + x[i]

avg = temp/len(x)
print(avg)


# In[17]:


x.mean()


# In[23]:


np.flip(x)


# In[14]:


import numpy as np
x = np.array([1,2,3,4,5,44])
xp = np.array(['jj','np','tt','zz','pp'])


# In[15]:


import math as m
end = len(xp)-1
print(end)
for i in range(m.floor(len(xp)/2)):
    temp = xp[i]
   # print(temp)
    xp[i]=xp[end]
    xp[end]=temp
    
    end = end -1


# In[16]:


xp


# In[8]:


y = np.array([1,2,3,4,5])


# In[9]:


for i in range(len(y)-1, -1, -1):     
    print(y[i]) 


# In[10]:


y


# In[40]:


m.floor(len(x))


# # even odd number detection

# In[19]:


a = int(input('enter anumber='))

if a%2==0:
    print('Even')
else:
    print('Odd')


# In[26]:


a = int(input('enter anumber='))
flag = 0
for i in range(2,a):
    if a%i==0:
        flag = 1
        break

if flag == 1:
    print('not prime')
else:
    print('prime')


# In[1]:


import numpy as np


# In[3]:


np.random.randint(1, 40, size=(15))


# In[4]:


import pandas as pd


# In[9]:


df = pd.read_csv('C:\\Users\\ce\\BigDataAnalytics\\dataset\\shootings.csv')


# In[11]:


df.head(10)


# # --- Visualization  15-04-2022 ---

# In[1]:


# Import statements
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# load datasets
google = pd.read_csv('C:\\Users\\ce\\BigDataAnalytics\\dataset\\Visualization Dataset\\data Activity 12\\GOOGL_data.csv')
facebook = pd.read_csv('C:\\Users\\ce\\BigDataAnalytics\\dataset\\Visualization Dataset\\data Activity 12\\FB_data.csv')
apple = pd.read_csv('C:\\Users\\ce\\BigDataAnalytics\\dataset\\Visualization Dataset\\data Activity 12\\AAPL_data.csv')
amazon = pd.read_csv('C:\\Users\\ce\\BigDataAnalytics\\dataset\\Visualization Dataset\\data Activity 12\\AMZN_data.csv')
microsoft = pd.read_csv('C:\\Users\\ce\\BigDataAnalytics\\dataset\\Visualization Dataset\\data Activity 12\\MSFT_data.csv')


# In[7]:


google.head(10)


# In[10]:


google.size


# In[14]:


google['close'].max()


# In[21]:


# Create figure
plt.figure(figsize=(16, 8), dpi=300)
# Plot data
plt.plot('date', 'close', data=google, label='Google',color='black')
plt.plot('date', 'close', data=facebook, label='Facebook', )
plt.plot('date', 'close', data=apple, label='Apple')
plt.plot('date', 'close', data=amazon, label='Amazon')
plt.plot('date', 'close', data=microsoft, label='Microsoft')

# Specify ticks for x and y axis
plt.xticks(np.arange(0, 1260, 40), rotation=70)
plt.yticks(np.arange(0, 1450, 100))
# Add title and label for y-axis
plt.title('Stock trend', fontsize=16)
plt.ylabel('Closing price in $', fontsize=14)


# Add grid
plt.grid()
# Add legend
plt.legend()
# Show plot
plt.show()


# In[23]:


google


# # Bar chat

# In[32]:


plt.barh(['A', 'B', 'C', 'D'], [20, 25, 40, 10], color='red')


# In[33]:


# Load dataset
movie_scores = pd.read_csv('C:\\Users\\ce\\BigDataAnalytics\\dataset\\Visualization Dataset\\data Activity 13\\movie_scores.csv')


# In[34]:


movie_scores.head(10)


# In[39]:


np.arange(1,101,3)


# In[ ]:





# In[ ]:





# In[55]:


# Create figure
plt.figure(figsize=(10, 5), dpi=300)

# Create bar plot
pos = np.arange(len(movie_scores['MovieTitle']))
width = 0.3
plt.bar(pos - width / 2, movie_scores['Tomatometer'], width, label='Tomatometer')
plt.bar(pos + width / 2, movie_scores['AudienceScore'], width, label='AudienceScore')

# Specify ticks
plt.xticks(pos, rotation=10)
plt.yticks(np.arange(0, 101, 20))
# Get current Axes for setting tick labels and horizontal grid
ax = plt.gca()
# Set tick labels
ax.set_xticklabels(movie_scores['MovieTitle'])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

# Add minor ticks for y-axis in the interval of 5
ax.set_yticks(np.arange(0, 100, 5), minor=True)
# Add major horizontal grid with solid lines
ax.yaxis.grid(which='major')
#Add minor horizontal grid with dashed lines
ax.yaxis.grid(which='minor', linestyle='--')

# Add title
plt.title('Movie comparison')
# Add legend
plt.legend()
# Show plot
plt.show()


# In[40]:


pos = np.arange(len(movie_scores['MovieTitle']))


# In[41]:


pos


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('C:\\Users\\ce\\BigDataAnalytics\\dataset\\titanic_train.csv')


# In[4]:


df.head()


# In[5]:


df.Survived.value_counts()


# In[8]:


X = df.Sex.value_counts()


# In[9]:


X


# In[10]:


X.male


# In[ ]:


import pandas as pd


# In[11]:


dfW = pd.read_csv('C:\\Users\\ce\\BigDataAnalytics\\dataset\\Visualization Dataset\\water_usage.csv')


# In[12]:


dfW.head()


# # --- 22-04-2022 ---

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dfA = pd.read_csv('C:\\Users\\ce\\BigDataAnalytics\\dataset\\Visualization Dataset\\data Activity 17\\anage_data.csv')


# In[4]:


dfA.shape


# In[5]:


dfA.head()


# In[6]:


dfA.info()


# In[7]:


longevity = 'Maximum longevity (yrs)'
mass = 'Body mass (g)'
dfA = dfA[np.isfinite(dfA[longevity]) & np.isfinite(dfA[mass])]


# In[9]:


dfA.info()


# In[10]:


dfA.head()


# In[11]:


dfA.Class.value_counts()


# In[12]:


# Sort according to class
amphibia = dfA[dfA['Class'] == 'Amphibia']
aves =dfA[dfA['Class'] == 'Aves']
mammalia =dfA[dfA['Class'] == 'Mammalia']
reptilia = dfA[dfA['Class'] == 'Reptilia']


# In[13]:


amphibia.shape


# In[15]:


mammalia.shape


# In[22]:


# Create figure
plt.figure(figsize=(10, 6), dpi=300)
# Create scatter plot
plt.scatter(amphibia[mass], amphibia[longevity], label='Amphibia')
plt.scatter(aves[mass], aves[longevity], label='Aves')
plt.scatter(mammalia[mass], mammalia[longevity], label='Mammalia')
plt.scatter(reptilia[mass], reptilia[longevity], label='Reptilia')

# Add legend
plt.legend()
# Log scale
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')


# Add labels
plt.xlabel('Body mass in grams')
plt.ylabel('Maximum longevity in years')
# Show plot
plt.show()


# # --- Data Wrangling 06-06-2022 ---

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv('titanic_train.csv')


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df1 = pd.read_csv('C:\\Users\\ce\\BigDataAnalytics\\dataset\\CarPrice.csv')


# In[7]:


df1.shape


# In[8]:


df2 = pd.read_csv('C:/Users/ce/BigDataAnalytics/dataset/CarPrice.csv')


# In[9]:


df2.shape


# In[10]:


df3 = pd.read_csv(r'C:\Users\ce\BigDataAnalytics\dataset\CarPrice.csv')


# In[11]:


df4 = pd.read_csv('C:\Users\ce\BigDataAnalytics\dataset\CarPrice.csv')


# In[17]:


df1 = pd.read_csv('./dataset/CarPrice.csv')


# In[18]:


df1.shape


# In[19]:


df2 = pd.read_csv('Hour Price Prediction -lessAtt.csv')


# In[22]:


df = pd.read_csv('titanic_train.csv')


# In[23]:


df.shape


# In[24]:


df.head()


# In[25]:


df.info()


# In[26]:


df.isnull()


# In[27]:


import seaborn as sns


# In[28]:


sns.heatmap(df.isnull())


# In[30]:


df.columns


# In[34]:


df.drop('Cabin',axis=1,inplace=True)


# In[35]:


df.shape


# In[36]:


df.head()


# In[37]:


df.info()


# In[38]:


df.head(20)


# In[39]:


df.info()


# In[41]:


df1 = df.dropna(how='any')


# In[42]:


df1.info()


# In[43]:


df.info()


# # 09-06-2022

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('titanic_train.csv')


# In[82]:


df.info()


# In[3]:


df.drop('Cabin',axis=1,inplace=True)


# In[84]:


df.info()


# In[4]:


df.Embarked.value_counts()


# In[86]:


df.head()


# In[5]:


df.Embarked.fillna('S',inplace=True)


# In[88]:


df.info()


# In[89]:


df.Age.mean()


# In[90]:


df.Embarked.mode()


# In[91]:


import statistics as st


# In[92]:


st.mode(df.Embarked)


# In[93]:


df.head()


# In[94]:


df.Age.fillna(df.Age.mean(),inplace=True)


# In[95]:


df.info()


# In[96]:


df.head()


# In[6]:


df.drop('PassengerId',axis=1, inplace=True)


# In[98]:


df.info()


# In[99]:


df.head()


# In[7]:


df.replace(['male','female'],[0,1],inplace=True)


# In[101]:


df.head()


# In[8]:


df.groupby('Embarked')['Age'].mean()


# In[10]:


df.Embarked.value_counts()


# In[11]:


df.replace(['S','C','Q'],[2,1,0],inplace=True)


# In[104]:


df.head()


# In[12]:


df['Title']= df.Name.str.extract('([A-Za-z]+)\.',expand=False)


# In[106]:


df.head()


# # -- 1-11-2022 -- 

# In[107]:


df.head()


# In[108]:


import seaborn as sns


# In[109]:


sns.catplot(x ="Sex", hue ="Survived", kind ="count", data = df)


# In[112]:


sns.catplot(x ="Sex", hue ="Survived", col='Pclass', kind ="count", data = df)


# In[114]:


df['Fare_Range'] = pd.qcut(df['Fare'], 4,labels=[0,1,2,3])
sns.barplot(x ='Fare_Range', y ='Survived', data = df)


# In[115]:


sns.catplot(x ="Fare_Range", hue ="Survived", kind ="count", data = df)


# In[116]:


df['new_Age'] = pd.qcut(df['Age'], 5,labels=[0,1,2,3,4])
sns.barplot(x ='new_Age', y ='Survived', data = df)


# In[120]:


sns.catplot(x ="new_Age", hue ="Survived",col='Sex', kind ="count", data = df)


# In[123]:


sns.catplot(x ="Sex", hue ="Pclass", kind ="count", data = df)


# In[129]:


sns.catplot(x ="Fare_Range", hue ="Sex", kind ="count", data = df)


# In[125]:


sns.catplot(x ="Embarked", hue ="Sex", kind ="count", data = df)


# In[130]:


df.Sex.value_counts()


# In[139]:


df.to_csv('clean_titanic_BDA3.csv', index=True)


# # hotel_booking ---

# In[131]:


dfH = pd.read_csv('./dataset/hotel_bookings.csv')


# In[132]:


dfH.shape


# In[133]:


dfH.head()


# In[135]:


dfH.columns


# In[137]:


dfH.is_canceled.value_counts()


# In[138]:


dfH.info()


# In[142]:


dfH.isnull().sum()


# In[143]:


dfH.agent.value_counts()


# In[146]:


import statistics as st


# In[148]:


dfH.agent.fillna(st.mode(dfH.agent),inplace=True)


# In[149]:


dfH.agent = dfH.agent.astype(int)


# In[150]:


dfH.info()


# # -- 16-06-2022 

# In[2]:


import pandas as pd
import numpy as np
dfH = pd.read_csv('./dataset/hotel_bookings.csv')


# In[4]:


df = pd.read_csv('./dataset/titanic_train.csv')


# In[6]:


df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=True)


# In[8]:


df['Age'].fillna(df.groupby('Title')['Age'].transform('mean'),inplace=True)


# In[12]:


dfH.dropna(subset=['agent'], inplace=True)


# In[15]:


dfH.country.head(10)


# In[18]:


dfH.country.value_counts()


# In[6]:


pd.set_option('display.max_rows', 1000000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[19]:


countss = dfH.country.value_counts()


# In[21]:


mask  = dfH['country'].isin((countss)[countss<=300].index)


# In[26]:


dfH['country'][mask]='other'


# In[36]:


dfH.agent = dfH.agent.astype('int64')


# In[43]:


x = dfH[dfH['agent']==9]


# # --- 17-06-2022 ---

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")


# In[3]:


dfH = pd.read_csv('./dataset/hotel_bookings.csv')


# In[7]:


dfH.head()


# In[8]:


dfH.drop('company', axis=1, inplace=True)


# In[9]:


dfH.drop('arrival_date_week_number', axis=1, inplace=True)


# In[19]:


dfH['arrival_date'] = dfH['arrival_date_month'] + " " + dfH['arrival_date_day_of_month'].astype(str) + " " + dfH['arrival_date_year'].astype(str)


# In[21]:


dfH['arrival_date'] = pd.to_datetime(dfH['arrival_date'])


# In[14]:


dfH['country'].fillna(dfH['country'].mode()[0], inplace=True)
dfH['children'].fillna(dfH['children'].median(), inplace=True)
dfH['agent'].fillna(dfH['agent'].median(), inplace=True)
dfH.isna().sum().sort_values(ascending=False) / len(dfH)


# In[22]:


monthly = dfH.groupby(pd.Grouper(key='arrival_date', axis=0, freq='M')).sum()


# In[23]:


monthly


# In[24]:


dfH.head()


# In[26]:


df


# In[27]:


dfH = pd.read_csv('./dataset/titanic_train.csv')


# In[28]:


sns.boxplot(dfH.Age)


# # ---23-06-2022 ---

# # --- outlier ---

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston


# In[2]:


boston_dataset = load_boston()
print(boston_dataset.feature_names)


# In[3]:


boston = pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)[['RM', 'LSTAT', 'CRIM']]


# In[109]:


boston.info()


# In[110]:


sns.distplot(boston['RM'])


# In[4]:


sns.boxplot(boston['RM'])


# In[5]:


def find_boundaries(df, variable, distance):
    Q1 = df[variable].quantile(0.25)
    Q3 = df[variable].quantile(0.75)
    
    IQR = Q3 - Q1
    LB = Q1 - (IQR * distance)
    UB = Q3 + (IQR * distance)
    return LB, UB


# In[6]:


RM_LB, RM_UB = find_boundaries(boston, 'RM', 1.5)


# In[7]:


RM_LB, RM_UB 


# In[100]:


outliers_RM = np.where(boston['RM'] > RM_UB, True, np.where(boston['RM'] < RM_LB, True,False))


# In[101]:


boston['RM'][outliers_RM].count()


# In[102]:


boston_T = boston.loc[~(outliers_RM)]


# In[104]:


sns.boxplot(boston_T['LSTAT'])


# # --- loc and iloc ---

# # explicit and implicit location ---

# In[76]:


data = pd.Series(['a','b','c'],index =[1,3,5])


# In[77]:


data


# In[88]:


data1 = pd.Series(['d','e','f'],index=['a1','a2','a3'])


# In[89]:


data1


# In[93]:


data1.iloc[0]


# In[60]:


Q1 = boston['RM'].quantile(0.25)
Q3 = boston['RM'].quantile(0.75)
    
IQR = Q3 - Q1
LB = Q1 - (IQR * 1.3)
UB = Q3 + (IQR * 1.3)


# In[61]:


LB, UB


# In[63]:


find_boundaries(boston,'LSTAT',1.5)


# In[29]:


outliers_RM = np.where(boston['RM'] > RM_upper_limit, True, np.where(boston['RM'] < RM_lower_limit, True,False))


# In[30]:


boston_trimmed = np.NaN(boston.loc[~(outliers_RM)])


# In[31]:


sns.boxplot(boston_trimmed['RM'])


# In[8]:


df_new = boston.RM[(boston.RM < RM_UB) & (boston.RM > RM_LB)]


# In[9]:


boston['RM_new']=df_new      


# In[11]:


boston


# In[12]:


sns.boxplot(boston['RM_new'])


# In[42]:


boston.corr()


# # --- 24-06-2022 ---

# In[116]:


RM_LB, RM_UB


# In[117]:


df_new = boston.RM[(boston.RM < RM_UB) & (boston.RM > RM_LB)]


# In[118]:


df_new


# In[119]:


boston['RM_new'] = df_new


# In[120]:


boston


# In[123]:


boston.info()


# In[124]:


boston.drop('RM',axis=1,inplace=True)


# In[125]:


boston.info()


# In[126]:


boston.RM_new.fillna(boston.RM_new.mean(),inplace=True)


# In[127]:


boston.info()


# In[131]:


sns.boxplot(boston.RM_new)


# # --- Titanic dataset ---

# In[133]:


dfT = pd.read_csv('./dataset/titanic_train.csv')


# In[135]:


dfT.info()


# In[141]:


sns.boxplot(dfT.Fare)


# In[149]:


Fare_UB = 300
Fare_LB = 0
new_Fare = dfT.Fare[(dfT.Fare <= Fare_UB) & (dfT.Fare >= Fare_LB)]


# In[150]:


dfT['new_Fare']  = new_Fare


# In[151]:


dfT.info()


# In[148]:


dfT[dfT.Fare==dfT.Fare.max()]


# In[153]:


dfT.new_Fare.fillna(dfT.groupby('Pclass')['new_Fare'].transform('max'),inplace=True)


# In[154]:


dfT.info()


# In[155]:


sns.boxplot(dfT.new_Fare)


# In[ ]:


1. Age filling according title
2. outlier handling by NaN


# In[158]:


dfT['Title'] = dfT['Name'].str.extract('([A-Za-z]+\.)',expand=False)


# In[159]:


dfT['Age'].fillna(dfT.groupby('Title')['Age'].transform('mean'), inplace=True)


# In[160]:


dfT.drop('Tilte',axis=1,inplace=True)


# In[162]:


dfT.info()


# In[163]:


sns.boxplot(dfT.Age)


# In[174]:


dfT[dfT.Age>=65]


# In[175]:


Age_UB = 65
Age_LB = 0
new_Age = dfT.Age[(dfT.Age <= Age_UB) & (dfT.Age >= Age_LB)]


# In[176]:


dfT['new_Age']=new_Age


# In[178]:


dfT.info()


# In[179]:


dfT['new_Age'].fillna(dfT.groupby('Title')['new_Age'].transform('mean'), inplace=True)


# In[180]:


dfT.info()


# In[181]:


sns.boxplot(dfT.new_Age)


# In[182]:


sns.distplot(dfT.new_Age)


# In[183]:


dfT.corr()


# In[185]:


sns.heatmap(dfT.corr(),annot=True)


# # --- 28-06-2022 --- Regex

# In[34]:


import re
txt = "The rain in Pakistap"
#Find all lower case characters alphabetically between "a" and "m":
x = re.findall("[a-m]{1,3}", txt)
print(x)


# In[20]:


import re
txt = "Tha3t will be 59 dol89lars908"
#Find all digit characters:
x = re.findall("[^0-9]+", txt)
print(x)


# In[30]:


import re
txt = "hello plhelloanet jello Hello"
#Search for a sequence that starts with "he", followed by two (any) characters, and an "o":
x = re.findall("he..o|j...o", txt)
print(x)


# # --- 30-06-2022 Titatic EDA ---

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('titanic_train.csv')


# In[15]:


df.columns


# In[16]:


df.drop('PassengerId', axis=1, inplace=True)


# In[18]:


df.drop('Cabin',axis=1, inplace=True)


# In[19]:


df.Embarked.value_counts()


# In[21]:


df.Embarked.fillna('S',inplace=True)


# In[22]:


df['Title']= df.Name.str.extract('([A-Za-z]+)\.',expand=False)


# In[24]:


df['Age'].fillna(df.groupby('Title')['Age'].transform('mean'), inplace=True)


# In[25]:


df.info()


# In[27]:


df.Pclass.value_counts()


# In[33]:


df['Pclass'] = pd.factorize(df['Pclass'])[0]


# In[29]:


df.temp.value_counts()


# In[34]:


df.drop('temp',axis=1,inplace=True)


# In[3]:


df.head()


# # --- one-hot-encoding ---

# In[4]:


df.columns


# In[5]:


df.Pclass.value_counts()


# In[6]:


df1 = pd.get_dummies(df, columns=['Pclass'],prefix='Pc', drop_first=True)


# In[7]:


df1.head()


# In[9]:


df1 = pd.get_dummies(df, columns=['Pclass'],prefix='Pc', drop_first=True)


# In[10]:


df1.head()


# In[11]:


from sklearn.preprocessing import StandardScaler


# # --- 01-07-2022 --- 

# In[17]:


import pandas as pd
import numpy as np


# In[27]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[21]:


dfC = pd.read_csv('./dataset/CarPrice.csv')


# In[22]:


dfC.columns


# In[23]:


dfC.info()


# In[24]:


dfC.isna().sum().sort_values(ascending=False) / len(dfC)


# In[28]:


dfC.head()


# In[29]:


dfC.fueltype.value_counts()


# In[31]:


dfC.CarName.value_counts()


# In[32]:


import re


# In[43]:


#re.findall('^\S+',dfC.CarName)
dfC['CarNameNew'] = dfC.CarName.str.extract('(^\S+)') #


# In[46]:


dfC.drop('CarName',axis=1,inplace=True)


# In[47]:


dfC.head()


# In[48]:


import seaborn as sns


# In[50]:


sns.catplot(x ="doornumber", hue ="fueltype", kind ="count", data = dfC)


# In[53]:


dfC.groupby(['fueltype','doornumber'])['price'].max()


# In[54]:


import matplotlib.pyplot as plt


# In[55]:


dfC.groupby(['fueltype','doornumber'])['price'].max().plot(kind='bar')


# In[63]:


dfC.head()


# In[58]:


dfC.drop('car_ID',axis=1, inplace=True)


# In[62]:


dfC.fueltype.replace(['gas','diesel'],[1,0],inplace=True)


# In[64]:


dfC.aspiration.value_counts()


# In[67]:


dfC.groupby('aspiration')['price'].mean()


# In[69]:


dfC.aspiration.replace(['std','turbo'],[1,0],inplace=True)


# In[70]:


dfC.head()


# In[71]:


dfC.columns


# In[72]:


dfC['doornumber'] = pd.factorize(dfC['doornumber'])[0]


# In[74]:


dfC['carbody'] = pd.factorize(dfC['carbody'])[0]
dfC['drivewheel'] = pd.factorize(dfC['drivewheel'])[0]
dfC['enginelocation'] = pd.factorize(dfC['enginelocation'])[0]


# In[75]:


dfC.head()


# In[77]:


sns.heatmap(dfC.corr())


# In[81]:


sns.scatterplot(dfC.price,dfC.enginesize)


# In[82]:


sns.scatterplot(dfC.carlength,dfC.citympg)


# In[79]:


sns.boxplot(dfC.wheelbase)


# In[87]:


wheelbase_UB = 115
wheelbase_LB = 85
dfC['new_weheelbase'] = dfC.wheelbase[(dfC.wheelbase <= wheelbase_UB) & (dfC.wheelbase >= wheelbase_LB)]


# In[88]:


dfC.info()


# In[93]:


dfC['new_weheelbase'].fillna(dfC['new_weheelbase'].mean(),inplace=True)


# In[94]:


dfC.info()


# In[95]:


dfC.drop('wheelbase',axis=1,inplace=True)


# In[98]:


sns.distplot(dfC.new_weheelbase)


# In[99]:


dfC.head()


# # Machine learning - 14-07-2022 

# # - Titanic daset --- for classification-based problem

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


dfT = pd.read_csv('./dataset/titanic_clean1.csv')


# In[5]:


dfT.head()


# In[6]:


dfT.drop('Unnamed: 0',axis=1,inplace=True)


# In[7]:


dfT.head()


# In[8]:


X = dfT.drop('Survived',axis=1)


# In[9]:


y = dfT['Survived']


# # --- Train test split ----

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=42 )


# # --- machine learning Algorithm (Logistic regression) --- 

# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


modelLR = LogisticRegression()  # logistic regression object


# In[14]:


modelLR.fit(trainX, trainY)


# # --- Prediction ---

# In[15]:


predictR = modelLR.predict(testX)


# # --- Performance ---

# In[16]:


from sklearn.metrics import accuracy_score, classification_report


# In[17]:


accuracy_score(testY,predictR)


# In[18]:


print(classification_report(testY, predictR))


# # --- house price prediction dataset --- regression-based problem

# In[20]:


dfH = pd.read_csv('./dataset/House-Price-Prediction-minAtt.csv')


# In[21]:


dfH.head()


# In[22]:


dfH.drop('Id',axis=1, inplace=True)


# In[23]:


X = dfH.drop('SalePrice', axis=1)


# In[24]:


y = dfH['SalePrice']


# In[25]:


trainX, testX, trainY, testY = train_test_split(X, y, test_size= 0.3, random_state=42)


# In[26]:


from sklearn.linear_model import LinearRegression


# In[27]:


modelLG = LinearRegression()


# In[28]:


modelLG.fit(trainX, trainY)


# # --test performance ---

# In[33]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[30]:


PLR = modelLG.predict(testX)


# In[32]:


testY


# In[31]:


PLR


# In[26]:


r2_score(PLR,testY)


# In[34]:


mean_absolute_error(PLR,testY)


# In[36]:


np.sqrt(mean_squared_error(PLR,testY))


# # --- Training performance ---

# In[27]:


PLRT = modelLG.predict(trainX)


# In[28]:


r2_score(PLRT,trainY)


# # --- performance of Regression based problem--- MSE, RMSE, MAE

# In[41]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[42]:


mean_absolute_error(PLR, testY)


# In[43]:


mean_squared_error(PLR,testY)


# In[44]:


np.sqrt(mean_absolute_error(PLR, testY))


# # --- 19-07-2022 ---

# # --- KNN --- for regression

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[5]:


dfH = pd.read_csv('./dataset/House-Price-Prediction-minAtt.csv')


# In[6]:


dfH.head()


# In[7]:


dfH.drop('Id',axis=1,inplace=True)


# In[8]:


X = dfH.drop('SalePrice',axis=1)


# In[9]:


y = dfH['SalePrice']


# In[10]:


trainX, testX, trainY, testY = train_test_split(X,y, test_size=0.3, random_state=42)


# In[12]:


modelKNN = KNeighborsRegressor(n_neighbors=5)


# In[13]:


modelKNN.fit(trainX,trainY)


# In[14]:


precKNN = modelKNN.predict(testX)


# In[15]:


MAE = mean_absolute_error(testY, precKNN)


# In[16]:


MAE


# In[18]:


MSE = mean_squared_error(testY, precKNN)


# In[19]:


RMSE = np.sqrt(MSE)


# In[20]:


RMSE


# In[21]:


print(r2_score(testY,precKNN))


# In[28]:


np.mean(testY-precKNN)


# # --- 21-07-2022 --- classification-based problem and metrics

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


# In[37]:


dfD = pd.read_csv('./dataset/diabetes.csv')


# In[38]:


dfD.head()


# In[39]:


X = dfD.drop('Outcome',axis=1)


# In[40]:


y = dfD['Outcome']


# In[41]:


trainX, testX, trainY, testY = train_test_split(X,y, test_size=0.3, random_state=42)


# In[30]:


modelKNN = KNeighborsClassifier(n_neighbors=9)


# In[31]:


modelKNN.fit(trainX, trainY)


# In[32]:


preKNN = modelKNN.predict(testX)


# In[33]:


print(classification_report(preKNN, testY))


# In[18]:


print(confusion_matrix(preKNN, testY))


# In[22]:


del dfC


# # --- Decision Tree ---

# In[42]:


from sklearn.tree import DecisionTreeClassifier


# In[43]:


modelDT = DecisionTreeClassifier()


# In[44]:


modelDT.fit(trainX, trainY)


# In[28]:


preDT = modelDT.predict(testX)


# In[29]:


print(classification_report(preDT, testY))


# # 22-07-2022 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report


# In[3]:


dfT = pd.read_csv('./dataset/titanic_clean1.csv')


# In[6]:


dfT.head()


# In[4]:


dfT.drop('Unnamed: 0',axis=1, inplace=True)


# In[5]:


X = dfT.drop('Survived',axis=1)


# In[6]:


y = dfT['Survived']


# In[7]:


trainX, testX, trainY, testY = train_test_split(X,y, test_size=0.3, random_state=12)


# In[8]:


modelDT = DecisionTreeClassifier(criterion='gini', max_depth=4)


# In[9]:


modelDT = modelDT.fit(trainX, trainY)


# In[27]:


from sklearn import tree


# In[33]:


tree.plot_tree(modelDT) 


# # --- Test Performance --

# In[10]:


preDT = modelDT.predict(testX)


# In[11]:


print(classification_report(preDT,testY))


# # Training performance ----

# In[12]:


preDTT = modelDT.predict(trainX)


# In[14]:


print(classification_report(preDTT,trainY))


# In[30]:


print(classification_report(preDT,testY))


# In[35]:


print(classification_report(preDT,testY))


# # --- uci dataset ---

# In[49]:


dfU = pd.read_csv('./dataset/creditApprovalUCI.csv')


# In[50]:


dfU.info()


# In[51]:


dfU.head()


# In[53]:


dfU.A1.value_counts()


# In[54]:


dfU.A4.value_counts()


# In[55]:


dfU.A5.value_counts()


# In[56]:


dfU.A6.value_counts()


# In[57]:


dfU.A7.value_counts()


# In[58]:


count1 = dfU.A1.value_counts()


# In[60]:


count1


# In[61]:


mask  = dfU['A1'].isin((count1)[count1<=12].index)
dfU['A1'][mask]='a'


# In[62]:


dfU.A1.value_counts()


# In[65]:


dfU.head()


# In[64]:


dfU['A1'] = pd.factorize(dfU['A1'])[0]


# In[66]:


dfU.A1.value_counts()


# # --- Random Forest ---

# In[68]:


from sklearn.ensemble import RandomForestClassifier


# In[82]:


modelRF = RandomForestClassifier(n_estimators=101, criterion='entropy', max_depth=4)


# In[83]:


modelRF.fit(trainX,trainY)


# In[84]:


preRF = modelRF.predict(testX)


# In[72]:


print(classification_report(preRF,testY))


# In[77]:


print(classification_report(preRF,testY))


# In[81]:


print(classification_report(preRF,testY))


# In[85]:


print(classification_report(preRF,testY))


# # --- 28-07-2022   Cross validation ---

# In[37]:


from numpy import mean
from numpy import std
from sklearn.datasets import make_classification  # to create dataset
from sklearn.model_selection import KFold,StratifiedKFold  # splitting tech
from sklearn.model_selection import cross_val_score  # kfold score
from sklearn.linear_model import LogisticRegression   # classifier


# In[38]:


# create dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative = 15, n_redundant=5, random_state=42)
# prepare the cross-validation procedure
cv1 = KFold(n_splits=10, random_state=42, shuffle=True)
Skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
# create model
model = LogisticRegression(max_iter=500)


# In[42]:


# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=Skf, n_jobs=-1)


# In[43]:


scores


# In[44]:


# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# # --- 29-07-2021 ---

# In[2]:


from sklearn import datasets
cancer_data = datasets.load_breast_cancer()
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score


# In[6]:


import pandas as pd


# In[7]:


X = pd.DataFrame(cancer_data.data)


# In[11]:


y = pd.DataFrame(cancer_data.target)


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109)


# In[14]:


from sklearn.svm import SVC


# In[55]:


modelSVC = SVC(kernel='linear')


# In[25]:


modelSVC.fit(X_train,y_train)


# In[26]:


preSVC = modelSVC.predict(X_test)


# In[21]:


from sklearn.metrics import classification_report


# In[22]:


print(classification_report(preSVC,y_test))


# In[27]:



print(classification_report(preSVC,y_test))


# In[29]:


from sklearn.linear_model import LogisticRegression


# In[34]:


modelLgR = LogisticRegression(max_iter=1500)


# In[35]:


modelLgR.fit(X_train,y_train)


# In[36]:


preLgR = modelLgR.predict(X_test)


# In[38]:


print(classification_report(preLgR,y_test))


# # --- using cross validation ---

# In[40]:


cv1 = KFold(n_splits=5, random_state=42, shuffle=True)
Skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)


# In[56]:


scores = cross_val_score(modelSVC, X, y, scoring='accuracy', cv=Skf, n_jobs=-1)


# In[57]:


scores


# In[58]:


scores.mean()


# # --- PCA 02-08-2022 ---

# In[ ]:


learn.Decomposition import PCA
PCAcom = 2
pca = PCA()
pca.n_components = PCAcom
pca_data = pca.fit_transform(sample_data)
pca_df = pd.DataFrame(data=pca_data)


# # --- K-mean --

# In[2]:


from sklearn.cluster import KMeans
import numpy as np


# In[3]:


X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])


# In[4]:


X


# In[5]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(X)


# In[7]:


# Kemans.fit(X)
Y = kmeans.labels_


# In[8]:


Y


# # --- Titanic data ---

# In[9]:


import pandas as pd


# In[11]:


dfT = pd.read_csv('./dataset/titanic_clean1.csv')


# In[12]:


dfT


# In[13]:


X = dfT.drop('Survived',axis=1)


# In[14]:


X.head()


# In[17]:


X.drop('Unnamed: 0',axis=1, inplace=True)


# In[18]:


X.head()


# In[19]:


dfTkmean = KMeans(n_clusters=2, random_state=42).fit(X)


# In[23]:


Y = dfTkmean.labels_


# In[33]:


Y = pd.DataFrame(Y,columns=['Survived'])


# In[35]:


newDF = pd.concat([X,Y],axis=1)


# In[36]:


newDF.head(50)


# # Hyperparameter - Fine Tuning 
# 

# In[7]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[3]:


Data = pd.read_csv('./dataset/diabetes.csv')


# In[4]:


X = Data.drop('Outcome',axis='columns')
y = Data['Outcome']


# In[6]:


train_x, test_x, train_y, test_y = train_test_split(X,y,test_size = 0.3)
modelRF = RandomForestClassifier(n_estimators=200, max_depth=5,random_state=0, criterion='entropy')


# In[9]:


modelRF.fit(train_x,train_y)
RF_pred = modelRF.predict(test_x)


# In[11]:


print(classification_report(RF_pred,test_y))


# # now implement the Hyperparameters

# In[31]:


from sklearn.model_selection import RandomizedSearchCV,GridSearchCV


# In[13]:


random_search = {'criterion': ['entropy', 'gini'],
                   'max_depth': list(np.linspace(10, 1200, 10, dtype = int)) +[None],
                   'max_features': ['auto', 'sqrt','log2', None],
                   'min_samples_leaf': [4, 6, 8, 12],
                   'min_samples_split': [5, 7, 10, 14],
                   'n_estimators': list(np.linspace(151, 1200, 10, dtype = int))}


# In[17]:


clf = RandomForestClassifier()
model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 1, cv = 4, verbose= 5, random_state= 101, n_jobs = -1)
model.fit(train_x,train_y)


# In[19]:


model.best_params_


# In[26]:


modelRF_hyp = RandomForestClassifier(n_estimators=267, max_depth=1200,random_state=0, criterion='gini',min_samples_split= 14,min_samples_leaf= 4,max_features= 'log2')


# In[27]:


modelRF_hyp.fit(train_x,train_y)
predict_hyp = modelRF_hyp.predict(test_x)


# In[28]:


print(classification_report(predict_hyp,test_y))


# In[29]:


table = pd.pivot_table(pd.DataFrame(modelRF_hyp.cv_results_),values='mean_test_score', index='param_max_depth',columns='param_min_samples_split')


# In[ ]:


class sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)[source]


# In[35]:


modelGS = GridSearchCV(estimator = clf, param_grid= random_search, scoring=None, n_jobs=1, refit=True, cv=4, verbose=1,  pre_dispatch='2*n_jobs',  return_train_score=False)


# In[36]:


modelGS.fit(train_x,train_y)


# # kaggle submission

# In[43]:


dfT = pd.read_csv('./dataset/Titanic_sub/train.csv')


# In[44]:


dfT.head()


# In[45]:


dfTe = pd.read_csv('./dataset/Titanic_sub/test.csv')


# In[46]:


dfTe.head()


# In[47]:


dfTPI = pd.read_csv('./dataset/Titanic_sub/gender_submission.csv')


# In[48]:


dfTPI


# In[ ]:


Pre_sub = model.fit(dfTe)


# In[ ]:


test_ids = test['PassengerId']
submission_preds = clf.predict(test)
df = pd.DataFrame({'PassengerId':test_ids.values,
                    'Survived': submission_preds,)}

df.to_csv('submission.csv',index=False)


# # Statistics for Data Science

# In[ ]:




