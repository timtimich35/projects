#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Импортируем базовые библиотеки для работы с данными и визуализации
import pandas as pd
import numpy as np
# import os
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Загрузим нашим данные
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')


# ### Пока будем работать только с датасетом train_data. После его предобработки по аналогии сделаем предобработку для test_data

# In[3]:


# Выведем информацию о нашем датасете
train_data.info()


# In[4]:


# Посмотрим на внешний вид датасета
train_data.head()


# ### Наш датасет состоит из 73799 строк и 19 столбцов. В 6 столбцах строковые величины, в 1 - с плавающей точкой, в остальных - целочиленные

# In[5]:


# Визуализируем количество пустых значений по признакам
plt.figure(figsize=(10,10))
sns.heatmap(train_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# train_data.isnull().sum()


# ### Как мы видим, пустые значения в нашем датасете встречаются только в признаке 'education'

# In[6]:


# Посмотрим, сколько этих значений
train_data.isna().sum()


# In[7]:


# Посмотрим на распределение уникальных значений 'education'
train_data['education'].value_counts(dropna=False)


# ### Мы не знаем какое в действительности образование у тех людей, у которых оно отсутствует, поэтому наиболее вероятным видится присвоить им значение 'SCH', т.к. в школе они точно учились.

# In[8]:


# Присвоим значение 'SCH' тем ячейкам, в которых значение отсутствовало
train_data['education'].fillna('SCH', inplace=True)


# In[9]:


# Проверим, что все пустые значения пропали
train_data.isna().sum()


# In[10]:


# Проверим, что у нас нет задублированных строк в 'client_id'
# Остальные признаки можно не проверять, т.к. по ним значения объективно могут повторяться
train_data['client_id'].duplicated().any()


# In[11]:


num_cols = ['age', 'score_bki', 'decline_app_cnt', 'score_bki', 'bki_request_cnt', 'income']
cat_cols = ['first_time', 'sna', 'work_address', 'home_address', 'region_rating']
bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']


# In[12]:


# Построим визуализацию распределения наших признаков
# fig, axes = plt.subplots(6, 3, figsize=(12,4))
# for i in num_cols:
#     try:
#         plt.figure()
#         plt.hist(np.log(train_data[i]))
#         plt.title(i)
#     except Exception:
#         continue
        
plt.style.use('seaborn-notebook')
plt.figure(figsize=(20,10))

for i in enumerate(num_cols):
    plt.subplot(2, 4, i[0]+1, )
    plt.title(i[1])
    plt.hist(i[1], data=train_data, log=True)


# In[13]:


# Посмотрим на распределение
for i in train_data.drop(['client_id', 'app_date', 'score_bki', 'income'], axis=1).columns:
    print(train_data[i].value_counts(), end='\n')


# In[14]:


# Теперь посмотрим на описательные статистики наших численных данных
# Больше всего нас интересуют 2, 4 и последняя строки - среднее, минимальное и максимальное значения
train_data.describe()


# In[15]:


train_data.head()


# In[16]:


type(train_data['app_date'].iloc[0])


# In[17]:


from datetime import datetime


# In[18]:


def month_to_season(x):
    if x in [12,1,2]:
        return 'winter'
    elif x in [3,4,5]:
        return 'spring'
    elif x in [6,7,8]:
        return 'summer'
    else:
        return 'fall'


# In[19]:


train_data['season'] = train_data['app_date'].apply(lambda x: month_to_season(datetime.strptime(x, '%d%b%Y').month))


# In[20]:


train_data.head()


# In[21]:


datetime.strptime(train_data['app_date'].iloc[0], '%d%b%Y').weekday()


# In[22]:


def weekend_or_not(x):
    if x in [0,1,2,3,4]:
        return 0
    else:
        return 1


# In[23]:


train_data['weekend'] = train_data['app_date'].apply(lambda x: weekend_or_not(datetime.strptime(x, '%d%b%Y').weekday()))


# In[24]:


train_data.head()


# In[25]:


def age_group(x):
    if x >= 60:
        return 'mature'
    elif x >= 30:
        return 'adult'
    else:
        return 'young'


# In[26]:


train_data['age_group'] = train_data['age'].apply(lambda x: age_group(x))


# In[27]:


# train_data['social_group'] = train_data[train_data.columns[[3, 21, 2, 8]].apply(lambda x: ','.join(x))
train_data['social_group'] = train_data[['sex', 'age_group']].apply(lambda x: ', '.join(x.astype(str)), axis=1)


# In[28]:


train_data.head()


# In[29]:


train_data['social_group'].value_counts()


# In[30]:


train_data.drop('social_group', axis=1, inplace=True)


# In[31]:


train_data['income'].value_counts(bins=20)


# In[32]:


def pay_group(x):
    if x > 200_000:
        return 4 #wealthy
    elif x > 140_000:
        return 3 #upper_midclass
    elif x > 80_000:
        return 2 #midclass
    elif x > 50_000:
        return 1 #lower_midclass
    else:
        return 0 #poor


# In[33]:


train_data['income_status'] = train_data['income'].apply(lambda x: pay_group(x))


# In[34]:


train_data[['income', 'income_status']].head(20)


# In[35]:


train_data['income_status'].value_counts()


# In[36]:


train_data.info()


# In[37]:


list(train_data.drop('client_id', axis=1).select_dtypes(exclude=[object]).columns)


# In[38]:


features = list(train_data.drop(['client_id', 'default', 'good_work', 'weekend', 'home_address', 'work_address', 'region_rating'], axis=1).select_dtypes(exclude=[object]).columns)
list(enumerate(features))


# ### Посмотрим на выбросы

# In[39]:


plt.style.use('seaborn-notebook')
plt.figure(figsize=(15,30))

for i in enumerate(features):
    plt.subplot(4, 4, i[0]+1, )
    plt.title(i[1])
    plt.boxplot(i[1], data=train_data)


# ### Теперь, т.к. для нашей прогнозной модели необходимы только численные значения признаков, займемся их преобразованием и нормализацией, т.к. модель Логистической Регрессии чувствительна к ненормализованным данным.

# In[40]:


# Удалим колонки client_id и app_date, т.к. они нам больше не нужны
train_data.drop(['client_id', 'app_date'], axis=1, inplace=True)
train_data.head()


# In[41]:


list(train_data.columns)


# In[42]:


# Создадим список наименований колонок для dummy переменных
dummy_cols = ['education', 'season', 'age_group']


# In[43]:


# Создадим dummy переменные
train_data = pd.get_dummies(data=train_data, columns=dummy_cols)


# In[44]:


train_data


# In[45]:


# Прологарифмируем данные в колонках из списка num_cols
train_data[num_cols] = train_data[num_cols].apply(lambda x: np.log(np.abs(x+1)))


# In[46]:


train_data.head()


# In[47]:


# С помощью LabelEncoder заменить значения в бинарных колонках
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
for i in ['sex', 'car', 'car_type', 'foreign_passport']:
    train_data[i] = encoder.fit_transform(train_data[i])


# In[48]:


train_data.head()


# In[49]:


train_data.info()


# In[50]:


# Импортировать метод для стандартизации всех данных

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()


# In[51]:


# Разделим датасет на предикторы и определеяемую переменную
# Стандартизируем предикторы

train_data_y = train_data['default']

train_data.drop('default', axis=1, inplace=True)

columns = list(train_data.columns)

train_data[columns] = ss.fit_transform(train_data[columns])


# In[52]:


pd.options.display.max_columns=train_data.shape[1]
train_data.head()


# In[53]:


# Вернем в наш стандартизированный датасет поле с определяемой переменной

train_data = train_data.merge(train_data_y, left_index=True, right_index=True)


# In[145]:


# Определим наши X и y для модели
X, y = train_data.drop('default', axis=1).values, train_data['default'].values


# In[146]:


X, y


# ### Hold-out

# In[147]:


# Импортируем метод для разбиения данных
from sklearn.model_selection import train_test_split


# In[148]:


# Разобъем данные на тренировочные и тестовые. Стратифицируем выборки.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)


# In[149]:


# Импортируем метод Логистической Регрессии и укажем ее параметры

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced', C=100, penalty='l1')
# logreg = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced', C=0.0001, penalty='l1')
# logreg = LogisticRegression(C=0.1, class_weight='balanced', dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=1000, multi_class='auto', n_jobs=None, penalty='l1', random_state=None, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)


# In[150]:


logreg.fit(X_train, y_train)


# In[151]:


y_pred = logreg.predict(X_test)


# In[152]:


from sklearn.metrics import f1_score


# In[153]:


f1_score(y_test, y_pred)


# In[143]:


train_data.drop(['income_status','work_address','car_type','education_SCH','season_spring','age_group_young'], axis=1, inplace=True)


# In[144]:


plt.figure(figsize=(20,15))
sns.heatmap(train_data.corr(), annot=True)


# In[ ]:


X, y = train_data.drop('default', axis=1).values, train_data['default'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
f1_score(y_test, y_pred)


# In[ ]:


train_data.drop(['age_group_adult','first_time'], axis=1, inplace=True)


# In[ ]:


plt.figure(figsize=(20,15))
sns.heatmap(train_data.corr(), annot=True)


# In[ ]:


X, y = train_data.drop('default', axis=1).values, train_data['default'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
f1_score(y_test, y_pred)


# In[126]:


# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import make_scorer, f1_score
# f1 = make_scorer(f1_score, average='micro')


# In[125]:


# grid = {'C':np.logspace(-3,3,7), 'penalty':['l1','l2'], 'solver':['liblinear']}
# grid = {'C':[0.0001,0.001,0.01,0.1,1,10,100], 'penalty':['l1','l2'], 'solver':['liblinear']}
# logreg_cv = GridSearchCV(logreg, grid, cv=10, n_jobs=-1)
# logreg_cv.fit(X_train, y_train)
# print('best parameters: ', logreg_cv.best_params_)
# print('f1: ', logreg_cv.best_score_)


# In[123]:


# grid_search = GridSearchCV(logreg, grid, scoring='f1', n_jobs=-1, cv=5)

# grid_search.fit(X_train, y_train)

# mod = grid_search.best_estimator_

# best_parameters = mod.get_params()
# for param_name in sorted(best_parameters.keys()):
#     print('\t%s: %r' % (param_name, best_parameters[param_name]))


# ### Проделаем все те же действия для датасета test и запишем наши предсказанные значения в submission

# In[154]:


test_data.head()


# In[159]:


test_data['client_id'].duplicated().any()


# In[160]:


test_data['education'].fillna('SCH', inplace=True)

test_data['season'] = test_data['app_date'].apply(lambda x: month_to_season(datetime.strptime(x, '%d%b%Y').month))

test_data['weekend'] = test_data['app_date'].apply(lambda x: weekend_or_not(datetime.strptime(x, '%d%b%Y').weekday()))

test_data['age_group'] = test_data['age'].apply(lambda x: age_group(x))

test_data['income_status'] = test_data['income'].apply(lambda x: pay_group(x))

test_data.drop(['client_id', 'app_date'], axis=1, inplace=True)

test_data = pd.get_dummies(data=test_data, columns=dummy_cols)

test_data[num_cols] = test_data[num_cols].apply(lambda x: np.log(np.abs(x+1)))

for i in ['sex', 'car', 'car_type', 'foreign_passport']:
    test_data[i] = encoder.fit_transform(test_data[i])

columns = list(test_data.columns)

test_data[columns] = ss.fit_transform(test_data[columns])

test_data.drop(['income_status','work_address','car_type','education_SCH','season_spring','age_group_young','age_group_adult','first_time'], axis=1, inplace=True)

predict_submission = logreg.predict(test_data.values)


# In[161]:


predict_submission


# In[164]:


sample_submission.default.unique()


# In[165]:


sample_submission['default'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)


# In[ ]:




