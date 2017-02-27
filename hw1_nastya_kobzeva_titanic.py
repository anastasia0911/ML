# coding=utf-8
import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score

data = pandas.read_csv('titanic.csv', index_col='PassengerId')
# print(data.head())
# print(data.describe())

# 1.1. Вероятность выжить для мужчин и женщин
plt.figure()
sns.barplot(x="Sex", y="Survived", data=data)
''' Средняя вероятность выжить для женщин выше, чем для мужчин - примерно 74%
и 19% соответственно. '''

# 1.2. Вероятность выжить для пассажиров разных социально-экономических классов
plt.figure()
sns.barplot(x="Pclass", y="Survived", data=data)
''' Наибольшая вероятность выжить - для пассажиров первого класса, примерно 63%,
для пассажиров второго класса - примерно 47%, для пассажиров третьего класса -
24%. Таким образом, чем выше социально-экономический класс пассажира, тем
больше вероятность, что его посадят в лодку и он выживет.'''

# 1.3. Cтоимость билета в зависимости от социально-экономического класса
plt.figure()
sns.barplot(x="Pclass", y="Fare", data=data)
''' Как и следовало ожидать, билеты пассажиров первого класса оказались самыми
дорогими, при этом в этой группе наблюдается самый большой разброс по цене; в
среднем, билет первого класса можно было купить примерно за 84 у.е.(я погуглила:
настоящие цены были в фунтах на порядок выше, поэтому пишу у.е.), билет второго
класса в среднем стоял 20 у.е. (при этом разброс меньше, чем в первом случае),
третьего - 13 (разбос совсем маленький).'''

# 2. Средняя вероятность выжить в зависимости от пола и соц. статуса
plt.figure()
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=data)
''' В среднем, вероятность выжить для женщин оказалась выше, чем для мужчин,вне
зависимоти от их класса, P(A)>P(B), где А - "выживет женщина" и В - "выживет
мужчина". При этом P(A)>P(B)>P(C), где А, В и С - это события
"выживет женщина/мужчина из 1го, 2го и 3го класса соответственно".'''

# 3. Преобразование данных
print("Пустых ячеек в столбце 'Возраст':", data['Age'].isnull().sum())
# 177 пустых ячеек с возрастом, вставляю в них значение медианы
data['Age'].fillna((data['Age'].mean()), inplace=True)

# Избавляюсь от категориальных данных в графе 'Пол'
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
x_labels = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare']
# 6 фичей: 2 дополнительные, чтобы посмотреть, влияет ли наличие родственников
X, y = data[x_labels], data['Survived']

# 4. Деление выборки на обучающую и тестовую и дерево решений
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print(X_test.describe())
scores = {}
arr = []
''' Я решила посмотреть, какое минимальное число экземпляров оптимально для
листа дерева; этот параметр также связан с глубиной дерева. Он должен не быть
слишком маленьким, иначе дерево переобучится, но и не слишком большим - иначе
дерево недообучится. '''
for x in range(2, 8):
    # print('Min samples split =', x)
    clf = DecisionTreeClassifier(min_samples_split=x)
    clf.fit(np.array(X_train), np.array(y_train))
    importances = pandas.Series(clf.feature_importances_, index=x_labels)
    # print(importances)
    y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred))
    # print(clf.score(np.array(X_test), np.array(y_test)))
    if x not in scores:
        scores[x] = clf.score(np.array(X_test), np.array(y_test))
        arr.append(clf.score(np.array(X_test), np.array(y_test)))
df_scores = pandas.DataFrame.from_dict(data=scores, orient='index')
df_scores.plot(kind='bar', legend=False)
plt.xlabel('min_samples_split')
plt.ylabel('f1_score)')
best_split = 5  # пусть будет по умолчанию так
for k, v in scores.items():
    if v == max(arr):  # выбираю лучшее значение f-меры и строю дерево
        best_split = k

dt_clf = DecisionTreeClassifier(min_samples_split=best_split)
dt_clf.fit(np.array(X_train), np.array(y_train))
importances = pandas.Series(dt_clf.feature_importances_, index=x_labels)
print('\nDecision Tree')
print('Best min_samples_split_is', best_split)  # лучший обычно - 5, 6 или 7
# print(importances)
y_pred = dt_clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(dt_clf.score(np.array(X_test), np.array(y_test)))

''' Никакими возможными способами, описанными в тьюториале, у меня не получилось
создать pdf, и даже 5 статей на Stackoverflow не помогли. Поэтому ограничусь
здесь созданием dot-файла. '''
with open("titanic.dot", 'w') as f:
    f = export_graphviz(dt_clf, out_file=f)

# 5. Random Forest
scores = []
d = {}
for t in range(1, 100):
    rfc = RandomForestClassifier(n_estimators=t)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    scores.append(f1_score(y_test, y_pred))
    if t not in d:
        d[t] = f1_score(y_test, y_pred)
plt.figure()
plt.plot(scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
# plt.show() # если раскомментить эту строку, появятся все графики (все 6)
best_estimator = 70  # пусть будет по умолчанию так
for k, v in d.items():
    if v == max(scores):
        best_estimator = k
print('\nRandom Forest')
print('Best estimators number is', best_estimator)
model = RandomForestClassifier(n_estimators=best_estimator)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
''' В целом, Random Forest - лучший классификатор, чем Decision Tree. F-мера RF,
как правило, выше 80, тогда как для DT - около 80. '''
