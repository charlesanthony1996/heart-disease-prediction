
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')


df = pd.read_csv('heart.csv')

# print(df.head())

# print(df.shape)

# print(df.info())

# print(df.describe())

"""

for column in df.columns:
    if(df[column].dtype == 'object'):
        print(
            f'The unique values in column "{column}" are: {df[column].unique()}\n')

"""

"""
for column in df.columns:
    if(df[column].dtype == 'object'):
        sns.countplot(df[column], hue=df['HeartDisease'])
        plt.show()
"""

"""
sns.countplot(df['FastingBS'], hue=df['HeartDisease'])
plt.show()

"""


"""

sns.countplot(df['HeartDisease'])
plt.show()

"""


"""
for column in df.columns:
    if((df[column].dtype == 'int64') | (df[column].dtype == 'float64')):
        sns.histplot(df[column])
        plt.show()
"""

"""

for column in df.columns:
    if((df[column].dtype == 'int64') | (df[column].dtype == 'float64')):
        sns.boxplot(df[column])
        plt.show()

"""


# sns.pairplot(df, hue='HeartDisease')


# data cleaning

numerical_col = []
for column in df.columns:
    if((df[column].dtype != 'object') & (len(df[column].unique()) > 2)):
        numerical_col.append(column)


# print(numerical_col)


for column in numerical_col:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    df = df[(df[column] > lower) & (df[column] < upper)]


# lets check the outliers again

"""
for column in numerical_col:
    sns.boxplot(df[column])
    plt.show()

"""


# sns.pairplot(df, hue='HeartDisease')


# print(df.info())


for column in numerical_col:
    scale = MinMaxScaler()
    df[column] = scale.fit_transform(df[column].values.reshape(-1, 1))


# print(df.head())

# lets first differentiate the categorical columns

categorical_col = df.drop(numerical_col, axis=1)
# print(categorical_col)


for column in categorical_col.drop('HeartDisease', axis=1):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

# print(df.head())


# print(df.corr())


plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True)
# plt.show()


best = SelectKBest(chi2, k=8)

best.fit(df.drop('HeartDisease', axis=1), df['HeartDisease'])


# print(best.scores_)


X_train, X_test, y_train, y_test = train_test_split(
    df[['Age', 'Sex', 'ChestPainType', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']], df['HeartDisease'], test_size=0.2)


# knn classifier

params_knn = [{'n_neighbors': [3, 4, 5, 6, 7, 8,
                               9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}]
model_knn = KNeighborsClassifier()
cv_knn = GridSearchCV(estimator=model_knn, param_grid=params_knn)

cv_knn.fit(X_train, y_train)

y_predict_knn = cv_knn.predict(X_test)


sns.heatmap(confusion_matrix(y_test, y_predict_knn), annot=True)

#print(classification_report(y_test, y_predict_knn))


#print(roc_auc_score(y_test, y_predict_knn))

params_rf = [{'n_estimators': [10, 20, 30, 40, 50, 60,
                               70, 80, 90, 100], 'max_depth':list(range(1, 20))}]

cv_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params_rf)

cv_rf.fit(X_train, y_train)

y_predict_cv_rf = cv_rf.predict(X_test)

sns.heatmap(confusion_matrix(y_test, y_predict_cv_rf), annot=True)
