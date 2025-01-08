import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('diabetes_.csv')
df.head()  # first 5 records

df.shape  #rows & columns count, finding all the data is present

df.info()  #datatype of the variables

df.nunique()

# To reduce the complexity of the model, converting non-numerical data into numerical

df.tail()  #last 5 records

df.isnull().sum()  # finding the missing data

df = df.fillna(df.mode().iloc[0])  #updating the null values with frequent values
df.isnull().sum()  # validating whether any data is missed

df.describe()


# finding the correlation between the features
corr = df.astype(float).corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr,
            linewidths=0.1,
            vmax=1.0,
            square=True,
            linecolor='white',
            annot=True,
            cmap="Blues",
            mask=np.triu(corr))
plt.show()


df.head()


import random
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report,f1_score, accuracy_score,roc_curve,roc_auc_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

X = df.iloc[:, :-1].values  # taking all rows , all columns except last column as independent variables(features)
y = df.iloc[:,-1]  # taking all rows, last column as dependent variable(labels) ie. Diabetes

#splitting training & testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#defining all necessary models for classification
def getBaseModels():
    baseModels = []
    baseModels.append(('KNN', KNeighborsClassifier()))
    baseModels.append(('DT', DecisionTreeClassifier()))
    baseModels.append(('SVC', SVC(probability=True)))
    baseModels.append(('AB', AdaBoostClassifier()))
    baseModels.append(('GB', GradientBoostingClassifier()))
    baseModels.append(('RF', RandomForestClassifier()))
    baseModels.append(('ET', ExtraTreesClassifier()))
    return baseModels

#using k-fold on training data to evaluate the model accuracy
models = getBaseModels()
modelScores = []
modelStd = []
modelNameList = []
for name, model in models:
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model,X_train,y_train,cv=cv,scoring="accuracy",error_score="raise")
    modelScores.append(cv_results.mean()*100)
    modelNameList.append(name)
    modelStd.append(cv_results.std())

modelScores = np.round(modelScores, 2)
modelStd = np.round(modelStd, 2)
modelResult = pd.DataFrame({
    'Models': modelNameList,
    'Scores': modelScores,
    'σ': modelStd
})
print(modelResult.sort_values(by=['Scores', 'σ'], ascending=False))

#visualing the models performance
plt.title('Algorithm Comparison')
pal = sns.color_palette("Blues", len(modelScores)+2)
rank = modelScores.argsort().argsort()
ax=sns.barplot(y=modelScores, x=modelNameList, palette=np.array(pal[::-1])[rank])
ax.bar_label(ax.containers[0])
plt.show()

def gridSearchFunction(model, params, cv, X_val, y_val):
    gs = GridSearchCV(model, param_grid=params, cv=cv)
    gs.fit(X_val, y_val)
    print("accuracy: %.4f" % gs.best_score_)
    print("best params:", gs.best_params_)
    return gs.best_params_
### KNN
# n_neighbors: Number of neighbors to use by default for k_neighbors queries


param_grid = {
    "n_neighbors":
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}
model = KNeighborsClassifier()
bestParams = gridSearchFunction(model, param_grid, 5, X_val, y_val)

### SVC
# C: The Penalty parameter C of the error term.
# Kernel: Kernel type could be linear, poly, rbf or sigmoid.


param_grid = {
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
}
model = SVC()
bestParams = gridSearchFunction(model, param_grid, 5, X_val, y_val)

### Decision Tree
# max_depth: Maximum depth of the tree (double).
# row_subsample: Proportion of observations to consider (double).
# max_features: Proportion of columns (features) to consider in each level (double).


param_grid = {
    "max_depth": [3, None],
    'max_features': [random.randint(1, 4)],
    'min_samples_leaf': [random.randint(1, 4)],
    'criterion': ["gini", "entropy"]
}
model = DecisionTreeClassifier()
bestParams = gridSearchFunction(model, param_grid, 5, X_val, y_val)

### Random forest
# n_estimators: The number of trees in the forest.
# max_depth: The maximum depth of the tree


param_grid = {"max_depth": [5, 10, None], "n_estimators": [50, 100, 200]}
model = RandomForestClassifier()
bestParams = gridSearchFunction(model, param_grid, 5, X_val, y_val)

### AdaBoostClassifier
# learning_rate: Learning rate shrinks the contribution of each classifier by learning_rate.
# n_estimators: Number of trees to build.


param_grid = {
    "learning_rate": [.01, .05, .1, .5, 1],
    'n_estimators': [50, 100, 150, 200, 250, 300]
}
model = AdaBoostClassifier()
bestParams = gridSearchFunction(model, param_grid, 5, X_val, y_val)

### GradientBoostingClassifier
# learning_rate: Learning rate shrinks the contribution of each classifier by learning_rate.
# n_estimators: Number of trees to build.


param_grid = {
    "learning_rate": [.01, .05, .1, .5, 1],
    'n_estimators': [50, 100, 150, 200, 250, 300]
}
model = GradientBoostingClassifier()
bestParams = gridSearchFunction(model, param_grid, 5, X_val, y_val)

#updating the models with best parameters

param = {'n_neighbors': 12}
model1 = KNeighborsClassifier(**param)

param = {'C': 0.5, 'kernel': 'sigmoid'}
model2 = SVC(probability=True, **param)

param = {
    'criterion': 'gini',
    'max_depth': None,
    'max_features': 1,
    'min_samples_leaf': 3
}
model3 = DecisionTreeClassifier(**param)

param = {'max_depth': None, 'n_estimators': 100}
model4 = RandomForestClassifier(**param)

param = {'learning_rate': 0.01, 'n_estimators': 250}
model5 = AdaBoostClassifier(**param)

param = {'learning_rate': 0.01, 'n_estimators': 100}
model6 = GradientBoostingClassifier(**param)

param = {'max_depth': 5, 'n_estimators': 200}
model7 = ExtraTreesClassifier(**param)
def updatedBaseModels():
    baseModels = []
    baseModels.append(('KNN', model1))
    baseModels.append(('SVC', model2))
    baseModels.append(('DT', model3))
    baseModels.append(('RF', model4))
    baseModels.append(('AB', model5))
    baseModels.append(('GB', model6))
    baseModels.append(('ET', model7))
    return baseModels

models = updatedBaseModels()
modelScores = []
modelStd = []
modelNameList = []
for name, model in models:
    cv_results = cross_val_score(model,X_train,y_train,scoring="accuracy",error_score="raise")
    modelScores.append(cv_results.mean()*100)
    modelNameList.append(name)
    modelStd.append(cv_results.std())

modelScores = np.round(modelScores, 2)
modelStd = np.round(modelStd, 2)
modelResult = pd.DataFrame({
    'Models': modelNameList,
    'Scores': modelScores,
    'σ': modelStd
})
print(modelResult.sort_values(by=['Scores', 'σ'], ascending=False))

#visualing the models performance
plt.title('Algorithm Comparison')
pal = sns.color_palette("Blues", len(modelScores)+2)
rank = modelScores.argsort().argsort()
ax=sns.barplot(y=modelScores, x=modelNameList, palette=np.array(pal[::-1])[rank])
ax.bar_label(ax.containers[0])
plt.show()

# predicts the dependent value based on training
def model_predict(classifier, X_train, y_train, X_test):
    model = classifier.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


def display_result(modelName, classifier, X_train, y_train, X_test, y_test):
    y_pred = model_predict(classifier, X_train, y_train, X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
    plt.title(modelName)
    plt.show()
    accuracy = np.round(accuracy_score(y_test, y_pred), 2)
    auc = np.round(roc_auc_score(y_test, y_pred), 2)
    f1 = np.round(f1_score(y_test, y_pred, average='macro'), 2)
    roc = metrics.roc_curve(y_test, y_pred)
    result = pd.DataFrame({
        'Model': [modelName],
        'Accuracy': [accuracy],
        'AUC Score': [auc],
        'F1 Score': [f1],
        'ROC': [roc]
    })
    return result


def compareResults(modelList):
    frames = []
    for model in modelList:
        frames.append(model)
    result = pd.concat(frames)
    result = pd.melt(frame=result.iloc[:, 0:-1], id_vars='Model', var_name='Statistic', value_name='value')
    ax = sns.barplot(data=result, x='Model', y='value', hue='Statistic', palette='Blues')
    sns.move_legend(ax, "lower right")
    for i in ax.containers:
        ax.bar_label(i, )

    # roc curve
    plt.figure(0).clf()
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    for model in modelList:
        roc = model['ROC'].values
        fpr, tpr, _ = roc[0]
        auc = model['AUC Score'].values
        plt.plot(fpr, tpr, label=model.iloc[:, 0][0] + ', AUC=' + str(auc))
    plt.legend(loc='lower right')
    plt.show()

from sklearn.ensemble import VotingClassifier
baseModels = updatedBaseModels()
votingModel = VotingClassifier(baseModels, voting='hard')
votingRes=display_result('Voting', votingModel, X_train, y_train, X_test, y_test)

from sklearn.ensemble import StackingClassifier
baseModels = updatedBaseModels()
baseModels.remove(('GB', model6)) #removing gradient boosting from basemodels since using GB as metamodel
metaModel = GradientBoostingClassifier(n_estimators=150,
                                       loss="exponential",
                                       max_features=6,
                                       max_depth=3,
                                       subsample=0.5,
                                       learning_rate=0.01)
stackModel = StackingClassifier(estimators=baseModels,
                                final_estimator=metaModel)
stackingRes=display_result('Stacking', stackModel, X_train, y_train, X_test, y_test)

models=[votingRes,stackingRes]
compareResults(models)




