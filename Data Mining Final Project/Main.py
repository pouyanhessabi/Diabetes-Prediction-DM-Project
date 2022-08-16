import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import sklearn as sk
from sklearn.model_selection import GridSearchCV


def pre_processing():
    data_frame = remove_white_space(read_date())
    data_frame = clean_data_frame(data_frame)
    data_frame = normalize(data_frame)
    data_frame = categorical_features(data_frame)
    labeled_diabetes = data_frame['Diabetes_binary']
    data_frame.drop('Diabetes_binary', axis=1, inplace=True)
    data_frame.drop('I_Unknown', axis=1, inplace=True)

    return labeled_diabetes, data_frame


def read_date():
    data_frame = pd.read_csv('diabetes.csv')
    data_frame.drop('Row', axis=1, inplace=True)
    return data_frame


def remove_white_space(data_frame):
    for attribute in data_frame.columns.values:
        data_frame.rename(columns={attribute: attribute.replace(" ", "_")}, inplace=True)
    data_frame['General_Health'] = data_frame['General_Health'].replace(['Very Low'], 'Very_Low')
    return data_frame


def clean_data_frame(data_frame):
    """
    1- fill null values
    2- edit wrong values
    :param data_frame:
    :return: data_frame
    """
    data_frame['Diabetes_binary'].fillna(data_frame['Diabetes_binary'].median(), inplace=True)
    data_frame['HighBP'].fillna(data_frame['HighBP'].median(), inplace=True)
    data_frame['High_Cholesterol'].fillna(data_frame['High_Cholesterol'].median(), inplace=True)
    data_frame['Cholesterol_Check'].fillna(data_frame['Cholesterol_Check'].median(), inplace=True)
    data_frame['BMI'].fillna(data_frame['BMI'].median(), inplace=True)
    data_frame['Smoker'].fillna(data_frame['Smoker'].median(), inplace=True)
    data_frame['Stroke'].fillna(data_frame['Stroke'].median(), inplace=True)
    data_frame['HeartDiseaseorAttack'].fillna(data_frame['HeartDiseaseorAttack'].median(), inplace=True)
    data_frame['Physical_Activity'].fillna(data_frame['Physical_Activity'].median(), inplace=True)
    data_frame['Fruits'].fillna(data_frame['Fruits'].median(), inplace=True)
    data_frame['Veggies'].fillna(data_frame['Veggies'].median(), inplace=True)
    data_frame['Heavy_Alcohol_Consumption'].fillna(data_frame['Heavy_Alcohol_Consumption'].median(), inplace=True)
    data_frame['Any_Health_Care'].fillna(data_frame['Any_Health_Care'].median(), inplace=True)
    data_frame['No_Doctor_because_of_Cost'].fillna(data_frame['No_Doctor_because_of_Cost'].median(), inplace=True)
    data_frame['General_Health'].fillna('Medium', inplace=True)
    data_frame['Mental_Health'].fillna(data_frame['Mental_Health'].median(), inplace=True)
    data_frame['Physical_Health'].fillna(data_frame['Physical_Health'].median(), inplace=True)
    data_frame['Difficulty_Walking'].fillna(data_frame['Difficulty_Walking'].median(), inplace=True)
    data_frame['Sex'].fillna('male', inplace=True)
    data_frame['Age'].fillna(data_frame['Age'].median(), inplace=True)
    data_frame['Education'].fillna('Cat5', inplace=True)
    data_frame['Income'].fillna('Cat7', inplace=True)

    data_frame['BMI'].mask(data_frame['BMI'] > 45, 45, inplace=True)

    return data_frame


def normalize(data_frame):
    data_frame['BMI'] = np.where((data_frame['BMI'] > 0) & (data_frame['BMI'] <= 18.5), 1, data_frame['BMI'])
    data_frame['BMI'] = np.where((data_frame['BMI'] > 18.5) & (data_frame['BMI'] < 25), 2, data_frame['BMI'])
    data_frame['BMI'] = np.where((data_frame['BMI'] >= 25) & (data_frame['BMI'] < 30), 3, data_frame['BMI'])
    data_frame['BMI'] = np.where((data_frame['BMI'] >= 30), 4, data_frame['BMI'])

    data_frame['Mental_Health'] = np.where((data_frame['Mental_Health'] >= 0) & (data_frame['Mental_Health'] < 10),
                                           1, data_frame['Mental_Health'])
    data_frame['Mental_Health'] = np.where((data_frame['Mental_Health'] >= 10) & (data_frame['Mental_Health'] < 20),
                                           2, data_frame['Mental_Health'])
    data_frame['Mental_Health'] = np.where((data_frame['Mental_Health'] >= 20) & (data_frame['Mental_Health'] < 30),
                                           3, data_frame['Mental_Health'])
    data_frame['Mental_Health'] = np.where((data_frame['Mental_Health'] >= 30), 4, data_frame['Mental_Health'])

    data_frame['Physical_Health'] = np.where(
        (data_frame['Physical_Health'] >= 0) & (data_frame['Physical_Health'] < 10), 1, data_frame['Physical_Health'])
    data_frame['Physical_Health'] = np.where(
        (data_frame['Physical_Health'] >= 10) & (data_frame['Physical_Health'] < 20), 2, data_frame['Physical_Health'])
    data_frame['Physical_Health'] = np.where(
        (data_frame['Physical_Health'] >= 20) & (data_frame['Physical_Health'] < 30), 3, data_frame['Physical_Health'])
    data_frame['Physical_Health'] = np.where((data_frame['Physical_Health'] >= 30), 4, data_frame['Physical_Health'])

    return data_frame


def categorical_features(data_frame):
    separated = pd.get_dummies(data_frame['Sex'], prefix="S")
    data_frame = data_frame.drop('Sex', axis=1)
    data_frame = data_frame.join(separated)

    separated = pd.get_dummies(data_frame['General_Health'], prefix="GH")
    data_frame = data_frame.drop('General_Health', axis=1)
    data_frame = data_frame.join(separated)

    separated = pd.get_dummies(data_frame['Education'], prefix="E")
    data_frame = data_frame.drop('Education', axis=1)
    data_frame = data_frame.join(separated)

    separated = pd.get_dummies(data_frame['Income'], prefix="I")
    data_frame = data_frame.drop('Income', axis=1)
    data_frame = data_frame.join(separated)
    return data_frame


def classifier(label_data, data_frame):
    train, test, labeled_train, labeled_test = sk.model_selection.train_test_split(data_frame, label_data,
                                                                                   test_size=0.75, random_state=1)
    xgbc = XGBClassifier(Learning_rate=0.1, Max_depth=4, N_estimator=200, Subsample=0.5, Colsample_bytree=1,
                         Random_seed=123, Eval_metric='auc', Verbosity=1)
    # print("All params: ", str(XGBClassifier.get_params(xgbc)))
    xgbc.fit(train, labeled_train)
    test_prediction = xgbc.predict(test)
    train_prediction = xgbc.predict(train)
    return labeled_train, train_prediction, labeled_test, test_prediction


def model_evaluation(label, prediction, is_test: bool):
    print("This is for ", ("test" if is_test else "train"))
    accuracy = sk.metrics.accuracy_score(label, prediction)
    confusion = sk.metrics.confusion_matrix(label, prediction)
    precision = (confusion[0][0] + confusion[1][1]) / (
            confusion[0][0] + confusion[1][1] + confusion[0][1] + confusion[1][0])
    recall = confusion[0][0] / (confusion[0][0] + confusion[1][0])
    print("Confusion Matrix: ", str(confusion), "\nAccuracy: ", str(accuracy), "\nprecision: ", str(precision),
          "\nrecall: ", str(recall))


labeled, df = pre_processing()
l_train, train_predict, l_test, test_predict = classifier(labeled, df)

print("Part 2:")
model_evaluation(l_train, train_predict, False)
model_evaluation(l_test, test_predict, True)
print("|||||||")

learning_rates = [0.02, 0.05, 0.1, 0.3]
max_depths = [2, 3, 4]
n_estimators = [100, 200, 300]
colsample_bytrees = [0.8, 1]
hyper_parameters = [learning_rates, max_depths, n_estimators, colsample_bytrees]
param_grid = dict(learning_rate=learning_rates, max_depth=max_depths, n_estimators=n_estimators,
                  colsample_bytree=colsample_bytrees)
cv_k_fold = sk.model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
the_model = XGBClassifier(eval_metric='auc', subsample=0.5)
grid_search_result = GridSearchCV(the_model, param_grid, scoring='roc_auc', n_jobs=-1, cv=cv_k_fold).fit(df, labeled)
print(grid_search_result)
print("Best parameters: ", str(grid_search_result.best_params_))
train_data, test_data, label_train, label_test = sk.model_selection.train_test_split(df, labeled, test_size=0.75,
                                                                                     random_state=1)
the_model.fit(train_data, label_train)
predict_label_test = the_model.predict(test_data)
predict_label_train = the_model.predict(train_data)
model_evaluation(label_test, predict_label_test, True)
model_evaluation(label_train, predict_label_train, False)


