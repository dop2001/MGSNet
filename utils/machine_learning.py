import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from dataloader import FeatureDataset
from random_state import RandomState
from metrics import getMetrics
import warnings
import csv
warnings.filterwarnings('ignore')


def get_dataset(data_path=r'../datasets/FMCC.csv', mode='train', ratio=0.8):
    dataloader = FeatureDataset(dataset_path=data_path, mode=mode, ratio=ratio)
    data, label = [], []
    for x, y in dataloader:
        data.append(np.array(x).tolist())
        label.append(np.array(y).tolist())
    data = np.stack(data, axis=0)
    label = np.stack(label, axis=0)

    return data, label


def train_test_dataset(data_path=r'../datasets/FMCC.csv', ratio=0.8):
   x_train, y_train = get_dataset(data_path=data_path, mode='train', ratio=ratio)
   x_test, y_test = get_dataset(data_path=data_path, mode='test', ratio=ratio)
   return x_train, y_train, x_test, y_test


def evaluate_model(model, x_train, y_train, x_test, y_test, scaled=False):
    if scaled:
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = getMetrics(y_true=y_test, y_pred=predictions)

    return metrics

def getModel():
    model_dict = {
        'Extra Trees Classifier': ExtraTreesClassifier(),
        'Random Forest Classifier': RandomForestClassifier(),
        # 'Light Gradient Boosting Machine': lgb(),
        'Quadratic Discriminant Asnalysis': QuadraticDiscriminantAnalysis(),
        'K Neighbors Classifier': KNeighborsClassifier(),
        'Decision Tree Classifier': DecisionTreeClassifier(),
        'Gradient Boosting Classifier':GradientBoostingClassifier(),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        # 'Ridege Classifier': Ridge(),
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(probability=True),
        'Naive Bayes': GaussianNB(),
        'Dummy Classifier': DummyClassifier()
    }
    return model_dict


if __name__ == '__main__':
    randomState = RandomState(seed=1)
    x_train, y_train, x_test, y_test = train_test_dataset(data_path=r'../datasets/test1_000.csv')

    model_dict = getModel()

    with open(r'../results/ML_metrics.csv', 'w', newline='') as f:
        head = ['type', 'acc', 'precision', 'recall', 'f1', 'specificity', 'auc', 'mcc']
        writer = csv.writer(f)
        writer.writerow(head)

        for key, val in model_dict.items():
            metrics = evaluate_model(val, x_train, y_train, x_test, y_test, scaled=False)
            print('{} | {}'.format(key, metrics))
            temp = [key]
            for name in head[1:]:
                temp.append(metrics[name])
            writer.writerow(temp)



    # evaluate_model(SVC(probability=True), x_train, y_train, x_test, y_test, scaled=False)
    # evaluate_model(LogisticRegression(max_iter=1000), x_train, y_train, x_test, y_test, scaled=True)
    # evaluate_model(RandomForestClassifier(), x_train, y_train, x_test, y_test, scaled=True)
    # evaluate_model(KNeighborsClassifier(), x_train, y_train, x_test, y_test, scaled=True)
    # evaluate_model(DecisionTreeClassifier(), x_train, y_train, x_test, y_test, scaled=True)
    # evaluate_model(ExtraTreesClassifier(), x_train, y_train, x_test, y_test, scaled=False)
    # evaluate_model(GradientBoostingClassifier(), x_train, y_train, x_test, y_test, scaled=False)
    # evaluate_model(LinearDiscriminantAnalysis(), x_train, y_train, x_test, y_test, scaled=True)
    # # evaluate_model(Ridge(), x_train, y_train, x_test, y_test, scaled=True)
    # evaluate_model(GaussianNB(), x_train, y_train, x_test, y_test, scaled=True)
    # evaluate_model(DummyClassifier(), x_train, y_train, x_test, y_test, scaled=True)
    # evaluate_model(QuadraticDiscriminantAnalysis(), x_train, y_train, x_test, y_test, scaled=True)
    # evaluate_model(lgb(), x_train, y_train, x_test, y_test, scaled=True)


