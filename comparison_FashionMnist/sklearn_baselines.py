import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

def sklearn_comparison(train_dataset, test_dataset, use_svm=False):
    # Flatten image data
    X_train = train_dataset.data.numpy().reshape(-1, 28*28)
    y_train = train_dataset.targets.numpy()

    X_test  = test_dataset.data.numpy().reshape(-1, 28*28)
    y_test  = test_dataset.targets.numpy()
    
    classifiers = {
        'DecisionTree': DecisionTreeClassifier(),
        'NaiveBayes':   GaussianNB(),
        'LDA':          LinearDiscriminantAnalysis(),
        'RandomForest': RandomForestClassifier(),
        'AdaBoost':     AdaBoostClassifier(),
    }
    
    if use_svm:
        classifiers['SVM'] = SVC(kernel="rbf", gamma=0.5, C=1.0)

    results = {}
    for name, clf in classifiers.items():
        print(f'Processing {name}...')
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        results[name] = score
        print(f"{name}: {score*100:.2f}%")
        print('---------------------------------------')

    return results
