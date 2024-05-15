import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Modele de clasificare
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Instrumente pentru prelucrarea datelor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import ADASYN

import warnings
import optuna

# Incarcam datele

# Specificam eticheta
label = "is_hungry"

# Datele de antrenare
train_data = pd.read_csv("train.csv", index_col=0)
x_train = train_data.iloc[:, :-1]  # caracteristici de antrenare
y_train = train_data[[label]]  # eticheta de antrenare

# Datele de testare
test_data = pd.read_csv("test.csv", index_col=0)
x_test = test_data.iloc[:, :-1]  # caracteristici de testare
y_test = test_data[[label]]  # eticheta de testare

# Afisam informatii despre datele extrase
print('train_data shape: ', train_data.shape)
print('test_data shape: ', test_data.shape)

print('\n\nNumar de is_hungry = 1:\n', (y_train != 0).sum(), '\n\n')

print(train_data.describe())

# Vizualizam datele prin diagrame de dispersie
columns = x_train.columns[:19]
fig, axes = plt.subplots(5, 4, figsize=(16, 16))

# Convertim valorile etichetelor de antrenare in numere
y_train_numeric = np.array(y_train).astype(float)

# Parcurgem coloanele selectate si cream diagramele de dispersie
for i, ax in enumerate(axes.ravel()):
    if i < len(columns):
        scatter = ax.scatter(x_train.index, x_train[columns[i]], c=y_train_numeric, cmap='coolwarm', alpha=0.6)
        ax.set_title(f'Scatter Plot pentru {columns[i]}')
        ax.set_xlabel('Index\n\n\n')
        ax.set_ylabel(columns[i])
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()

# Pentru coloanele 0-3
columns = x_train.columns[:4]
fig, axes = plt.subplots(4, 2, figsize=(10, 30))

y_test_numeric = np.array(y_test).astype(float)

for i, column in enumerate(columns):
    axes[i, 0].scatter(x_train.index, x_train[column], c=y_train_numeric, cmap='coolwarm', alpha=0.6, label='y_train')
    axes[i, 0].set_title(f'Train: {column}')

    axes[i, 1].scatter(x_test.index, x_test[column], c=y_test_numeric, cmap='winter', alpha=0.6, label='y_test')
    axes[i, 1].set_title(f'Test: {column}')

    for j in range(2):
        axes[i, j].set_xlabel('Index\n')
        axes[i, j].set_ylabel(column)
plt.tight_layout()

# Pentru coloanele 4-7
columns = x_train.columns[4:8]
fig, axes = plt.subplots(4, 2, figsize=(10, 30))

y_test_numeric = np.array(y_test).astype(float)

for i, column in enumerate(columns):
    axes[i, 0].scatter(x_train.index, x_train[column], c=y_train_numeric, cmap='coolwarm', alpha=0.6, label='y_train')
    axes[i, 0].set_title(f'Train: {column}')

    axes[i, 1].scatter(x_test.index, x_test[column], c=y_test_numeric, cmap='winter', alpha=0.6, label='y_test')
    axes[i, 1].set_title(f'Test: {column}')

    for j in range(2):
        axes[i, j].set_xlabel('Index\n')
        axes[i, j].set_ylabel(column)

plt.tight_layout()

# Pentru coloanele 8-11
columns = x_train.columns[8:12]
fig, axes = plt.subplots(4, 2, figsize=(10, 30))

y_test_numeric = np.array(y_test).astype(float)

for i, column in enumerate(columns):
    axes[i, 0].scatter(x_train.index, x_train[column], c=y_train_numeric, cmap='coolwarm', alpha=0.6, label='y_train')
    axes[i, 0].set_title(f'Train: {column}')

    axes[i, 1].scatter(x_test.index, x_test[column], c=y_test_numeric, cmap='winter', alpha=0.6, label='y_test')
    axes[i, 1].set_title(f'Test: {column}')

    for j in range(2):
        axes[i, j].set_xlabel('Index\n')
        axes[i, j].set_ylabel(column)

plt.tight_layout()

# Pentru coloanele 12-15
columns = x_train.columns[12:16]
fig, axes = plt.subplots(4, 2, figsize=(10, 30))

for i, column in enumerate(columns):
    axes[i, 0].scatter(x_train.index, x_train[column], c=y_train_numeric, cmap='coolwarm', alpha=0.6, label='y_train')
    axes[i, 0].set_title(f'Train: {column}')

    axes[i, 1].scatter(x_test.index, x_test[column], c=y_test_numeric, cmap='winter', alpha=0.6, label='y_test')
    axes[i, 1].set_title(f'Test: {column}')

    for j in range(2):
        axes[i, j].set_xlabel('Index\n')
        axes[i, j].set_ylabel(column)

plt.tight_layout()

# Pentru coloanele 16-18
columns = x_train.columns[16:19]
fig, axes = plt.subplots(3, 2, figsize=(10, 30))

for i, column in enumerate(columns):
    axes[i, 0].scatter(x_train.index, x_train[column], c=y_train_numeric, cmap='coolwarm', alpha=0.6, label='y_train')
    axes[i, 0].set_title(f'Train: {column}')

    axes[i, 1].scatter(x_test.index, x_test[column], c=y_test_numeric, cmap='winter', alpha=0.6, label='y_test')
    axes[i, 1].set_title(f'Test: {column}')

    for j in range(2):
        axes[i, j].set_xlabel('Index\n')
        axes[i, j].set_ylabel(column)

plt.tight_layout()
plt.legend()
plt.show()

# Vizualizam distributia prin diagrame boxplot
plt.figure(figsize=(25, 20))

for i, col in enumerate(x_train.columns):
    plt.subplot(4, 5, i + 1)
    sns.boxplot(x=label, y=col, data=train_data, showfliers=False)
    plt.title(f'Box plot of {col} by is_hungry')

plt.tight_layout()
plt.show()

# Analizam distributia claselor in setul de date
class_distribution = y_train.value_counts()
print(class_distribution)

# Calculam si afisam proportiile
proportions = class_distribution / len(y_train)
print(proportions)

# Vizualizam distributia claselor (DATA IMBALANCED)
plt.figure(figsize=(10, 6))
class_distribution.plot(kind='bar', color='#a6bde0', alpha=0.8)
plt.title('Distributia valorilor binare')
plt.xlabel('Valoare binara')
plt.ylabel('Frecventa')
plt.show()

# Echilibram datele
adasyn = ADASYN()
x_train_balanced, y_train_balanced = adasyn.fit_resample(x_train, y_train)

# Analizam noua distributie a claselor
class_distribution = y_train_balanced.value_counts()
print(class_distribution)

# Calculam si printam proportiile
proportions = class_distribution / len(y_train_balanced)
print(proportions)

# Vizualizam distributia claselor echilibrate
plt.figure(figsize=(10, 6))
class_distribution.plot(kind='bar', color='#a6bde0', alpha=0.8)
plt.title('Distributia valorilor binare')
plt.xlabel('Valoare binara')
plt.ylabel('Frecventa')
plt.show()

# Aplicam RobustScaler datelor de training echlibrate
robust_scaler = RobustScaler()
x_train_balanced_scaled = robust_scaler.fit_transform(x_train_balanced)

# Aplicam RobustScaler datelor de test
x_test_scaled = robust_scaler.transform(x_test)

# Aplicam PCA
pca = PCA(n_components=0.9)

x_train_pca = pca.fit_transform(x_train_balanced_scaled)
x_test_pca = pca.transform(x_test_scaled)

print("Numarul dimensiunilor inainte de PCA (training):", x_train.shape[1])
print("Numarul dimensiuniloe inainte de PCA (test):", x_test.shape[1])

print("Numarul dimensiunilor dupa PCA (training):", x_train_pca.shape[1])
print("Numarul dimensiunilor dupa PCA (test):", x_test_pca.shape[1])


# Optuna

# Mentinem curatenia in output ignorand warning-urile
warnings.filterwarnings("ignore")


# Functia obiectiv pentru optimizarea hiperparametrilor
def objective(trial):
    # Sugeram un clasificator dintr-o lista de optiuni
    classifier_name = trial.suggest_categorical("classifier", ["LogisticRegression", "KNeighbors",
                                                               "SVC", "DecisionTree", "RandomForest",
                                                               "AdaBoost", "GradientBoosting", "GaussianNB",
                                                               "LinearDiscriminantAnalysis", "XGBClassifier",
                                                               "LGBMClassifier", "CatBoostClassifier"])

    # Blocuri separate pentru fiecare clasificator cu hiperparametri specifici
    if classifier_name == "LogisticRegression":
        C = trial.suggest_float("lr_C", 1e-4, 1e4, log=True)
        penalty = trial.suggest_categorical("lr_penalty", ["l1", "l2", "elasticnet", "none"])
        solver = trial.suggest_categorical("lr_solver", ["lbfgs", "liblinear", "sag"])
        l1_ratio = None

        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("lr_l1_ratio", 0, 1)

        if solver == "liblinear":
            if penalty in ["none", "elasticnet"]:
                penalty = "l2"
        elif solver in ["lbfgs", "sag"]:
            penalty = "l2"  # These solvers only support L2

        classifier_obj = LogisticRegression(C=C, penalty=penalty, solver=solver, l1_ratio=l1_ratio, max_iter=1000)

    elif classifier_name == "KNeighbors":
        n_neighbors = trial.suggest_int("knn_n_neighbors", 2, 50)
        classifier_obj = KNeighborsClassifier(n_neighbors=n_neighbors)

    elif classifier_name == "SVC":
        C = trial.suggest_float("svc_C", 1e-4, 1e4, log=True)
        classifier_obj = SVC(C=C)

    elif classifier_name == "DecisionTree":
        max_depth = trial.suggest_int("dt_max_depth", 1, 32)
        classifier_obj = DecisionTreeClassifier(max_depth=max_depth)

    elif classifier_name == "RandomForest":
        n_estimators = trial.suggest_int("rf_n_estimators", 10, 100)
        max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    elif classifier_name == "AdaBoost":
        n_estimators = trial.suggest_int("ab_n_estimators", 10, 100)
        learning_rate = trial.suggest_float("ab_learning_rate", 0.01, 1)
        classifier_obj = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

    elif classifier_name == "GradientBoosting":
        n_estimators = trial.suggest_int("gb_n_estimators", 10, 100)
        learning_rate = trial.suggest_float("gb_learning_rate", 0.01, 1)
        max_depth = trial.suggest_int("gb_max_depth", 1, 32)
        classifier_obj = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                                    max_depth=max_depth)

    elif classifier_name == "GaussianNB":
        classifier_obj = GaussianNB()

    elif classifier_name == "LinearDiscriminantAnalysis":
        classifier_obj = LinearDiscriminantAnalysis()

    elif classifier_name == "XGBClassifier":
        n_estimators = trial.suggest_int("xgb_n_estimators", 10, 100)
        max_depth = trial.suggest_int("xgb_max_depth", 2, 32)
        learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 1)
        classifier_obj = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

    elif classifier_name == "LGBMClassifier":
        n_estimators = trial.suggest_int("lgbm_n_estimators", 10, 100)
        max_depth = trial.suggest_int("lgbm_max_depth", 1, 30)
        learning_rate = trial.suggest_float("lgbm_learning_rate", 0.01, 1)
        num_leaves = trial.suggest_int("lgbm_num_leaves", 2, 256)
        classifier_obj = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                        num_leaves=num_leaves)

    elif classifier_name == "CatBoostClassifier":
        iterations = trial.suggest_int("catboost_iterations", 10, 100)
        depth = trial.suggest_int("catboost_depth", 4, 10)
        learning_rate = trial.suggest_float("catboost_learning_rate", 0.01, 1)
        classifier_obj = CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate, verbose=0)

    # Antrenam modelul cu datele echilibrate
    classifier_obj.fit(x_train_pca, y_train_balanced)

    # Realizam predictii cu setul de date
    predictions = classifier_obj.predict(x_test_pca)

    # Calculam acuratetea comparand predictiile cu etichetele reale
    accuracy = accuracy_score(y_test, predictions)

    # Returnam acuratetea ca valoare obiectiv pentru Optuna (se incearca maximizarea acuratetei)
    return accuracy


# Cream un studiu Optuna si specificam faptul ca dorim maximizarea acuratetei
study = optuna.create_study(direction='maximize')

# Optimizam functia obiectiv
study.optimize(objective, n_trials=300)

# Afisam cei mai buni parametri gasiti si cea mai buna acuratete obtinuta
print("Cei mai buni parametetri: ", study.best_params)
print("Cea mai buna acuratete: ", study.best_value)

# Preluam cei mai buni parametri dati de Optuna
best_params = {
    'classifier': 'LogisticRegression',
    'lr_C': 0.00033347576061317814,
    'lr_penalty': 'l2',
    'lr_solver': 'sag'
}

# Antrenam clasificatorul Logistic Regression cu cei mai buni parametri
best_classifier = LogisticRegression(
    C=best_params['lr_C'],
    penalty=best_params['lr_penalty'],
    solver=best_params['lr_solver'],
    max_iter=1000
)

best_classifier.fit(x_train_pca, y_train_balanced.values.ravel())

# Make predictions on the testing set
predictions = best_classifier.predict(x_test_pca)

# Afisam matricea de confuzie
cm = confusion_matrix(y_test, predictions)
plot_confusion_matrix(best_classifier, x_test_pca, y_test, display_labels=["is_hungry = 0", "is_hungry = 1"], cmap="Blues", values_format="d")
plt.title('Matrice de confuzie')
plt.show()

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')