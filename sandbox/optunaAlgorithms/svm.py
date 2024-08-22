import optuna
from sklearn.svm import SVC
import warnings
from sklearn.model_selection import cross_val_score
import time

optuna.logging.set_verbosity(optuna.logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)

def objective(trial, X, y):
    
    C = trial.suggest_loguniform('C', 1e-4, 1e4)
    gamma = trial.suggest_loguniform('gamma', 1e-5, 1e1)
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    clf = SVC(C=C, gamma=gamma, kernel=kernel, random_state=42)
    
    return cross_val_score(clf, X, y, n_jobs=-1, cv=5).mean()

def optimize(X, y, n_trials=100):
    
    start_time = time.time()

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    best_params = study.best_params
    best_accuracy = study.best_value
    
    end_time = time.time()

    return {'accuracy':best_accuracy,
            'best_params':best_params,
            'elapsed_time': round(end_time - start_time)
            }