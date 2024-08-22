import optuna
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.model_selection import cross_val_score
import time

optuna.logging.set_verbosity(optuna.logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)

def objective(trial, X, y):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p = trial.suggest_int('p', 1, 2)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    
    return cross_val_score(clf, X, y, n_jobs=-1, cv=5).mean()

def optimize(X, y, n_trials=100):
    start_time = time.time()

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    best_params = study.best_params
    best_accuracy = study.best_value
    
    end_time = time.time()

    return {'accuracy': best_accuracy,
            'best_params': best_params,
            'elapsed_time': round(end_time - start_time)
            }