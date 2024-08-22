import optuna
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.model_selection import cross_val_score
import time

optuna.logging.set_verbosity(optuna.logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)

def objective(trial, X, y):
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    C = trial.suggest_loguniform('C', 1e-4, 1e4)
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    
    clf = LogisticRegression(solver='saga', penalty=penalty, C=C, max_iter=max_iter, random_state=42)
    
    return cross_val_score(clf, X, y, n_jobs=-1, cv=5, scoring='accuracy').mean()

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