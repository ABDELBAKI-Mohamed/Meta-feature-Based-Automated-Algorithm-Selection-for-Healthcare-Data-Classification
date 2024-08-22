import optuna
from sklearn.ensemble import AdaBoostClassifier
import warnings
from sklearn.model_selection import cross_val_score
import time

optuna.logging.set_verbosity(optuna.logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)

def objective(trial, X, y):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 1.0)
    
    clf = AdaBoostClassifier(n_estimators=n_estimators,
                             learning_rate=learning_rate,
                             random_state=42)
    
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