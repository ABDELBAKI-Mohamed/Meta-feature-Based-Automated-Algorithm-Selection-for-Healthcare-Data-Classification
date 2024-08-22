import optuna
from sklearn.ensemble import GradientBoostingClassifier
import warnings
from sklearn.model_selection import cross_val_score
import time

optuna.logging.set_verbosity(optuna.logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)

def objective(trial, X, y):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_float('min_samples_split', 0.1, 1.0)
    min_samples_leaf = trial.suggest_float('min_samples_leaf', 0.1, 0.5)
    
    clf = GradientBoostingClassifier(n_estimators=n_estimators,
                                     learning_rate=learning_rate,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     random_state=42)
    
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