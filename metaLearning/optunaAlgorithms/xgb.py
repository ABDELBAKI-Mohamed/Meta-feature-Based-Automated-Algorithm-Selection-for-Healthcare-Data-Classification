import optuna
import xgboost as xgb
import warnings
from sklearn.model_selection import cross_val_score
import time

optuna.logging.set_verbosity(optuna.logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)

def objective_old(trial, X, y):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'eta': trial.suggest_loguniform('eta', 0.001, 0.1),
        'gamma': trial.suggest_loguniform('gamma', 0.001, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 0.9, 0.1),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 0.9, 0.1),
        'random_state': 42
    }
    
    dtrain = xgb.DMatrix(X, label=y)
    
    cv_result = xgb.cv(param, dtrain, num_boost_round=100, nfold=5, metrics='logloss', early_stopping_rounds=10, seed=42)
    
    return -cv_result['test-logloss-mean'].values[-1]

def objective_loss(trial, X, y):
    param = {
        'objective': 'multi:softmax',
        'num_class': len(set(y)),  # Number of unique classes in the labels
        'eval_metric': 'mlogloss',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'eta': trial.suggest_loguniform('eta', 0.001, 0.1),
        'gamma': trial.suggest_loguniform('gamma', 0.001, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 0.9, 0.1),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 0.9, 0.1),
        'random_state': 42
    }
    
    dtrain = xgb.DMatrix(X, label=y)
    
    cv_result = xgb.cv(param, dtrain, num_boost_round=100, nfold=5, metrics='mlogloss', early_stopping_rounds=10, seed=42)
    
    return cv_result['test-mlogloss-mean'].values[-1]

def objective_merror(trial, X, y):
    param = {
        'objective': 'multi:softmax',
        'num_class': len(set(y)),  # Number of unique classes in the labels
        'eval_metric': 'merror',    # Use 'error' for classification error (1 - accuracy)
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'eta': trial.suggest_loguniform('eta', 0.001, 0.1),
        'gamma': trial.suggest_loguniform('gamma', 0.001, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 0.9, 0.1),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 0.9, 0.1),
        'random_state': 42
    }
    
    dtrain = xgb.DMatrix(X, label=y)
    
    cv_result = xgb.cv(param, dtrain, num_boost_round=100, nfold=5, metrics='error', early_stopping_rounds=10, seed=42)
    
    return cv_result['test-merror-mean'].values[-1]

def objective(trial, X, y):
    param = {
        'objective': 'multi:softmax',
        'num_class': len(set(y)),  # Number of unique classes in the labels
        'eval_metric': 'mlogloss', # Use 'mlogloss' for multi-class logarithmic loss
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'eta': trial.suggest_loguniform('eta', 0.001, 0.1),
        'gamma': trial.suggest_loguniform('gamma', 0.001, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 0.9, 0.1),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 0.9, 0.1),
        'random_state': 42
    }
    
    dtrain = xgb.DMatrix(X, label=y)
    
    cv_result = xgb.cv(param, dtrain, num_boost_round=100, nfold=5, metrics='mlogloss', early_stopping_rounds=10, seed=42, stratified=True)
    
    return cv_result['test-mlogloss-mean'].values[-1]

def optimize(X, y, n_trials=100):
    start_time = time.time()

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    best_params = study.best_params
    
    end_time = time.time()

    return {'accuracy': study.best_value,
            'best_params': best_params,
            'elapsed_time': round(end_time - start_time)
            }