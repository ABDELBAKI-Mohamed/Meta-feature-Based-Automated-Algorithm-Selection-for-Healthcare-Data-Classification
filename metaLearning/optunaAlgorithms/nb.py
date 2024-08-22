from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning)

def optimize(X, y, trials=1):
    start_time = time.time()

    clf = GaussianNB()
    
    end_time = time.time()

    return {
        'accuracy':cross_val_score(clf, X, y, cv=5).mean(), 
        'best_params':'does not use hyperparameters',
        'elapsed_time': round(end_time - start_time)
        }
