"""
Model training and hyperparameter optimization utilities.
"""
import optuna
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score


def create_optuna_objective(X_train, y_train, random_state=123):
    """
    Create an Optuna objective function for SGDClassifier hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random state for reproducibility
        
    Returns:
        objective: Optuna objective function
    """
    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        eta0 = trial.suggest_float('eta0', 1e-3, 0.5, log=True)
        learning_rate = trial.suggest_categorical(
            'learning_rate',
            ['constant', 'optimal', 'invscaling', 'adaptive']
        )
        
        model = SGDClassifier(
            loss='log_loss',
            alpha=alpha,
            eta0=eta0,
            learning_rate=learning_rate,
            max_iter=5000,
            random_state=random_state,
            tol=1e-3
        )
        
        score = cross_val_score(
            model, X_train, y_train,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1
        ).mean()
        
        return score
    
    return objective


def optimize_sgd_classifier(X_train, y_train, n_trials=100, random_state=123):
    """
    Optimize SGDClassifier hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_trials: Number of optimization trials
        random_state: Random state for reproducibility
        
    Returns:
        best_params: Dictionary of best hyperparameters
        study: Optuna study object
    """
    objective = create_optuna_objective(X_train, y_train, random_state)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Best trial:")
    print("F1_macro:", study.best_trial.value)
    print("Params:", study.best_trial.params)
    
    return study.best_trial.params, study


def train_final_model(X_train, y_train, best_params, random_state=123):
    """
    Train final SGDClassifier with optimized hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        best_params: Dictionary of best hyperparameters
        random_state: Random state for reproducibility
        
    Returns:
        model: Trained SGDClassifier
    """
    model = SGDClassifier(
        loss='log_loss',
        alpha=best_params['alpha'],
        eta0=best_params['eta0'],
        learning_rate=best_params['learning_rate'],
        max_iter=5000,
        random_state=random_state,
        tol=1e-3
    )
    
    model.fit(X_train, y_train)
    return model