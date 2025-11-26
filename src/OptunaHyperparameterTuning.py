import optuna
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier

def optuna_hyperparameter_search(X_train, y_train, random_state, n_trials=100):
    """
    Runs an Optuna hyperparameter search to optimize an SGDClassifier
    using F1-macro as the evaluation metric.

    Parameters
    ----------
    X_train : array-like
        Training input features.
    y_train : array-like
        Training labels.
    random_state : int
        Random seed for reproducibility.
    n_trials : int, optional
        Number of Optuna trials to run (default: 100).

    Returns
    -------
    dict
        The best set of hyperparameters found by Optuna.
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
            model,
            X_train,
            y_train,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1
        ).mean()

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Best F1_macro:", study.best_trial.value)
    print("Best Params:", study.best_trial.params)

    return study.best_trial.params