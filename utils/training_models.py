import torch
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from torch.nn.parallel import DistributedDataParallel as DDP

class EarlyStoppingMax:
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): Número de épocas sem melhoria antes de parar.
            verbose (bool): Se True, imprime mensagens de log.
            delta (float): Mínima mudança considerada como melhoria.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_auc, model, path='model_checkpoints/best_model_checkpoint.pth'):
        if self.best_score is None:
            self.best_score = val_auc
            self.save_checkpoint(val_auc, model, path)
        elif val_auc < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_auc
            self.save_checkpoint(val_auc, model, path)
            self.counter = 0

    def save_checkpoint(self, val_auc, model, path):
        torch.save(model.module.state_dict(), path)
        print(f"Modelo salvo em: {path}")


def objective_ci_index(trial, tab_model_df, y, kf):
    # sugere um alpha (pode ajustar o intervalo/log)
    alpha = trial.suggest_float("alpha", 1e-3, 1.0)

    ci_scores = []
    for train_idx, test_idx in kf.split(tab_model_df, y["Cause of death"]):
        X_train, X_test = tab_model_df.iloc[train_idx], tab_model_df.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # structured arrays
        y_train_surv = np.array(
            [(e, t) for e, t in zip(y_train["Cause of death"], y_train["days_4years"])],
            dtype=[("Cause of death", bool), ("days_4years", float)]
        )
        y_test_surv = np.array(
            [(e, t) for e, t in zip(y_test["Cause of death"], y_test["days_4years"])],
            dtype=[("Cause of death", bool), ("days_4years", float)]
        )

        # treina com o alpha sugerido
        estimator = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=[alpha]).fit(X_train, y_train_surv)
        y_pred = estimator.predict(X_test)

        ci = concordance_index_censored(
            y_test["Cause of death"].astype(bool),
            y_test["days_4years"],
            y_pred
        )[0]
        ci_scores.append(ci)

    # retorna a média para o Optuna maximizar
    return np.mean(ci_scores)


def objective_ci_index_alpha_l1(trial, tab_model_df):
    # sugere um alpha (pode ajustar o intervalo/log)
    alpha = trial.suggest_float("alpha", 1e-3, 1.0)
    l1_ratio = trial.suggest_float("l1_ratio", 1e-3, 1.0)

    # treina com o alpha sugerido
    estimator = CoxPHFitter(l1_ratio=l1_ratio, penalizer=alpha).fit(tab_model_df, "days_4years", "Cause of death")

    # retorna a média para o Optuna maximizar
    return estimator.score(tab_model_df, scoring_method="concordance_index")


def find_best_cut_logrank(trial, tab_model_df, lower_bound, upper_bound, test_column):
    cut_point = trial.suggest_float("cut_point", lower_bound, upper_bound)
    A = tab_model_df[tab_model_df[test_column] >= cut_point]
    B = tab_model_df[tab_model_df[test_column] < cut_point]
    
    result = logrank_test(A["days_4years"], B["days_4years"], event_observed_A=A["Cause of death"], event_observed_B=B["Cause of death"])
    trial.set_user_attr("p_value", result.p_value)
    
    if A.shape[0] < 100:
        return 0
    elif B.shape[0] < 100:
        return 0
    else:
        return result.test_statistic
