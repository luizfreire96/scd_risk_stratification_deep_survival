from sklearn.preprocessing import StandardScaler
from sksurv.nonparametric import kaplan_meier_estimator
from scipy.stats import chi2

from lifelines import CoxPHFitter

from sklearn.metrics import roc_auc_score

from sksurv.compare import compare_survival
from sksurv.util import Surv

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

colors = ['#7c9dd2', '#156082']
groups = ['Low Risk', 'High Risk']
linestyles = {'Low Risk': 'dashed', 'High Risk': 'solid'}

data_path = os.path.join("raw_data/Holter_ECG/")
results_path = "treated_data/ts_features/"

tab_data = pd.read_parquet("test_tab.parquet")

tab_data.loc[tab_data['Cause of death'] != 3, ['Cause of death']] = 0
tab_data.loc[tab_data['Cause of death'] == 3, ['Cause of death']] = 1

index_df = pd.read_csv("z_dl_hazard.csv").drop(
    columns=["days_4years", "Cause of death"])

tab_data = tab_data.merge(index_df, on="Patient ID")
univariate_aux = tab_data.copy()

with open("variaveis_artigo.json", "r") as f:
    data = json.load(f)

vars_sem_laudo = data["vars_artigo"]
ai_index_columns = ["dl_hazard"]

label_columns = ['Patient ID', "Cause of death", "days_4years"]
tab_data = tab_data[label_columns + vars_sem_laudo + ai_index_columns]
tab_data["NYHA class"] = tab_data["NYHA class"] == 3
tab_data["dl_hazard"] = tab_data["dl_hazard"] > tab_data["dl_hazard"].quantile(
    0.75).astype(int)

categorical_columns = [
    'HF etiology - Diagnosis']

# agregando classes abaixo de 10 eventos em outros
tab_data["HF etiology - Diagnosis"] = np.where(
    tab_data["HF etiology - Diagnosis"].isin([5, 4, 7, 9]), 9, tab_data["HF etiology - Diagnosis"])

onehot = pd.get_dummies(tab_data,
                        columns=categorical_columns,
                        prefix=categorical_columns,
                        dtype=float,
                        drop_first=True).set_index("Patient ID")

onehot = onehot.drop(columns="HF etiology - Diagnosis_9")

scaler = StandardScaler()

cols_to_normalize = ["Age"]
onehot[cols_to_normalize] = scaler.fit_transform(onehot[cols_to_normalize])

############
# using AI #
############

estimator_ai = CoxPHFitter(l1_ratio=0, penalizer=0).fit(
    onehot, "days_4years", "Cause of death")

ci_ai = estimator_ai.score(onehot, scoring_method="concordance_index")

coef_df_ai = pd.DataFrame({
    "Feature": onehot.columns[2:],
    "Hazard Ratio": estimator_ai.hazard_ratios_,
    "Lower Bound": estimator_ai.confidence_intervals_["95% lower-bound"].apply(np.exp),
    "Upper Bound": estimator_ai.confidence_intervals_["95% upper-bound"].apply(np.exp),
    "p-value":  estimator_ai.summary["p"]

})

coef_df_ai["err_minus"] = coef_df_ai["Hazard Ratio"] - \
    coef_df_ai["Lower Bound"]
coef_df_ai["err_plus"] = coef_df_ai["Upper Bound"] - coef_df_ai["Hazard Ratio"]

y_pred_ai = estimator_ai.predict_partial_hazard(onehot)

pred_ai_df = pd.DataFrame({"Patient ID": onehot.index,
                           "risk prediction": y_pred_ai})

#################################
# using AI + ECG variables only #
#################################

ecg_df = onehot[["Cause of death", "days_4years", "Q-waves (necrosis, yes=1)",
                 "QRS > 120 ms ", "Non-sustained ventricular tachycardia (CH>10)"]]
estimator_ecg = CoxPHFitter(l1_ratio=0, penalizer=0).fit(
    ecg_df, "days_4years", "Cause of death")
ci_ecg = estimator_ecg.score(ecg_df, scoring_method="concordance_index")

coef_df_ecg = pd.DataFrame({
    "Feature": ecg_df.columns[2:],
    "Hazard Ratio": estimator_ecg.hazard_ratios_,
    "Lower Bound": estimator_ecg.confidence_intervals_["95% lower-bound"].apply(np.exp),
    "Upper Bound": estimator_ecg.confidence_intervals_["95% upper-bound"].apply(np.exp),
    "p-value":  estimator_ecg.summary["p"]

})

coef_df_ecg["err_minus"] = coef_df_ecg["Hazard Ratio"] - \
    coef_df_ecg["Lower Bound"]
coef_df_ecg["err_plus"] = coef_df_ecg["Upper Bound"] - \
    coef_df_ecg["Hazard Ratio"]

y_pred_ecg = estimator_ecg.predict_partial_hazard(ecg_df)

pred_ecg_df = pd.DataFrame({"Patient ID": ecg_df.index,
                            "risk prediction": y_pred_ecg})

####################
# without using AI #
####################
estimator_no_ai = CoxPHFitter(l1_ratio=0, penalizer=0).fit(
    onehot.drop(columns="dl_hazard"), "days_4years", "Cause of death")

ci_no_ai = estimator_no_ai.score(onehot.drop(
    columns="dl_hazard"), scoring_method="concordance_index")

coef_df_no_ai = pd.DataFrame({
    "Feature": onehot.drop(columns="dl_hazard").columns[2:],
    "Hazard Ratio": estimator_no_ai.hazard_ratios_,
    "Lower Bound": estimator_no_ai.confidence_intervals_["95% lower-bound"].apply(np.exp),
    "Upper Bound": estimator_no_ai.confidence_intervals_["95% upper-bound"].apply(np.exp),
    "p-value":  estimator_no_ai.summary["p"]

})
coef_df_no_ai["err_minus"] = coef_df_no_ai["Hazard Ratio"] - \
    coef_df_no_ai["Lower Bound"]
coef_df_no_ai["err_plus"] = coef_df_no_ai["Upper Bound"] - \
    coef_df_no_ai["Hazard Ratio"]


y_pred_no_ai = estimator_no_ai.predict_partial_hazard(
    onehot.drop(columns="dl_hazard"))

no_ai_pred_df = pd.DataFrame({"Patient ID": onehot.index,
                              "risk prediction": y_pred_no_ai})

coef_df_ai["Source"] = "DLHS + ECG + Clinical"
coef_df_no_ai["Source"] = "ECG + Clinical"

df_all = pd.concat([coef_df_ai, coef_df_no_ai], ignore_index=True)

print("C-index médio (com AI):", ci_ai)
print("C-index médio (sem AI):", ci_no_ai)
print("C-index médio (variaveis ECG):", ci_ecg)

#####################
# plotando gráficos #
#####################

######################
# analise univariada #

# grarfico de barras
univariate_df = univariate_aux[["dl_hazard", "days_4years", "Cause of death"]]
p_75 = univariate_df["dl_hazard"].quantile(0.75)
univariate_df['cat_dl_hazard'] = pd.cut(
    univariate_df["dl_hazard"],
    bins=[float('-inf'), p_75, float('inf')],
    labels=['Low Risk', 'High Risk'])

bar_plot_df = univariate_df.groupby("cat_dl_hazard")[
    "Cause of death"].mean().reset_index()

# --- gráfico em matplotlib ---
plt.figure(figsize=(8, 8))
bars = plt.bar(
    bar_plot_df["cat_dl_hazard"],
    bar_plot_df["Cause of death"],
    color=["#1f77b4", "#ff7f0e"],
    width=0.6
)

# adiciona rótulos acima das barras
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height:.3f}",
        ha='center', va='bottom',
        fontsize=14
    )

# títulos e rótulos dos eixos
plt.xlabel("Risk Group", fontsize=16)
plt.ylabel("SCD rate in 4 years", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# salvar em PDF
plt.savefig("results/univariate_bar_plot_scd_rate.pdf",
            format="pdf", bbox_inches="tight")
plt.close()

# Kaplan meier + CI index
estimator_univariate = CoxPHFitter(l1_ratio=0, penalizer=0).fit(
    univariate_df.drop(columns="cat_dl_hazard"), "days_4years", "Cause of death")
ci_univariate = estimator_univariate.score(univariate_df.drop(
    columns="cat_dl_hazard"), scoring_method="concordance_index")

structured_array = Surv.from_dataframe(
    "Cause of death", "days_4years", univariate_df)
log_rank, p_value = compare_survival(
    structured_array, univariate_df["cat_dl_hazard"], return_stats=False)

# --- gráfico ---
plt.figure(figsize=(8, 8))
for g, color in zip(groups, colors):
    aux_y = univariate_df[univariate_df["cat_dl_hazard"] == g]
    time, survival_prob = kaplan_meier_estimator(
        aux_y["Cause of death"].astype(bool), aux_y["days_4years"]
    )
    plt.step(
        time, survival_prob,
        where="post",
        label=f"KM real - {g}",
        color=color,
        linestyle=linestyles[g],
        linewidth=2.5
    )

# --- layout e estilo ---
plt.xlabel("Time (days)", fontsize=16)
plt.ylabel("Survival Rate", fontsize=16)
plt.xlim(0, 1460)
plt.ylim(0, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=14)
plt.title(f"p-value = {p_value:.3f} | Logrank = {log_rank:.3f} | CI = {ci_univariate:.3f}",
          fontsize=16, pad=20)
plt.tight_layout()

# --- salvar em PDF ---
plt.savefig("results/univariate_kaplan_meier.pdf",
            format="pdf", bbox_inches="tight")
plt.close()


######
# AI #
evaluation_y = onehot[["days_4years", "Cause of death"]]
evaluation_df = pred_ai_df.merge(evaluation_y, on="Patient ID")

evaluation_df["risk"] = pd.cut(evaluation_df["risk prediction"],
                               bins=[-float("inf"), evaluation_df["risk prediction"].quantile(
                                   0.75), float("inf")],
                               labels=['Low Risk', 'High Risk'])

structured_array = Surv.from_dataframe(
    "Cause of death", "days_4years", evaluation_df)
log_rank, p_value = compare_survival(
    structured_array, evaluation_df["risk"], return_stats=False)

# --- gráfico Kaplan-Meier ---
plt.figure(figsize=(8, 8))
for g, color in zip(groups, colors):
    aux_y = evaluation_df[evaluation_df["risk"] == g]
    time, survival_prob = kaplan_meier_estimator(
        aux_y["Cause of death"].astype(bool),
        aux_y["days_4years"]
    )
    plt.step(
        time, survival_prob,
        where="post",
        label=f"KM real - {g}",
        color=color,
        linestyle=linestyles.get(g, 'solid'),
        linewidth=2.5
    )

# --- layout e formatação ---
plt.xlabel("Time (days)", fontsize=16)
plt.ylabel("Survival Rate", fontsize=16)
plt.xlim(0, 1460)
plt.ylim(0, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=14, frameon=False)
plt.title(f"p-value = {p_value:.3f} | Logrank = {log_rank:.3f} | CI = {ci_ai:.3f}",
          fontsize=16, pad=20)
plt.tight_layout()

# --- salvar em PDF ---
plt.savefig("results/dl_hazard_kaplan_meier.pdf",
            format="pdf", bbox_inches="tight")
plt.close()


##############################
# AI + time series variables #
evaluation_y = ecg_df[["days_4years", "Cause of death"]]
evaluation_df = pred_ecg_df.merge(evaluation_y, on="Patient ID")

evaluation_df["risk"] = pd.cut(evaluation_df["risk prediction"],
                               bins=[-float("inf"), evaluation_df["risk prediction"].quantile(
                                   0.75), float("inf")],
                               labels=['Low Risk', 'High Risk'])

structured_array = Surv.from_dataframe(
    "Cause of death", "days_4years", evaluation_df)
log_rank, p_value = compare_survival(
    structured_array, evaluation_df["risk"], return_stats=False)

# --- gráfico Kaplan-Meier ---
plt.figure(figsize=(8, 8))
for g, color in zip(groups, colors):
    aux_y = evaluation_df[evaluation_df["risk"] == g]
    time, survival_prob = kaplan_meier_estimator(
        aux_y["Cause of death"].astype(bool),
        aux_y["days_4years"]
    )
    plt.step(
        time, survival_prob,
        where="post",
        label=f"KM real - {g}",
        color=color,
        linestyle=linestyles.get(g, 'solid'),
        linewidth=2.5
    )

# --- layout e formatação ---
plt.xlabel("Time (days)", fontsize=16)
plt.ylabel("Survival Rate", fontsize=16)
plt.xlim(0, 1460)
plt.ylim(0, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=14, frameon=False)
plt.title(f"p-value = {p_value:.3f} | Logrank = {log_rank:.3f} | CI = {ci_ecg:.3f}",
          fontsize=16, pad=20)
plt.tight_layout()

# --- salvar em PDF ---
plt.savefig("results/ecg_kaplan_meier.pdf", format="pdf", bbox_inches="tight")
plt.close()


# no AI
evaluation_y = onehot[["days_4years", "Cause of death"]]
evaluation_df = no_ai_pred_df.merge(evaluation_y, on="Patient ID")

evaluation_df["risk"] = pd.cut(evaluation_df["risk prediction"],
                               bins=[-float("inf"), evaluation_df["risk prediction"].quantile(
                                   0.75), float("inf")],
                               labels=['Low Risk', 'High Risk'])

structured_array = Surv.from_dataframe(
    "Cause of death", "days_4years", evaluation_df)
log_rank, p_value = compare_survival(
    structured_array, evaluation_df["risk"], return_stats=False)

# --- gráfico Kaplan-Meier ---
plt.figure(figsize=(8, 8))
for g, color in zip(evaluation_df["risk"].unique(), colors):
    aux_y = evaluation_df[evaluation_df["risk"] == g]
    time, survival_prob = kaplan_meier_estimator(
        aux_y["Cause of death"].astype(bool),
        aux_y["days_4years"]
    )
    plt.step(
        time, survival_prob,
        where="post",
        label=f"KM real - {g}",
        color=color,
        linestyle=linestyles.get(g, 'solid'),
        linewidth=2.5
    )

# --- formatação ---
plt.xlabel("Time (days)", fontsize=16)
plt.ylabel("Survival Rate", fontsize=16)
plt.xlim(0, 1460)
plt.ylim(0, 1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=14, frameon=False)
plt.title(f"p-value = {p_value:.3f} | Logrank = {log_rank:.3f} | CI = {ci_no_ai:.3f}",
          fontsize=16, pad=20)
plt.tight_layout()

# --- salvar em PDF ---
plt.savefig("results/tabular_kaplan_meier.pdf",
            format="pdf", bbox_inches="tight")
plt.close()


###################
# cox analysiseis #
###################

#################
# hazard ratios #

mapeamento = {
    'Amiodarone (yes=1)': 'Amiodarone',
    'ACE inhibitor (yes=1)': 'ACE inhibitor',
    'HF etiology - Diagnosis_2': 'Ischemic Cardiomyopathy',
    'Diabetes (yes=1)': 'Diabetes',
    'HF etiology - Diagnosis_3': 'Enolic dilated cardiomyopathy',
    'HF etiology - Diagnosis_8': 'Hypertensive cardiomyopathy',
    'NYHA class': 'NYHA class III',
    'dl_hazard': 'DLHS',
    'Q-waves (necrosis, yes=1)': 'Q-waves necrosis'}

df_all["Feature"] = df_all["Feature"].map(mapeamento).fillna(df_all["Feature"])

# 1) Orderna só o modelo referência
ref = df_all[df_all["Source"] == "DLHS + ECG + Clinical"]
ordem_features = ref.sort_values("Hazard Ratio")["Feature"].tolist()

# 2) Reordena o dataframe inteiro usando essa ordem
df_all["Feature"] = pd.Categorical(
    df_all["Feature"], categories=ordem_features, ordered=True)
df_all = df_all.sort_values("Feature")

features = df_all["Feature"].unique()
sources = df_all["Source"].unique()

bar_width = 0.35
y_pos = np.arange(len(features))

bar_width = 0.28
alpha = 0.9

all_vars = df_all["Feature"].unique().tolist()

# Eixos / posições
y = np.arange(len(all_vars))
sources = list(df_all["Source"].unique())

# centralizar grupos de barras: cria offsets centrados em 0
n_sources = len(sources)
offsets = (np.arange(n_sources) - (n_sources - 1) / 2.0) * bar_width
plt.figure(figsize=(10, max(4, len(all_vars) * 0.35)))
highlight = "DLHS"

for i, source in enumerate(sources):
    subset = df_all[df_all["Source"] == source].set_index("Feature")
    subset = subset.reindex(all_vars)

    x = subset["Hazard Ratio"].values.astype(float)
    err_minus = subset["err_minus"].fillna(0).values.astype(float)
    err_plus = subset["err_plus"].fillna(0).values.astype(float)
    pos = y + offsets[i]

    mask = ~np.isnan(x)
    if not mask.any():
        continue

    base_color = colors[i % len(colors)]

    # Plotar barras primeiro
    bars = plt.barh(
        pos[mask],
        x[mask],
        height=bar_width * 0.9,
        color=base_color,
        edgecolor='black',
        linewidth=0.7,
        alpha=1.0,
        label=source,
        zorder=1  # Barras com zorder baixo
    )

    # Agora plotar barras de erro separadamente
    feats = np.array(all_vars)[mask]
    for j, (feat, x_val, err_min, err_plus_val, y_pos) in enumerate(zip(feats, x[mask], err_minus[mask], err_plus[mask], pos[mask])):
        if feat == highlight:
            bar_alpha = 1.0
            error_alpha = 1.0
            error_color = 'black'  # Cor contrastante
            error_width = 1.5
        else:
            bar_alpha = 0.7
            error_alpha = 0.7
            error_color = base_color
            error_width = 1.0

        # Ajustar opacidade da barra
        bars[j].set_alpha(bar_alpha)

        # Desenhar barra de erro customizada com zorder ALTO
        plt.errorbar(
            x=x_val,
            y=y_pos,
            xerr=[[err_min], [err_plus_val]],
            fmt='none',
            ecolor=error_color,
            elinewidth=error_width,
            capsize=3,
            alpha=error_alpha,
            zorder=3  # zorder ALTO para ficar na frente
        )


# ajustes de aparência
plt.yticks(y, all_vars)
plt.axvline(1.0, color='gray', linewidth=0.3,
            linestyle='--')  # referência HR = 1
plt.xlabel("Hazard Ratio")
plt.legend(frameon=True)
plt.gca().invert_yaxis()  # normalmente mais legível com primeiro item em cima
plt.tight_layout()

# --- Linha vertical x=1 ---
plt.axvline(x=1, color='red', linestyle='--', linewidth=2)

# --- Eixos e estilo ---
plt.yticks(np.arange(len(all_vars)), all_vars, fontsize=10)
plt.xlabel("Hazard Ratio", fontsize=10)
plt.ylabel("Variable", fontsize=10)
plt.xticks(fontsize=10)
plt.legend(fontsize=15, frameon=False)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()


# --- Salvar em PDF ---
plt.savefig("results/survival_hazards.pdf", format="pdf", bbox_inches="tight")
plt.close()

# LRT
l_ai = estimator_ai.log_likelihood_
l_no_ai = estimator_no_ai.log_likelihood_

lr_stat = 2 * (l_ai - l_no_ai)

p_value = chi2.sf(lr_stat, df=1)
print("p-value:", p_value)

################
# AUC DINAMICO #
################
auc_df = pred_ai_df.merge(tab_data, on="Patient ID")
censoring_days = [182, 365, 547, 730, 912, 1095, 1277, 1460]
years = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
aucs = []

for censoring_day in censoring_days:
    event = (auc_df["Cause of death"] == 1) & (
        auc_df["days_4years"] < censoring_day)
    score = auc_df["risk prediction"]
    auc = roc_auc_score(event, score)
    aucs.append(auc)


plt.plot(years, aucs, marker="o")  # marker é opcional
plt.xlabel("Censoring Years")
plt.ylabel("AUC")
plt.grid(True)

plt.xticks(fontsize=15)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.ylim(0.5, 1)  # ← ADICIONE ESTA LINHA


plt.savefig("results/dynamic_auc.pdf", format="pdf", bbox_inches="tight")
plt.close()

###########################
# AUC DINAMICO univariado #
###########################
auc_df = pred_ai_df.merge(tab_data, on="Patient ID")
censoring_days = [182, 365, 547, 730, 912, 1095, 1277, 1460]
years = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
aucs = []

for censoring_day in censoring_days:
    event = (auc_df["Cause of death"] == 1) & (
        auc_df["days_4years"] < censoring_day)
    score = univariate_aux["dl_hazard"]
    auc = roc_auc_score(event, score)
    aucs.append(auc)


plt.plot(years, aucs, marker="o")  # marker é opcional
plt.xlabel("Censoring Years")
plt.ylabel("AUC")
plt.grid(True)

plt.xticks(fontsize=15)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.ylim(0.5, 1)  # ← ADICIONE ESTA LINHA


plt.savefig("results/univariate_dynamic_auc.pdf",
            format="pdf", bbox_inches="tight")
plt.close()
