#%%
# 0. Paquetes necesarios
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from scipy.stats import norm

#%%
# 1. Carga de la base de datos consolidada de jóvenes (2014 - 2026)
RUTA_ARCHIVO = r"C:\Users\jerso\OneDrive\Escritorio\Modelos\base_jovenes_2014_2026.csv"
base_total = pd.read_csv(RUTA_ARCHIVO)

#%%
# 2. Estimación del modelo logit agrupado ponderado (referencia: 4_media)
formula = (
    "ocupado ~ edad + I(edad**2) + mujer + jefe_hogar + en_pareja + estudia + "
    "C(educacion, Treatment(reference='4_Media')) + C(ciudad) + C(anio)"
)

modelo_logit = smf.glm(
    formula=formula,
    data=base_total,
    family=sm.families.Binomial(link=sm.families.links.Logit()),
    var_weights=base_total["peso"]
).fit(cov_type="HC1")

print("\n=== 1. RESULTADOS LOGIT AGRUPADO PONDERADO (2014 - 2026) ===")
print(modelo_logit.summary())

#%%
# 3. Efectos marginales promedio ponderados (ame) y odds ratios
X = modelo_logit.model.exog
beta = modelo_logit.params.values
cov_beta = modelo_logit.cov_params().values
w_norm = base_total["peso"].values / base_total["peso"].sum()

xb = X @ beta
p_xb = expit(xb)                  # Probabilidad logística acumulada Λ(x'β)
pdf_xb = p_xb * (1.0 - p_xb)      # Densidad logística λ(x'β) = Λ(x'β)[1 - Λ(x'β)]

# Efectos Marginales Promedio ponderados
ame = np.sum(w_norm[:, None] * pdf_xb[:, None] * beta[None, :], axis=0)

# Jacobiano de los AME para errores estándar por Método Delta
term1 = np.sum(w_norm * pdf_xb) * np.eye(len(beta))
factor = w_norm * pdf_xb * (1.0 - 2.0 * p_xb)
J = term1 + np.outer(beta, factor @ X)

se_ame = np.sqrt(np.diag(J @ cov_beta @ J.T))
z_ame = ame / se_ame
p_ame = 2 * (1 - norm.cdf(np.abs(z_ame)))

tabla_ame = pd.DataFrame({
    "AME (dy/dx)": ame,
    "Std. Err.": se_ame,
    "z": z_ame,
    "P>|z|": p_ame,
    "Odds Ratio": np.exp(beta)
}, index=modelo_logit.params.index).drop("Intercept")

print("\n=== 2. EFECTOS MARGINALES PROMEDIO (AME) Y ODDS RATIOS ===")
print(tabla_ame.round(4))

# SUPUESTOS DEL MODELO LOGIT
#%%
# 4. Prueba de especificación (Link Test de Pregibon)
base_total["xb_hat"] = xb  # Ya calculado en el bloque 3 como X @ beta
base_total["xb_hat_sq"] = base_total["xb_hat"] ** 2

link_test = smf.glm(
    formula="ocupado ~ xb_hat + xb_hat_sq",
    data=base_total,
    family=sm.families.Binomial(),
    freq_weights=base_total["peso"]
).fit(cov_type="HC1")

print("\n=== 3. LINK TEST (Linealidad en el Logit) ===")
print(link_test.summary().tables[1])

#%%
# 5. Verificación de separación perfecta / cuasi-perfecta
categoricas = ["mujer", "jefe_hogar", "en_pareja", "estudia", "educacion", "ciudad", "anio"]

print("\n=== 4. VERIFICACIÓN DE SEPARACIÓN PERFECTA (Tasa de ocupación por categoría) ===")
for col in categoricas:
    resumen = base_total.groupby(col)["ocupado"].agg(n="count", tasa_ocup="mean")
    extremos = resumen[(resumen["tasa_ocup"] < 0.02) | (resumen["tasa_ocup"] > 0.98) | (resumen["n"] < 30)]
    if not extremos.empty:
        print(f"\n[Alerta en {col}]:")
        print(extremos)
print("Verificación de celdas completada.")

#%%
# 6. Factor de Inflación de Varianza (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

exog_df = pd.DataFrame(modelo_logit.model.exog, columns=modelo_logit.model.exog_names)
# Excluimos Intercept y la cuadrática (que por construcción tiene VIF alto con edad)
cols_vif = [c for c in exog_df.columns if c not in ["Intercept", "I(edad ** 2)"]]
X_vif = sm.add_constant(exog_df[cols_vif])

vif_data = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
}).query("Variable != 'const'")

print("\n=== 5. MULTICOLINEALIDAD (Top 10 VIF más altos) ===")
print(vif_data.sort_values("VIF", ascending=False).head(10).round(2))

#%%
# 7. Errores estándar agrupados (ejemplo por ciudad-año)
grupos = base_total["ciudad"].astype(str) + "_" + base_total["anio"].astype(str)

modelo_cluster = smf.glm(
    formula=formula,
    data=base_total,
    family=sm.families.Binomial(),
    freq_weights=base_total["peso"]
).fit(cov_type="cluster", cov_kwds={"groups": grupos})

comparacion_se = pd.DataFrame({
    "Beta": modelo_logit.params,
    "SE_HC1": modelo_logit.bse,
    "SE_Cluster": modelo_cluster.bse,
    "Ratio_SE": modelo_cluster.bse / modelo_logit.bse
})
print("\n=== 6. COMPARACIÓN ERRORES ESTÁNDAR (HC1 vs Cluster Ciudad-Año) ===")
print(comparacion_se.head(10).round(4))

#%%
# 8. Curva de calibración por deciles (ponderada) y Brier Score
base_total["prob_pred"] = p_xb
base_total["decil"] = pd.qcut(base_total["prob_pred"], q=10, labels=False) + 1

calibracion = base_total.groupby("decil").apply(
    lambda g: pd.Series({
        "Observado_Pond": np.average(g["ocupado"], weights=g["peso"]),
        "Predicho_Pond": np.average(g["prob_pred"], weights=g["peso"]),
        "Diferencia": np.average(g["ocupado"] - g["prob_pred"], weights=g["peso"])
    }),
    include_groups=False
)

brier_score = np.average((base_total["ocupado"] - base_total["prob_pred"]) ** 2, weights=base_total["peso"])

print(f"\n=== 7. CALIBRACIÓN POR DECILES (Brier Score ponderado: {brier_score:.4f}) ===")
print(calibracion.round(4))