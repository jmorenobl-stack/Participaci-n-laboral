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