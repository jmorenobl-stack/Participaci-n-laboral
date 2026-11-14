#%%
# 0. Paquetes necesarios
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

#%%
# 1. Carga de la base de datos consolidada de jóvenes (2014 - 2026)
RUTA_ARCHIVO = r"C:\Users\jerso\OneDrive\Escritorio\Modelos\base_jovenes_2014_2026.csv"
base_total = pd.read_csv(RUTA_ARCHIVO)

#%%
# 2. Estimación del modelo probit agrupado ponderado (referencia: 4_media)
formula = (
    "ocupado ~ edad + I(edad**2) + mujer + jefe_hogar + en_pareja + estudia + "
    "C(educacion, Treatment(reference='4_Media')) + C(ciudad) + C(anio)"
)

modelo_pooled = smf.glm(
    formula=formula,
    data=base_total,
    family=sm.families.Binomial(link=sm.families.links.Probit()),
    var_weights=base_total["peso"]
).fit(cov_type="HC1")

print("\n=== 1. RESULTADOS PROBIT AGRUPADO PONDERADO (2014 - 2026) ===")
print(modelo_pooled.summary())

#%%
# 3. Efectos marginales promedio ponderados (ame) y errores estándar (método delta)
X = modelo_pooled.model.exog
beta = modelo_pooled.params.values
cov_beta = modelo_pooled.cov_params().values
w_norm = base_total["peso"].values / base_total["peso"].sum()

xb = X @ beta
pdf_xb = norm.pdf(xb)

ame = np.sum(w_norm[:, None] * pdf_xb[:, None] * beta[None, :], axis=0)
term1 = np.sum(w_norm * pdf_xb) * np.eye(len(beta))
factor = -w_norm * xb * pdf_xb
J = term1 + np.outer(beta, factor @ X)

se_ame = np.sqrt(np.diag(J @ cov_beta @ J.T))
z_ame = ame / se_ame
p_ame = 2 * (1 - norm.cdf(np.abs(z_ame)))

tabla_ame = pd.DataFrame({
    "AME (dy/dx)": ame,
    "Std. Err.": se_ame,
    "z": z_ame,
    "P>|z|": p_ame
}, index=modelo_pooled.params.index).drop("Intercept")

print("\n=== 2. EFECTOS MARGINALES PROMEDIO PONDERADOS (AME) ===")
print(tabla_ame.round(4))

#%%
# 4. DIAGNÓSTICO DE SUPUESTOS DEL MODELO PROBIT
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2

# Alinear exactamente la muestra utilizada por el modelo (evita errores por NaN)
idx_valido = modelo_pooled.model.data.row_labels
df_est = base_total.loc[idx_valido].copy()
y = modelo_pooled.model.endog
X = modelo_pooled.model.exog
nombres_vars = modelo_pooled.model.exog_names
w = df_est["peso"].values
w_norm = w / w.sum()
xb = X @ modelo_pooled.params.values

print("=== DIAGNÓSTICO 1: CONVERGENCIA Y SEPARACIÓN PERFECTA ===")
print(f"¿Convergió el algoritmo MLE?: {modelo_pooled.converged}")
print(f"Observaciones válidas: {len(y):,} (descartadas por NaN: {len(base_total) - len(y):,})")

# Revisar coeficientes o errores estándar explosivos (señal de separación cuasi-perfecta)
resumen_extremos = pd.DataFrame({
    "beta": modelo_pooled.params,
    "std_err": modelo_pooled.bse
})
sospechosos_sep = resumen_extremos[
    (resumen_extremos["beta"].abs() > 8) | (resumen_extremos["std_err"] > 5)
]
print("Variables con posible separación perfecta/cuasi-perfecta:")
print(sospechosos_sep if not sospechosos_sep.empty else "Ninguna detectada.")

# Revisar tasas de ocupación por categoría para detectar celdas con 0% o 100%
for col in ["educacion", "ciudad", "anio", "estudia", "mujer"]:
    tasas = df_est.groupby(col)["ocupado"].agg(["count", "mean"])
    extremos_cat = tasas[(tasas["mean"] == 0) | (tasas["mean"] == 1)]
    if not extremos_cat.empty:
        print(f"¡Alerta! Separación exacta en '{col}':\n{extremos_cat}")

#%%
print("\n=== DIAGNÓSTICO 2: MULTICOLINEALIDAD (VIF EN REGRESORES PRINCIPALES) ===")
# Calculamos VIF sobre variables continuas y dummies principales (excluyendo Intercept y FE masivos)
cols_vif = [
    i for i, name in enumerate(nombres_vars)
    if name != "Intercept" and not name.startswith("C(ciudad)") and not name.startswith("C(anio)")
]
vif_df = pd.DataFrame({
    "Variable": [nombres_vars[i] for i in cols_vif],
    "VIF": [variance_inflation_factor(X[:, cols_vif], idx) for idx in range(len(cols_vif))]
})
print(vif_df.round(2))
print("Nota: VIF alto entre 'edad' e 'I(edad ** 2)' es normal; vigila valores > 10 en otras variables.")

#%%
print("\n=== DIAGNÓSTICO 3: ESPECIFICACIÓN DEL ÍNDICE Y ENLACE NORMAL (LINK TEST / RESET) ===")
# Pregibon Link Test & RESET: si xb^2 o xb^3 son significativos, falla la forma funcional o normalidad
df_link = pd.DataFrame({
    "y": y,
    "xb": xb,
    "xb2": xb**2,
    "xb3": xb**3,
    "peso": w
})
link_test = smf.glm(
    "y ~ xb + xb2 + xb3",
    data=df_link,
    family=sm.families.Binomial(link=sm.families.links.Probit()),
    var_weights=df_link["peso"]
).fit(cov_type="HC1")

wald_reset = link_test.wald_test("xb2 = 0, xb3 = 0", scalar=True)
print(f"Test conjunto RESET (H0: forma funcional y enlace normal correctos):")
print(f"Estadístico Wald chi2(2) = {wald_reset.statistic:.4f} | p-valor = {wald_reset.pvalue:.4f}")
if wald_reset.pvalue < 0.05:
    print("-> Advertencia: Evidencia de mala especificación funcional o desviación de normalidad.")
else:
    print("-> OK: No se rechaza la especificación del enlace probit.")

#%%
print("\n=== DIAGNÓSTICO 4: HETEROCEDASTICIDAD DEL ERROR LATENTE (TEST LM) ===")
# Contrasta H0: Var(e | X) = 1 frente a sigma_i = exp(Z_i' gamma)
Phi_xb = np.clip(norm.cdf(xb), 1e-12, 1 - 1e-12)
phi_xb = norm.pdf(xb)
residuo_gen = (y - Phi_xb) * phi_xb / (Phi_xb * (1 - Phi_xb))

# Variables sospechosas de generar heterocedasticidad en la varianza latente
vars_het = ["edad", "mujer", "estudia", "jefe_hogar"]
Z_het = df_est[vars_het].values
scores_beta = residuo_gen[:, None] * X
scores_gamma = -(residuo_gen * xb)[:, None] * Z_het
S = np.hstack([scores_beta, scores_gamma])

# Regresión auxiliar ponderada de vector de unos sobre los scores (OPG LM test)
sw = np.sqrt(w_norm * len(y))
S_w = S * sw[:, None]
ones_w = sw
beta_aux, _, _, _ = np.linalg.lstsq(S_w, ones_w, rcond=None)
ssr = np.sum((ones_w - S_w @ beta_aux) ** 2)
lm_stat = len(y) - ssr
p_val_het = 1 - chi2.cdf(lm_stat, df=Z_het.shape[1])

print(f"Test LM de heterocedasticidad latente sobre {vars_het}:")
print(f"LM chi2({Z_het.shape[1]}) = {lm_stat:.4f} | p-valor = {p_val_het:.4f}")
if p_val_het < 0.05:
    print("-> Advertencia: Se rechaza homocedasticidad latente; los coeficientes beta pueden estar sesgados.")
else:
    print("-> OK: No se detecta heterocedasticidad latente significativa en las variables evaluadas.")

#%%
print("\n=== DIAGNÓSTICO 5: AME CORREGIDO PARA EDAD (EFECTO TOTAL) Y MUESTRA ALINEADA ===")
b_edad = modelo_pooled.params["edad"]
b_edad2 = modelo_pooled.params["I(edad ** 2)"]
edad_vec = df_est["edad"].values

# Efecto marginal total de la edad: phi(xb) * (b_edad + 2 * b_edad2 * edad)
ame_edad_total = np.sum(w_norm * phi_xb * (b_edad + 2 * b_edad2 * edad_vec))
print(f"AME total real de 'edad' (combinando término lineal y cuadrático): {ame_edad_total:.4f}")

#%%
# 5. ESTIMACIÓN DE PROBIT HETEROCEDÁSTICO Y COMPARACIÓN DE AME
from scipy.optimize import minimize
from scipy.stats import norm

# Usa la muestra limpia alineada del diagnóstico anterior (df_est, X, y, w_norm)
vars_z = ["edad", "mujer", "estudia", "jefe_hogar"]
Z = df_est[vars_z].values  # Sin intercepto para identificación de escala
k_x = X.shape[1]
k_z = Z.shape[1]

def neg_loglik_y_grad(params):
    beta = params[:k_x]
    gamma = params[k_x:]
    
    xb = X @ beta
    inv_sigma = np.exp(-Z @ gamma)
    u = xb * inv_sigma
    
    # Log-verosimilitud ponderada numéricamente estable
    ll = np.where(y == 1, norm.logcdf(u), norm.logcdf(-u))
    nll = -np.sum(w_norm * ll)
    
    # Gradiente analítico exacto
    score_u = np.where(
        y == 1,
        np.exp(norm.logpdf(u) - norm.logcdf(u)),
        -np.exp(norm.logpdf(u) - norm.logcdf(-u))
    )
    g_w = w_norm * score_u
    grad_beta = -(X.T @ (g_w * inv_sigma))
    grad_gamma = Z.T @ (g_w * u)
    
    return nll, np.concatenate([grad_beta, grad_gamma])

# Inicialización en el Probit homocedástico (gamma = 0)
theta_init = np.concatenate([modelo_pooled.params.values, np.zeros(k_z)])
res_het = minimize(
    neg_loglik_y_grad,
    theta_init,
    method="BFGS",
    jac=True,
    options={"gtol": 1e-6, "maxiter": 500}
)

print(f"¿Convergió el Probit Heterocedástico?: {res_het.success}")
beta_het = res_het.x[:k_x]
gamma_het = res_het.x[k_x:]

print("\nParámetros de varianza (gamma en ln(sigma)):")
for var, g in zip(vars_z, gamma_het):
    print(f"  {var:15s}: {g:+.4f}")

# Cálculo de AME corregidos del Probit Heterocedástico
xb_het = X @ beta_het
sigma_het = np.exp(Z @ gamma_het)
u_het = xb_het / sigma_het
phi_u = norm.pdf(u_het)

# Mapeo de coeficientes gamma a las columnas de X (0 si la variable no está en Z)
gamma_en_X = np.zeros(k_x)
nombres_X = modelo_pooled.model.exog_names
for var, g_val in zip(vars_z, gamma_het):
    if var in nombres_X:
        gamma_en_X[nombres_X.index(var)] = g_val

ame_het = np.sum(
    w_norm[:, None] * phi_u[:, None] * ((beta_het[None, :] - xb_het[:, None] * gamma_en_X[None, :]) / sigma_het[:, None]),
    axis=0
)

# Comparación directa de AME: Probit Estándar vs. Probit Heterocedástico
ame_std = np.sum(w_norm[:, None] * norm.pdf(X @ modelo_pooled.params.values)[:, None] * modelo_pooled.params.values[None, :], axis=0)

comparacion_ame = pd.DataFrame({
    "AME Probit Estándar": ame_std,
    "AME Probit Heterocedástico": ame_het,
    "Diferencia": ame_het - ame_std
}, index=nombres_X).drop("Intercept")

print("\n=== COMPARACIÓN DE EFECTOS MARGINALES (ESTÁNDAR VS. HETEROCEDÁSTICO) ===")
print(comparacion_ame.round(4))