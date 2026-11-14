#%%
# 0. Paquetes necesarios
import os
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from scipy.stats import norm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

#%%
# 1. Carga de la base de datos consolidada de jóvenes (2014 - 2026)
RUTA_ARCHIVO = r"C:\Users\jerso\OneDrive\Escritorio\Modelos\base_jovenes_2014_2026.csv"
base_total = pd.read_csv(RUTA_ARCHIVO)

#%%
# 2. Estimación del modelo logit agrupado ponderado (referencia: 4_Media)
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
# 3. Efectos marginales promedio ponderados (AME) y odds ratios
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

#%%
# 4. Exportación a PDF con formato de tabla académica (incluyendo todas las ciudades)
def estrellas(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""

def formatear_nombre_var(var):
    if var.replace(" ", "") == "I(edad**2)":
        return "edad2"
    var = re.sub(r"C\(educacion.*?\)\[T\.(.*?)\]", r"\1", var)
    var = re.sub(r"C\(ciudad\)\[T\.(.*?)\]", r"\1", var)
    var = re.sub(r"C\(anio\)\[T\.(.*?)\]", r"anio_\1", var)
    return var

def exportar_tabla_logit_pdf(modelo, df_ame, ruta_pdf, incluir_anios=False):
    doc = SimpleDocTemplate(
        ruta_pdf,
        pagesize=letter,
        rightMargin=50, leftMargin=50, topMargin=45, bottomMargin=45
    )

    estilo_titulo = ParagraphStyle(
        "TituloTabla", fontName="Times-Roman", fontSize=10.5, leading=13,
        alignment=TA_LEFT, textColor=colors.black
    )
    estilo_header = ParagraphStyle(
        "HeaderCol", fontName="Times-Roman", fontSize=9.5, leading=12,
        alignment=TA_CENTER, textColor=colors.black
    )
    estilo_var = ParagraphStyle(
        "NombreVar", fontName="Times-Italic", fontSize=9.5, leading=11.5,
        alignment=TA_LEFT, leftIndent=8, textColor=colors.black
    )
    estilo_celda = ParagraphStyle(
        "CeldaNum", fontName="Times-Roman", fontSize=9.5, leading=12,
        alignment=TA_CENTER, textColor=colors.black
    )
    estilo_stat = ParagraphStyle(
        "StatNum", fontName="Times-Roman", fontSize=9.5, leading=11.5,
        alignment=TA_CENTER, textColor=colors.black
    )
    estilo_nota = ParagraphStyle(
        "NotaPie", fontName="Times-Roman", fontSize=8.5, leading=11.5,
        alignment=TA_LEFT, textColor=colors.black
    )

    elementos = []

    # Título y variable dependiente
    elementos.append(Paragraph(
        "<b>Tabla 1.</b> Determinantes de la ocupación juvenil — Modelo Logit (Total Nacional, 2014 - 2026).",
        estilo_titulo
    ))
    elementos.append(Paragraph("Variable dependiente: ocupado", estilo_titulo))
    elementos.append(Spacer(1, 6))

    # Incluye controles, niveles educativos y todas las ciudades; omite dummies de año por defecto
    variables = [v for v in modelo.params.index if v != "Intercept"]
    if not incluir_anios:
        variables = [v for v in variables if not v.startswith("C(anio)")]

    filas = [
        ["", "", "", ""],
        [
            "",
            Paragraph("(1) Logit (Coef.)", estilo_header),
            Paragraph("(2) AME (dy/dx)", estilo_header),
            Paragraph("(3) Odds Ratio", estilo_header),
        ]
    ]

    for var in variables:
        # Columna 1: Coeficiente Logit
        b_coef = modelo.params[var]
        se_coef = modelo.bse[var]
        p_coef = modelo.pvalues[var]
        celda_logit = Paragraph(
            f"{b_coef:.4f}{estrellas(p_coef)}<br/>({se_coef:.3f})",
            estilo_celda
        )

        # Columna 2: Efecto Marginal Promedio (AME)
        b_ame = df_ame.loc[var, "AME (dy/dx)"]
        se_a = df_ame.loc[var, "Std. Err."]
        p_a = df_ame.loc[var, "P>|z|"]
        celda_ame = Paragraph(
            f"{b_ame:.4f}{estrellas(p_a)}<br/>({se_a:.3f})",
            estilo_celda
        )

        # Columna 3: Odds Ratio (SE por método delta: exp(beta) * SE(beta))
        or_val = df_ame.loc[var, "Odds Ratio"]
        se_or = or_val * se_coef
        celda_or = Paragraph(
            f"{or_val:.4f}{estrellas(p_coef)}<br/>({se_or:.3f})",
            estilo_celda
        )

        filas.append([
            Paragraph(formatear_nombre_var(var), estilo_var),
            celda_logit,
            celda_ame,
            celda_or
        ])

    # Estadísticos finales
    n_obs = f"{int(modelo.nobs):,}"
    try:
        pseudo_r2 = 1 - (modelo.llf / modelo.llnull)
        r2_txt = f"{pseudo_r2:.4f}"
    except Exception:
        r2_txt = "-"

    idx_n = len(filas)
    filas.append([
        Paragraph("N", estilo_var),
        Paragraph(n_obs, estilo_stat),
        Paragraph(n_obs, estilo_stat),
        Paragraph(n_obs, estilo_stat)
    ])

    idx_r2 = len(filas)
    filas.append([
        Paragraph("Pseudo R2", estilo_var),
        Paragraph(r2_txt, estilo_stat),
        Paragraph(r2_txt, estilo_stat),
        Paragraph(r2_txt, estilo_stat)
    ])

    idx_fin = len(filas)
    filas.append(["", "", "", ""])

    tabla = Table(filas, colWidths=[172, 110, 110, 110], repeatRows=2)
    tabla._rowHeights[0] = 2
    tabla._rowHeights[idx_fin] = 2

    tabla.setStyle(TableStyle([
        # Doble línea superior
        ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("TOPPADDING", (0, 0), (-1, 0), 0),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 0),

        # Línea bajo encabezados
        ("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.black),
        ("TOPPADDING", (0, 1), (-1, 1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 5),

        # Filas de variables
        ("VALIGN", (0, 2), (-1, idx_n - 1), "MIDDLE"),
        ("TOPPADDING", (0, 2), (-1, idx_n - 1), 3),
        ("BOTTOMPADDING", (0, 2), (-1, idx_n - 1), 3),

        # N y Pseudo R2
        ("TOPPADDING", (0, idx_n), (-1, idx_r2), 4),
        ("BOTTOMPADDING", (0, idx_n), (-1, idx_r2), 4),
        ("LINEBELOW", (0, idx_n), (-1, idx_n), 0.35, colors.HexColor("#888888")),

        # Doble línea inferior
        ("LINEABOVE", (0, idx_fin), (-1, idx_fin), 0.5, colors.black),
        ("LINEBELOW", (0, idx_fin), (-1, idx_fin), 0.5, colors.black),
        ("TOPPADDING", (0, idx_fin), (-1, idx_fin), 0),
        ("BOTTOMPADDING", (0, idx_fin), (-1, idx_fin), 0),
    ]))

    elementos.append(tabla)
    elementos.append(Spacer(1, 5))

    notas = [
        "Errores estándar robustos entre paréntesis (método delta para AME y Odds Ratios).",
        "*** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1",
        "<i>Todos</i> los modelos incluyen variables dummy para cada año de la muestra (ref: 2014). Categoría base de educación: 4_Media.",
        "<i>Fuente:</i> Gran Encuesta Integrada de Hogares (GEIH)."
    ]
    for nota in notas:
        elementos.append(Paragraph(nota, estilo_nota))

    doc.build(elementos)
    print(f"\nPDF generado exitosamente en: {ruta_pdf}")

RUTA_PDF = os.path.join(os.path.dirname(RUTA_ARCHIVO), "Tabla1_Logit_Jovenes_2014_2026.pdf")
exportar_tabla_logit_pdf(modelo_logit, tabla_ame, RUTA_PDF, incluir_anios=False)