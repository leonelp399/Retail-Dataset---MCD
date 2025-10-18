# 📘 README — Ejecución del Notebook `MCD_Fundamento_Computacional_Trabajo_14.ipynb`

**Autor:** Leonel Pérez Moscoso  
**Curso:** Fundamento Computacional — MCD  
**Lenguaje:** Python (Jupyter / Google Colab)  
**Versión:** 1.0  

---

## 🧭 Descripción general

Este notebook desarrolla un **Análisis Exploratorio de Datos (EDA)** sobre un dataset de ventas retail.  
Incluye limpieza, análisis de correlaciones, segmentación RFM, evolución temporal de ingresos y detección de productos con alta tasa de cancelación.

---

## ⚙️ Librerías requeridas

Ejecutar al inicio del notebook:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

> 💡 En **Google Colab**, ya están incluidas; en local, instálalas con:
> ```bash
> pip install pandas numpy matplotlib seaborn statsmodels
> ```

---

## ▶️ Orden de ejecución y dependencias entre variables

A continuación se detalla el **orden recomendado**, las **variables clave creadas en cada bloque** y los **métodos principales usados**.

---

### **1️⃣ Carga y configuración inicial**
**Propósito:** preparar entorno y parámetros gráficos.

```python
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid", palette="muted", font_scale=1.1)
```

---

### **2️⃣ Carga del dataset**
**Métodos y variables:**
```python
df = pd.read_csv("Online Retail.csv", encoding="ISO-8859-1")
df.head(), df.info()
```

**Variables generadas:**
- `df`: DataFrame base con todas las transacciones.

---

### **3️⃣ Limpieza y preparación**
**Objetivo:** eliminar registros inválidos y preparar nuevas columnas.

```python
df_clean = (
    df.dropna(subset=['CustomerID'])
      .query("Quantity > 0 and UnitPrice > 0")
      .copy()
)
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean['Revenue'] = df_clean['Quantity'] * df_clean['UnitPrice']
df_clean['Year'] = df_clean['InvoiceDate'].dt.year
df_clean['Month'] = df_clean['InvoiceDate'].dt.month
df_clean['MonthName'] = df_clean['InvoiceDate'].dt.month_name()
df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.day_name()
df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
```

**Variables generadas:**  
`df_clean`, `Revenue`, `Year`, `Month`, `Hour`, `YearMonth`

---

### **4️⃣ Distribuciones y exploración**
**Propósito:** entender la forma de las variables principales.

**Métodos usados:**
```python
sns.histplot(df_clean['Revenue'], bins=50)
sns.boxplot(x=df_clean['Quantity'])
sns.countplot(x='DayOfWeek', data=df_clean)
```

**Variables utilizadas:**  
`df_clean`, `Revenue`, `Quantity`, `DayOfWeek`

---

### **5️⃣ Correlaciones numéricas**
**Propósito:** cuantificar relaciones entre variables.

**Cálculo Spearman:**
```python
num_cols = ['Revenue','Quantity','UnitPrice','Hour','Month','Year']
corr_spearman = df_clean[num_cols].corr(method='spearman')
sns.heatmap(corr_spearman, annot=True, cmap='coolwarm')
```

**Ranking de correlaciones:**
```python
ranking = corr_spearman['Revenue'].sort_values(ascending=False)
```

**Resultado esperado:**  
`Quantity` tiene mayor correlación con `Revenue`.

---

### **6️⃣ Multicolinealidad (VIF)**
**Propósito:** verificar redundancia entre predictores.

**Variables y métodos:**
```python
Xcols = ["Quantity", "UnitPrice", "Hour", "Month", "Year"]
X = df_clean[Xcols].dropna()
X_const = sm.add_constant(X)
vif = pd.DataFrame({
    "Variable": X_const.columns,
    "VIF": [variance_inflation_factor(X_const.values, i)
            for i in range(X_const.shape[1])]
})
```

**Resultado:** todos los VIF < 2 → sin colinealidad.

---

### **7️⃣ Análisis RFM (Recency, Frequency, Monetary)**
**Propósito:** segmentar clientes según valor de compra.

```python
ref_date = df_clean['InvoiceDate'].max().normalize()

rfm = (
    df_clean.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda s: (ref_date - s.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('Revenue', 'sum')
    )
)

rfm['R'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1])
rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['M'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['RFM_Score'] = rfm[['R','F','M']].astype(int).sum(axis=1)
```

**Variables generadas:**  
`rfm`, `R`, `F`, `M`, `RFM_Score`

---

### **8️⃣ Clientes de alto valor**
**Identificación y visualización:**
```python
umbral_alto = 10
rfm["HighValue"] = rfm["RFM_Score"] >= umbral_alto

share_high = (
    rfm.groupby("CountryDominant")["HighValue"].mean().sort_values(ascending=False)*100
)
sns.barplot(x=share_high.values, y=share_high.index)
```

**Variables clave:**  
`HighValue`, `share_high`

---

### **9️⃣ Evolución semanal del Revenue**
**Objetivo:** observar tendencia temporal de ingresos.

```python
ventas_semanales = (
    df_clean.set_index('InvoiceDate')
    .resample('W')['Revenue'].sum()
    .reset_index()
)
sns.lineplot(data=ventas_semanales, x='InvoiceDate', y='Revenue', marker='o')
```

**Variable generada:**  
`ventas_semanales`

---

### **🔟 Análisis por país (Revenue y Ticket promedio)**
**Agrupamiento y visualización:**
```python
ventas = df_clean.groupby(['Country','YearMonth'])['Revenue'].sum().reset_index()
ticket_prom = (
    df_clean.groupby(['Country','YearMonth','InvoiceNo'])['Revenue']
             .sum().groupby(['Country','YearMonth']).mean().reset_index(name='TicketPromedio')
)
patrones_pais = ventas.merge(ticket_prom, on=['Country','YearMonth'])
```

**Variables:**  
`ventas`, `ticket_prom`, `patrones_pais`

---

### **1️⃣1️⃣ Análisis de cancelaciones**
**Propósito:** detectar productos con más devoluciones.

```python
df2 = df_clean.copy()
df2['is_credit'] = df2['InvoiceNo'].astype(str).str.startswith('C')

vend = df2[~df2['is_credit']].groupby('StockCode')['Quantity'].sum().reset_index(name='QtyVendida')
canc = df2[df2['is_credit']].groupby('StockCode')['Quantity'].apply(lambda s: np.abs(s).sum()).reset_index(name='QtyCancelada')

rates = vend.merge(canc, on='StockCode', how='left').fillna(0)
rates['CancelRate'] = rates['QtyCancelada'] / rates['QtyVendida']
```

**Variables:**  
`df2`, `vend`, `canc`, `rates`, `CancelRate`

**Visualización top 20:**
```python
top_cancel = rates.query('QtyVendida >= 50').sort_values('CancelRate', ascending=False).head(20)
sns.barplot(data=top_cancel, x='CancelRate', y='StockCode', orient='h')
```

---

## 📈 Orden lógico resumido

| Paso | Tema principal | Variables clave |
|------|----------------|-----------------|
| 1 | Configuración inicial | — |
| 2 | Carga del dataset | `df` |
| 3 | Limpieza y creación de Revenue | `df_clean`, `Revenue` |
| 4 | Análisis descriptivo | — |
| 5 | Correlaciones | `corr_spearman` |
| 6 | Multicolinealidad | `vif` |
| 7 | RFM | `rfm`, `RFM_Score` |
| 8 | Clientes de alto valor | `share_high` |
| 9 | Evolución semanal | `ventas_semanales` |
| 10 | Ticket promedio por país | `patrones_pais` |
| 11 | Cancelaciones | `rates`, `CancelRate` |

---

## 🧠 Resultados esperados

1. Tendencia creciente de ingresos hacia final del año.  
2. Segmentación clara de clientes de alto valor.  
3. Países con distinto perfil: UK (volumen) vs Singapore/Netherlands (ticket alto).  
4. Productos con tasas de cancelación >100%, indicando incidencias.  
5. Insights accionables para marketing, logística y fidelización.

---

## 📄 Autor
**Leonel Pérez Moscoso**  
*MCD — Fundamento Computacional 2025*  
📧 leonelp399@gmail.com

---
