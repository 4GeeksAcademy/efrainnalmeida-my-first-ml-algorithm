# %% [markdown]
# # Tutorial de Proyecto de Regresión Logística

# %% [markdown]
# ## Paso 1: Recopilación de datos

# %%
import pandas as pd

total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv", sep = ";")

total_data.head()

# %%
# Definir la ruta donde se guardará el DataFrame original

ruta_data_frame_original = r"C:/Users/Efrain Almeida/Documents/4Geeks Academy/02 Proyectos/efrainnalmeida-my-first-ml-algorithm/data/raw/total_data.csv"

# Guardar el DataFrame en formato CSV

total_data.to_csv(ruta_data_frame_original, index=False, encoding='utf-8')

# %%
# Crear una copia del DataFrame original

interim_data = total_data.copy()

# Verificar que la copia se ha realizado correctamente

interim_data.head()

# %%
# Definir la ruta donde se guardará el DataFrame de datos intermedios

ruta_data_frame_intermedio = r"C:/Users/Efrain Almeida/Documents/4Geeks Academy/02 Proyectos/efrainnalmeida-my-first-ml-algorithm/data/interim/interim_data.csv"

# Guardar el DataFrame en formato CSV

interim_data.to_csv(ruta_data_frame_intermedio, index=False, encoding='utf-8')

# %% [markdown]
# ## Paso 2: Exploración y limpieza de datos

# %%
# Obtener las dimensiones

interim_data.shape

# %% [markdown]
# El DataFrame contiene 41.188 registros (filas) y 21 variables (columnas)

# %%
# Obtener información sobre tipos de datos y valores no nulos

interim_data.info()

# %% [markdown]
# 🏗 **Consideraciones para Preprocesamiento**
# 
# - **Variables categóricas (`object`)**: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`, `y`.
# 
# - **Variables numéricas (`int64`, `float64`)**: `age`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`.

# %% [markdown]
# ### Eliminar duplicados

# %%
interim_data = interim_data.drop_duplicates().reset_index(drop = True)

interim_data.head()

# %%
# Obtener las dimensiones

interim_data.shape

# %% [markdown]
# Después de eliminar los duplicados, el DataFrame contiene 41.176 registros (filas) y 21 variables (columnas); es decir, contiene 12 registros que el DataFrame original.

# %% [markdown]
# ## Paso 3: Ingeniería de características

# %% [markdown]
# ### Análisis de outliers

# %%
interim_data.describe()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 5, figsize=(12, 10))

sns.boxplot(ax=axes[0, 0], data=interim_data, y = "age")
sns.boxplot(ax=axes[0, 1], data=interim_data, y = "duration")
sns.boxplot(ax=axes[0, 2], data=interim_data, y = "campaign")
sns.boxplot(ax=axes[0, 3], data=interim_data, y = "pdays")
sns.boxplot(ax=axes[0, 4], data=interim_data, y = "previous")
sns.boxplot(ax=axes[1, 0], data=interim_data, y = "emp.var.rate")
sns.boxplot(ax=axes[1, 1], data=interim_data, y = "cons.price.idx")
sns.boxplot(ax=axes[1, 2], data=interim_data, y = "cons.conf.idx")
sns.boxplot(ax=axes[1, 3], data=interim_data, y = "euribor3m")
sns.boxplot(ax=axes[1, 4], data=interim_data, y = "nr.employed")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Análisis de valores faltantes

# %%
# Count NaN

interim_data.isnull().sum().sort_values(ascending = False) / len(interim_data)

# %% [markdown]
# No hay valores faltantes.

# %% [markdown]
# ### Escalado de valores

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Transformar variable objetivo a binaria
interim_data["y"] = interim_data["y"].map({"yes": 1, "no": 0})

# Variables categóricas
categorical_vars = ["job", "marital", "education", "default", "housing", 
                    "loan", "contact", "month", "day_of_week", "poutcome"]

# Variables numéricas
numeric_vars = ["age", "duration", "campaign", "pdays", "previous", 
                "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

# One-Hot Encoding para las categóricas
data_dummies = pd.get_dummies(interim_data[categorical_vars], drop_first=True)

# Concatenar con las variables numéricas
X = pd.concat([interim_data[numeric_vars], data_dummies], axis=1)
y = interim_data["y"]

# División en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar solo las variables numéricas
scaler = MinMaxScaler()
X_train_scaled_numeric = scaler.fit_transform(X_train[numeric_vars])
X_test_scaled_numeric = scaler.transform(X_test[numeric_vars])

# Cambiar explícitamente el tipo de las columnas numéricas a float64 para evitar el warning
X_train[numeric_vars] = X_train[numeric_vars].astype("float64")
X_test[numeric_vars] = X_test[numeric_vars].astype("float64")

# Sustituir las columnas numéricas por sus versiones escaladas (ya compatibles)
X_train.loc[:, numeric_vars] = X_train_scaled_numeric
X_test.loc[:, numeric_vars] = X_test_scaled_numeric

# %% [markdown]
# ## Paso 4: Selección de características

# %%
from sklearn.feature_selection import SelectKBest, f_classif

# Asegurar que los índices estén limpios
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Selección de características
selector = SelectKBest(score_func=f_classif, k=5)
X_train_sel_array = selector.fit_transform(X_train, y_train)
X_test_sel_array = selector.transform(X_test)

# Obtener nombres de columnas seleccionadas
selected_cols = X_train.columns[selector.get_support()]

# Crear DataFrames con índices alineados
X_train_sel = pd.DataFrame(X_train_sel_array, columns=selected_cols, index=X_train.index)
X_test_sel = pd.DataFrame(X_test_sel_array, columns=selected_cols, index=X_test.index)

# Añadir la variable objetivo a los DataFrames seleccionados
X_train_sel["y"] = y_train
X_test_sel["y"] = y_test

# %%
X_train_sel.head()

# %%
X_test_sel.head()

# %% [markdown]
# ### Guardar los datos limpios

# %%
# Guardar a CSV
X_train_sel.to_csv("C:/Users/Efrain Almeida/Documents/4Geeks Academy/02 Proyectos/efrainnalmeida-my-first-ml-algorithm/data/processed/clean_train.csv", index=False)
X_test_sel.to_csv("C:/Users/Efrain Almeida/Documents/4Geeks Academy/02 Proyectos/efrainnalmeida-my-first-ml-algorithm/data/processed/clean_test.csv", index=False)

# %% [markdown]
# ## Paso 5: Modelo de regresión logística

# %%
# Leer los datos procesados

train_data = pd.read_csv("C:/Users/Efrain Almeida/Documents/4Geeks Academy/02 Proyectos/efrainnalmeida-my-first-ml-algorithm/data/processed/clean_train.csv")
test_data = pd.read_csv("C:/Users/Efrain Almeida/Documents/4Geeks Academy/02 Proyectos/efrainnalmeida-my-first-ml-algorithm/data/processed/clean_test.csv")

# %%
train_data.head()

# %%
test_data.head()

# %%
# Establecer X_train, y_train, X_test, y_test

X_train = train_data.drop(["y"], axis = 1)
y_train = train_data["y"]
X_test = test_data.drop(["y"], axis = 1)
y_test = test_data["y"]

# %%
# Modelo de regresión logística

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# %%
# Predicciones

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# %%
coef = pd.Series(model.coef_[0], index=X_train.columns)
print(coef.sort_values(ascending=False))

# %% [markdown]
# 🧠 ¿Qué significa cada uno?
# 
# Los coeficientes representan el impacto marginal (log-odds) de cada variable sobre la probabilidad de que y = 1 (es decir, que el cliente sí contrate el depósito).
# 
# 🔼 Coeficientes positivos → aumentan la probabilidad de y = 1:
# 
# - duration (19.66): mientras más larga fue la llamada, mucho más probable que el cliente diga sí.
# 
# - ⚠️ Este valor es alto porque duration está escalado (MinMaxScaler).
# 
# - poutcome_success (0.70): si la campaña previa fue exitosa, hay más chance de éxito ahora.
# 
# 🔽 Coeficientes negativos → disminuyen la probabilidad de y = 1:
# 
# - euribor3m (-0.34): tasas más altas se asocian con menor probabilidad de contratación.
# 
# - pdays (-0.92): mientras más días pasaron desde la última campaña, menos probabilidad de éxito.
# 
# - nr.employed (-3.20): más empleados (indicador económico fuerte) → menor necesidad de ahorro/depósito.

# %% [markdown]
# 🧩 Nota sobre `duration`
# 
# `duration` suele tener una correlación muy fuerte con el target, pero es una variable que no se conoce antes de que el cliente responda, así que:
# 
# - 🔒 En problemas reales, a menudo se excluye duration del modelo para evitar data leakage.

# %%
# Evaluación

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# %%
# Resultados

print(f'Exactitud: {accuracy:.4f}') # qué proporción de predicciones fueron correctas
print(f'Precisión: {precision:.4f}') # de todas las veces que el modelo dijo "sí", ¿cuántas fueron realmente "sí"?
print(f'Recall: {recall:.4f}') # de todos los casos reales "sí", ¿cuántos encontró el modelo?
print(f'AUC: {roc_auc:.4f}') # área bajo la curva ROC, que mide la capacidad del modelo para distinguir entre clases

# %% [markdown]
# 📌 Interpretación:
# 
# - ✅ AUC = 0.91 → Excelente capacidad del modelo para distinguir entre clases.
# 
# - ✅ Exactitud = 0.90 → Muy alta, pero ojo...
# 
# - ⚠️ Recall = 0.34 → El modelo está fallando al detectar muchos positivos.
# 
# - ⚠️ Precisión = 0.65 → Cuando predice "sí", acierta un 65% de las veces.
# 
# Esto es típico cuando:
# 
# - La clase positiva (y=1) es minoritaria (desbalance de clases).
# 
# - El modelo prefiere predecir “no” para maximizar la exactitud.

# %%
# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar con heatmap
plt.figure(figsize=(4, 3))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No", "Sí"], yticklabels=["No", "Sí"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.show()

# %% [markdown]
# 🧠 ¿Qué dice esto?
# 
# - El modelo es muy bueno prediciendo los "No" (alta especificidad).
# 
# - Pero está fallando en detectar los "Sí" (recall bajo).
# 
# - Hay problemas de desbalance de clases.

# %%
# Curva ROC

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Aleatorio")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Paso 6: Optimización del modelo

# %%
from sklearn.model_selection import GridSearchCV
import warnings

# Evitar que se impriman advertencias innecesarias (pero no silenciar errores críticos)
warnings.filterwarnings("ignore")

# Definimos los hiperparámetros y sus posibles valores

hyperparams = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

# Inicializar GridSearchCV
grid = GridSearchCV(estimator=model,
                    param_grid=hyperparams,
                    scoring="roc_auc",
                    cv=5,
                    n_jobs=-1,
                    verbose=1)

# Ajustar el modelo
grid.fit(X_train, y_train)

# Resultados
print("✅ Mejores hiperparámetros:", grid.best_params_)
print("🎯 Mejor AUC:", round(grid.best_score_, 4))

# %%
# Reentrenar el modelo con los mejores hiperparámetros

best_model = LogisticRegression(C=0.001, penalty=None, solver="lbfgs", max_iter=1000)
best_model.fit(X_train, y_train)

y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1]

# %%
# Evaluación - Best Model

conf_matrix_best = confusion_matrix(y_test, y_pred_best)
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
fpr_best, tpr_best, thresholds_best = roc_curve(y_test, y_prob_best)
roc_auc_best = auc(fpr_best, tpr_best)

# %%
# Resultados - Best Model

print(f'Exactitud: {accuracy_best:.4f}') 
print(f'Precisión: {precision_best:.4f}') 
print(f'Recall: {recall_best:.4f}') 
print(f'AUC: {roc_auc_best:.4f}')

# %%
# Calcular la matriz de confusión - Best Model
conf_matrix_best = confusion_matrix(y_test, y_pred_best)

# Visualizar con heatmap - Best Model
plt.figure(figsize=(4, 3))
sns.heatmap(conf_matrix_best, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["No", "Sí"], yticklabels=["No", "Sí"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Best Model")
plt.tight_layout()
plt.show()

# %%
# Curva ROC - Best Model

plt.figure(figsize=(6, 4))
plt.plot(fpr_best, tpr_best, color="blue", label=f"ROC curve (AUC = {roc_auc_best:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Aleatorio")
plt.xlabel("Tasa de Falsos Positivos (FPR)")
plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
plt.title("Curva ROC - Best Model")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
import pickle

# Ruta donde guardar el modelo
ruta_modelo = "C:/Users/Efrain Almeida/Documents/4Geeks Academy/02 Proyectos/efrainnalmeida-my-first-ml-algorithm/models/logistic_model.pkl"

# Guardar el modelo
with open(ruta_modelo, "wb") as file:
    pickle.dump(best_model, file)

print("✅ Modelo guardado correctamente.")


