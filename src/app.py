import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

url = 'https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv'
df = pd.read_csv(url)

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

# Sustituir valores 0
def sustituir_ceros(df, columna):

    media = df[columna][df[columna]>0].mean()

    def sustituir_valor(valor):
        if valor == 0:
            return media
        else:
            return valor
    
    df[columna] = df[columna].apply(sustituir_valor)

columnas = list(df.columns.values)
columnas.remove('Pregnancies')
columnas.remove('Outcome')

for columna in columnas:
    sustituir_ceros(df_train, columna)

X_train = df_train.drop(columns=['Outcome'])
y_train = df_train['Outcome']
df_train = pd.concat([X_train, y_train], axis=1)

for columna in columnas:
    sustituir_ceros(df_test, columna)

X_test = df_test.drop(columns=['Outcome'])
y_test = df_test['Outcome']
df_test = pd.concat([X_test, y_test], axis=1)

# Eliminar outliers
def limites_outliers(columna):
    q1 = X_train[columna].quantile(0.25)
    q3 = X_train[columna].quantile(0.75)
    IQR = q3 - q1
    min_so = q1 - 1.5*IQR
    max_so = q3 + 1.5*IQR
    return [min_so, max_so]

limites_preg = limites_outliers('Pregnancies')
limites_ST = limites_outliers('SkinThickness')
limites_insulin = limites_outliers('Insulin')

df_train = df_train.drop(df_train[df_train['Pregnancies']>limites_preg[1]].index)
df_train = df_train.drop(df_train[df_train['SkinThickness']<limites_ST[0]].index)
df_train = df_train.drop(df_train[df_train['Insulin']<limites_insulin[0]].index)

X_train = df_train.drop(columns=['Outcome'])
y_train = df_train['Outcome']
df_train = pd.concat([X_train, y_train], axis=1)

# Guardar df_train y df_test
df_train.to_csv('../data/processed/df_train.csv')
print('Se guardó df_train como csv')

df_test.to_csv('../data/processed/df_test.csv')
print('Se guardó df_test como csv')

# Construir modelo
modelo_hp = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=3)

modelo_hp.fit(X_train, y_train)

# Guardar modelo
filename = '../models/modelo_tree.sav'
pickle.dump(modelo_hp, open(filename, 'wb'))

print('Se guardó el modelo')