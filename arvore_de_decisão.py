import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

base = pd.read_csv('./data/Bovespa.csv', delimiter=',')

def limpar_dados(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    
    base['Date'] = pd.to_datetime(base['Date'], dayfirst=True)
    base = base[base['Ticker'] == 'PETR3']

    colunas_float = ['Close', 'Open', 'High', 'Low', 'Adj Close']
    for col in colunas_float:
        base[col] = base[col].astype(str).str.replace(',', '.')
        base[col] = pd.to_numeric(base[col], errors='coerce')

    base = base.sort_values('Date')

    return base

def calcular_indicadores(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()

    base['SMA_10'] = base['Close'].rolling(window=10).mean()
    base['SMA_50'] = base['Close'].rolling(window=50).mean()

    base['EMA_10'] = base['Close'].ewm(span=10, adjust=False).mean()

    base['Volatilidade_10'] = base['Close'].rolling(window=10).std()

    base['Retorno_Diario'] = base['Close'].pct_change()

    return base

base = limpar_dados(base)
base = calcular_indicadores(base)

base = base.dropna()

X = base[['SMA_10', 'SMA_50', 'EMA_10', 'Volatilidade_10', 'Retorno_Diario']]
y = base['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

modelo = DecisionTreeRegressor()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'Erro Médio Absoluto (MAE): {mae:.2f}')
print(f'Erro Quadrático Médio (MSE): {mse:.2f}')

plt.figure(figsize=(12, 6))
plt.plot(base['Date'].iloc[-len(y_test):], y_test, label='Real', linestyle='-')
plt.plot(base['Date'].iloc[-len(y_test):], y_pred, label='Previsto', linestyle='--')
plt.xlabel('Data')
plt.ylabel('Preço')
plt.title('Previsão do Preço de PETR3 - Árvore de Decisão')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.show()
