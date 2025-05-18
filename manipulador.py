import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def carregar_arquivo(caminho_arquivo: str, formato_arquivo: str, delimitador: str) -> pd.DataFrame:
    if formato_arquivo == "excel":
        df = pd.read_excel(caminho_arquivo)
        return pd.DataFrame(df)
    
    elif formato_arquivo == "csv":
        df = pd.read_csv(caminho_arquivo, delimiter=delimitador)
        return pd.DataFrame(df)

def limpar_dados(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base['Date'] = pd.to_datetime(base['Date'], dayfirst=True)

    colunas_float = ['Close', 'Open', 'High', 'Low', 'Adj Close']
    for col in colunas_float:
        base[col] = base[col].astype(str).str.replace(',', '.')
        base[col] = pd.to_numeric(base[col], errors='coerce')

    base = base.sort_values('Date')

    return base

def adicionar_colunas_calculadas(base: pd.DataFrame) -> pd.DataFrame:

    base['MA_5'] = base.groupby('Ticker')['Close'].transform(lambda x: x.rolling(5).mean())
    base['Volatility_5'] = base.groupby('Ticker')['Close'].transform(lambda x: x.rolling(5).std())

    base.sort_values('Date')

    # 5. Remover valores nulos
    base = base.dropna(subset=['MA_5', 'Volatility_5'])

    return base

def aplicar_RL(base, features, target):

    X = base[features]
    y = base[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 8. Treinamento do modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 9. Previsões
    y_pred = model.predict(X_test)

    # 10. Avaliação
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # 11. Plotar gráfico de valores reais vs previstos
    plt.figure(figsize=(20, 6))
    plt.plot(base['Date'].iloc[-len(y_test):], y_test, label='Valor Real', color='blue')
    plt.plot(base['Date'].iloc[-len(y_pred):], y_pred, label='Valor Previsto', color='red', linestyle='--')
    plt.xlabel('Data')
    plt.ylabel('Preço de Fechamento (Close)')
    plt.title('Regressão Linear - Valor Real vs Valor Previsto')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    base = carregar_arquivo('data/Bovespa.csv', 'csv', ',')

    petr3 = base.query("Ticker == 'PETR3'")

    petr3 = limpar_dados(petr3)

    petr3 = adicionar_colunas_calculadas(petr3)

    colunas = ['Open', 'High', 'Low', 'Volume', 'MA_5', 'Volatility_5']
    target = 'Close'

    aplicar_RL(petr3 ,colunas, target)