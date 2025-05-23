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

def aplicar_RL_simples(base: pd.DataFrame, feature: str, target: str, nome_base: str, percentual_teste: float, estado_randomico: int = 0):
    X = base[[feature]]
    y = base[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percentual_teste, shuffle=False, random_state=estado_randomico)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"[{nome_base} - RL Simples] MSE: {mse:.4f} | R²: {r2:.4f}")

    # Gráfico MA_5 vs Close (com linha de regressão)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Valor Real')
    plt.plot(X_test, y_pred, color='red', label='Valor Previsto (Regressão)', linewidth=2)
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f'{nome_base} - Regressão Linear Simples: {feature} vs {target}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    base = carregar_arquivo('data/Bovespa.csv', 'csv', ',')

    petr3 = base.query("Ticker == 'PETR3'")

    petr3 = limpar_dados(petr3)

    petr3 = adicionar_colunas_calculadas(petr3)

    petr4 = base.query("Ticker == 'PTR4'")

    petr4['Ticker'] = petr4['Ticker'].astype(str).replace('PTR4', "PETR4")

    petr4 = limpar_dados(petr4)

    petr4 = adicionar_colunas_calculadas(petr4)

    colunas = ['Close', 'Open', 'High', 'Low', 'Volume']
    target = 'MA_5'

    aplicar_RL(petr3 ,colunas, target, 'PETR3', percentual_teste=0.3)
    aplicar_RL(petr4 ,colunas, target, 'PETR4', percentual_teste=0.3)