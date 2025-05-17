import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

def carregar_arquivo(caminho_arquivo: str, formato_arquivo: str, delimitador: str):
    if formato_arquivo == "excel":
        df = pd.read_excel(caminho_arquivo)
        return pd.DataFrame(df)
    
    elif formato_arquivo == "csv":
        df = pd.read_csv(caminho_arquivo, delimiter=delimitador)
        return pd.DataFrame(df)

if __name__ == "__main__":

    MMA = 8

    base = carregar_arquivo('data/Bovespa.csv', 'csv', ',')

    # Filtra PETR3
    petr3 = base.query("Ticker == 'PETR3'").copy()

    # Converte data
    petr3['Date'] = pd.to_datetime(petr3['Date'])

    # Converte colunas financeiras para float
    colunas_float = ['Close', 'Open', 'High', 'Low', 'Adj Close']
    for col in colunas_float:
        petr3[col] = petr3[col].astype(str).str.replace(',', '.')
        petr3[col] = pd.to_numeric(petr3[col], errors='coerce')

    # Ordena por data
    petr3 = petr3.sort_values('Date')

    # Calcula a média móvel de MMA dias do fechamento
    petr3[f'MM_{MMA}'] = petr3['Close'].rolling(window=MMA).mean()

    petr3.sort_values('Date', inplace=True)

    petr3 = petr3.dropna()

    # Define o tamanho do gráfico
    plt.figure(figsize=(14, 7))

    # Plota o preço de fechamento
    plt.plot(petr3['Date'], petr3['Close'], label='Fechamento', color='blue')

    # Plota a média móvel de 7 dias
    plt.plot(petr3['Date'], petr3[f'MM_{MMA}'], label=f'Média Móvel {MMA} dias', color='orange')

    # Personalização do gráfico
    plt.title('Preço de Fechamento vs Média Móvel - PETR3')
    plt.xlabel('Data')
    plt.ylabel('Preço (R$)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Exibe o gráfico
    plt.show()

    X = petr3[f'MM_{MMA}'].drop(axis=1)

    print(X)

    Xtrain, Xtest, ytrain, yteste = train_test_split(X, )

    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(Xtrain, ytrain)