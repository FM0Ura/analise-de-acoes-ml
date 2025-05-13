import pandas as pd

def carregar_arquivo(caminho_arquivo: str, formato_arquivo: str, delimitador: str):
    if formato_arquivo == "excel":
        df = pd.read_excel(caminho_arquivo)
        return pd.DataFrame(df)
    
    elif formato_arquivo == "csv":
        df = pd.read_csv(caminho_arquivo ,delimiter=delimitador)
        return pd.DataFrame(df)

def resgatar_colunas(base_dados: pd.DataFrame) -> list:
    return list(base_dados.columns)


if __name__ == "__main__":

    base = carregar_arquivo('data/Bovespa.csv', 'csv', ',')

    colunas = resgatar_colunas(base)

    print(colunas)

    petr3 = base[base['Ticker'].isin(['PETR3'])]

    petr3['Date'] = pd.to_datetime(petr3['Date'])

    sum_fechamento = petr3['Close'].str.replace(',','.', regex=False).astype(float).sum()

    data_mais_antiga, data_mais_nova = petr3['Date'].min(), petr3['Date'].max()

    dif_dias = abs((data_mais_antiga - data_mais_nova).days)

    print(sum_fechamento)

    mma = sum_fechamento / dif_dias # Média Móvel Aritmética

    print(f'''A Média Móvel Aritmética da PETR3 foi {mma}\n
            ''')
    
    multiplicador = (2 / (dif_dias + 1))*100

    mme = "" # FALTA IMPLEMENTAR