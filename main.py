import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv('movies.csv')
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

# Função para realizar a regressão linear e plotar o gráfico
def regressao_linear_e_plot(x_col, y_col, df, ax, title):
    X = df[[x_col]].dropna()
    y = df[y_col][X.index]

    # Dividir os dados em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Fazer predições
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Plotar o gráfico
    sns.regplot(x=X_test[x_col], y=y_test, line_kws={"color": "red"}, scatter_kws={"alpha": 0.5}, ax=ax)
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel(y_col.capitalize())
    ax.set_title(f'{title}\nR²: {r2:.4f} | MAE: {mae:.4f}')
    return r2, mae

# Criar figuras para os gráficos
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# Regressão para Popularidade
r2_popularity, mae_popularity = regressao_linear_e_plot('popularity', 'vote_average', df, axes[0], 'Popularidade vs Nota Média')

# Regressão para Ano
r2_year, mae_year = regressao_linear_e_plot('year', 'vote_average', df, axes[1], 'Ano vs Nota Média')

# Regressão para Contagem de Votos
r2_vote_count, mae_vote_count = regressao_linear_e_plot('vote_count', 'vote_average', df, axes[2], 'Contagem de Votos vs Nota Média')

# Ajustar layout e mostrar gráficos
plt.tight_layout()
plt.show()

# Resultados
print(f'Resultados da Regressão Linear:')
print(f'Popularidade vs Nota Média: R² = {r2_popularity:.4f}, MAE = {mae_popularity:.4f}')
print(f'Ano vs Nota Média: R² = {r2_year:.4f}, MAE = {mae_year:.4f}')
print(f'Contagem de Votos vs Nota Média: R² = {r2_vote_count:.4f}, MAE = {mae_vote_count:.4f}')
