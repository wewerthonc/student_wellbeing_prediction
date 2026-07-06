import matplotlib.pyplot as plt
import seaborn as sns


def plotar_distribuicao_alvo(df, target_col="Depression"):
    """Gera o gráfico de contagem da variável-alvo."""
    plt.figure(figsize=(8, 4.8))
    dados_x = df[target_col].map({0: "Sem depressão", 1: "Com depressão"})

    ax = sns.countplot(x=dados_x, hue=dados_x, palette=["#2563eb", "#dc2626"], legend=False)
    ax.set_title(f"Distribuição da Variável-Alvo ({target_col})")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade")

    for p in ax.patches:
        value = int(p.get_height())
        ax.annotate(
            f"{value}\n({value / len(df):.1%})",
            (p.get_x() + p.get_width() / 2, value),
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.show()


def plotar_dados_ausentes(df_raw):
    """Gera o gráfico de barras verticais mostrando dados ausentes."""
    missing = df_raw.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    plt.figure(figsize=(8, 3.6))
    if len(missing) > 0:
        ax = sns.barplot(
            x=missing.values,
            y=missing.index,
            hue=missing.index,
            palette="viridis",
            legend=False,
        )
        ax.set_title("Dados Ausentes por Atributo Original")

        ax.set_xlabel("Quantidade")
        ax.set_ylabel("")

        for i, v in enumerate(missing.values):
            ax.text(v + 0.05, i, str(v), va="center")
    else:
        plt.text(0.5, 0.5, "Não foram encontrados valores ausentes.", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plotar_distribuicoes_numericas(df, target_col="Depression"):
    """Gera histogramas combinados das principais variáveis numéricas e derivadas."""
    cols_interesse = [
        "Age",
        "Academic Pressure",
        "Financial Stress",
        "Work/Study Hours",
        "Pressure Satisfaction Ratio",
        "Stress Academic Index",
    ]

    plot_cols = [c for c in cols_interesse if c in df.columns]

    if not plot_cols:
        print("Nenhuma das colunas numéricas de interesse foi encontrada no dataset.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.ravel()

    for ax, col in zip(axes, plot_cols, strict=False):
        sns.histplot(
            data=df,
            x=col,
            hue=target_col,
            bins=24,
            multiple="stack",
            palette=["#2563eb", "#dc2626"],
            ax=ax,
        )
        ax.set_title(col)
        ax.set_xlabel("")

    for ax in axes[len(plot_cols) :]:
        ax.axis("off")

    fig.suptitle("Distribuições Numéricas e Features Derivadas", y=1.02)
    plt.tight_layout()
    plt.show()


def plotar_matriz_correlacao(df):
    """Calcula e exibe o heatmap de correlação linear (Pearson)."""
    numeric_features = df.select_dtypes(exclude="object").columns.tolist()
    corr = df[numeric_features].corr(numeric_only=True)

    plt.figure(figsize=(13, 10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, cbar_kws={"shrink": 0.75})
    plt.title("Matriz de Correlação Linear")
    plt.tight_layout()
    plt.show()


def plotar_taxas_categoricas(df, target_col="Depression"):
    """Gera gráficos de barra horizontal com a taxa da classe positiva por categoria."""
    cols_interesse = [
        "Gender",
        "Sleep Duration",
        "Dietary Habits",
        "Have you ever had suicidal thoughts ?",
        "Family History of Mental Illness",
    ]

    plot_cols = [c for c in cols_interesse if c in df.columns]

    if not plot_cols:
        print("Nenhuma das colunas categóricas foi encontrada no dataset.")
        return

    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(11, 3.1 * len(plot_cols)))

    if len(plot_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, plot_cols, strict=False):
        rates = df.groupby(col)[target_col].mean().sort_values(ascending=False)

        sns.barplot(
            x=rates.values, y=rates.index, hue=rates.index, palette="viridis", legend=False, ax=ax
        )

        ax.set_xlim(0, 1)
        ax.set_title(f"Taxa de {target_col}=1 por {col}")
        ax.set_xlabel("Proporção Positiva")
        ax.set_ylabel("")

        for i, v in enumerate(rates.values):
            ax.text(min(v + 0.015, 0.98), i, f"{v:.1%}", va="center")

    plt.tight_layout()
    plt.show()
