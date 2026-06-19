import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe o DataFrame bruto e cria novas features derivadas linha a linha.
    Limpa strings e mapeia variáveis categóricas ordinais para formatos numéricos.
    """
    # Cria uma cópia para não alterar o DataFrame original em memória
    data = df.copy()

    # 1. Limpeza básica de strings
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype(str).str.strip()

    # 2. Dicionários de mapeamento
    sleep_midpoint = {
        "Less than 5 hours": 4.5,
        "5-6 hours": 5.5,
        "7-8 hours": 7.5,
        "More than 8 hours": 8.5,
        "Others": np.nan,
    }
    sleep_risk = {
        "Less than 5 hours": 2.0,
        "5-6 hours": 1.0,
        "7-8 hours": 0.0,
        "More than 8 hours": 1.0,
        "Others": np.nan,
    }
    diet_risk = {"Healthy": 0.0, "Moderate": 1.0, "Unhealthy": 2.0, "Others": np.nan}
    yes_no = {"No": 0.0, "Yes": 1.0}

    # 3. Mapeamento de features baseadas em dicionários
    if "Sleep Duration" in data.columns:
        data["Sleep Hours Midpoint"] = data["Sleep Duration"].map(sleep_midpoint)
        data["Sleep Risk Score"] = data["Sleep Duration"].map(sleep_risk)

    if "Dietary Habits" in data.columns:
        data["Diet Risk Score"] = data["Dietary Habits"].map(diet_risk)

    if "Have you ever had suicidal thoughts ?" in data.columns:
        data["Suicidal Thoughts Binary"] = data["Have you ever had suicidal thoughts ?"].map(yes_no)

    if "Family History of Mental Illness" in data.columns:
        data["Family History Binary"] = data["Family History of Mental Illness"].map(yes_no)

    # 4. Criação de features combinadas e de interação
    def get_col(name):
        """Função auxiliar para garantir que a coluna existe e evitar erros de KeyError."""
        return data[name] if name in data.columns else 0.0

    total_pressure = get_col("Academic Pressure") + get_col("Work Pressure")
    total_satisfaction = get_col("Study Satisfaction") + get_col("Job Satisfaction")

    data["Total Pressure"] = total_pressure
    data["Total Satisfaction"] = total_satisfaction

    # Adiciona 1.0 para evitar divisão por zero
    data["Pressure Satisfaction Ratio"] = (total_pressure + 1.0) / (total_satisfaction + 1.0)
    data["Stress Academic Index"] = get_col("Academic Pressure") + get_col("Financial Stress")
    data["Load Pressure Interaction"] = get_col("Work/Study Hours") * get_col("Academic Pressure")
    data["CGPA Pressure Interaction"] = get_col("CGPA") * get_col("Academic Pressure")

    return data