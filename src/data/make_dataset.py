import subprocess
import pandas as pd
import zipfile
from pathlib import Path


def download_and_extract_data(url: str, raw_data_dir: str):
    """
    Baixa o dataset via curl e extrai usando a biblioteca nativa do Python.
    """
    raw_dir = Path(raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "dataset.zip"

    print(f"Baixando os dados para {raw_dir}...")
    subprocess.run(["curl", "-L", "-o", str(zip_path), url], check=True)

    print("Extraindo o arquivo...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)

    zip_path.unlink(missing_ok=True)

    print("Processo concluído!")


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Carrega os dados brutos de forma segura."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}. Rode o download primeiro.")

    return pd.read_csv(path)