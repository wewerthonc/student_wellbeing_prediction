"""Download, extract, and load the student depression dataset."""

import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from student_depression.config import SETTINGS
from student_depression.paths import RAW_DATA_DIR, RAW_DATA_FILE


def _extract_zip_safely(archive_path: Path, destination: Path) -> None:
    """Extract an archive only when every member stays inside the destination."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(destination):
                message = f"Unsafe path in dataset archive: {member.filename}"
                raise ValueError(message)
        archive.extractall(destination)


def download_dataset(
    url: str = SETTINGS.dataset.url,
    destination: Path = RAW_DATA_DIR,
) -> Path:
    """Download and extract the dataset, returning the expected CSV path."""
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "dataset.zip"

    request = urllib.request.Request(url, headers={"User-Agent": "student-depression/0.1"})
    try:
        with urllib.request.urlopen(request) as response, archive_path.open("wb") as archive_file:
            shutil.copyfileobj(response, archive_file)
        _extract_zip_safely(archive_path, destination)
    finally:
        archive_path.unlink(missing_ok=True)

    if not RAW_DATA_FILE.exists():
        message = f"Dataset archive did not contain the expected file: {RAW_DATA_FILE.name}"
        raise FileNotFoundError(message)
    return RAW_DATA_FILE


def ensure_dataset(data_path: Path = RAW_DATA_FILE) -> Path:
    """Return the local dataset path, downloading the default dataset when absent."""
    if data_path.exists():
        return data_path
    if data_path != RAW_DATA_FILE:
        message = f"Dataset not found: {data_path}"
        raise FileNotFoundError(message)
    return download_dataset()


def load_dataset(data_path: str | Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load a CSV dataset and reject empty input."""
    path = Path(data_path)
    if not path.is_file():
        message = f"Dataset not found: {path}. Run the project to download it first."
        raise FileNotFoundError(message)

    data = pd.read_csv(path)
    if data.empty:
        message = f"Dataset is empty: {path}"
        raise ValueError(message)
    return data
