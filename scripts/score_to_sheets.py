"""Score a CSV file with the trained model and write results to Google Sheets.

Setup
-----
1. Create a Google Cloud project and enable the Google Sheets API.
2. Create an OAuth2 client ID (Desktop app) and download the JSON credentials file.
3. Save the credentials file as ``credentials.json`` in the project root
   (or pass its path via ``--credentials``).
4. On first run, a browser window will open for authentication. After granting
   access, a ``token.json`` file is saved for subsequent runs.
5. Share your Google Sheet with the service account email (if using a service
   account) or ensure the authenticated user has edit access.

Usage
-----
    python scripts/score_to_sheets.py \\
        --input data/raw/test.csv \\
        --sheet-id YOUR_GOOGLE_SHEET_ID

    # Optional flags:
    #   --model   models/model.joblib   (default)
    #   --config  configs/project.toml  (default)
    #   --worksheet "Sheet1"            (default)
    #   --credentials credentials.json  (default)
    #   --top-k 20000                   (only write top-k rows by score)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _authenticate(credentials_path: Path) -> object:
    """Return an authorised gspread client using OAuth2."""
    import gspread
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
    ]
    token_path = credentials_path.parent / "token.json"
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path), scopes
            )
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
    return gspread.authorize(creds)


def score_and_upload(
    input_path: Path,
    sheet_id: str,
    model_path: Path,
    config_path: Path,
    worksheet_name: str,
    credentials_path: Path,
    top_k: int | None,
) -> None:
    import joblib
    import pandas as pd

    from health_insurance_cross_sell.config import load_config
    from health_insurance_cross_sell.features import model_matrix, prepare_features

    print(f"Loading model from {model_path} ...")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run `make train` or notebook 04 first."
        )
    model = joblib.load(model_path)

    print(f"Loading data from {input_path} ...")
    raw = pd.read_csv(input_path)
    config = load_config(config_path)
    prepared = prepare_features(raw.copy(), config, training=False)
    X, _, _ = model_matrix(prepared, config, training=False)

    print("Scoring ...")
    scores = model.predict_proba(X)[:, 1]
    result = raw.copy()
    result["score"] = scores
    result["prediction"] = model.predict(X)
    result = result.sort_values("score", ascending=False).reset_index(drop=True)

    if top_k is not None:
        result = result.head(top_k)
        print(f"Keeping top {top_k} rows by score.")

    print(f"Uploading {len(result):,} rows to Google Sheet {sheet_id} ...")
    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Credentials file not found: {credentials_path}.\n"
            "Follow the setup instructions in this script's docstring."
        )
    gc = _authenticate(credentials_path)
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
    except Exception:
        ws = sh.add_worksheet(title=worksheet_name, rows=str(len(result) + 10), cols="30")

    ws.clear()
    ws.update([result.columns.tolist()] + result.values.tolist())
    print(f"Done. Worksheet '{worksheet_name}' updated with {len(result):,} rows.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score customers and upload results to Google Sheets."
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV file to score")
    parser.add_argument("--sheet-id", required=True, help="Google Sheet ID")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "model.joblib",
        help="Path to trained model joblib file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "project.toml",
        help="Path to project TOML config",
    )
    parser.add_argument(
        "--worksheet",
        default="Sheet1",
        help="Worksheet name inside the Google Sheet",
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=PROJECT_ROOT / "credentials.json",
        help="OAuth2 credentials JSON file from Google Cloud Console",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Only upload top-K rows by score (e.g. 20000)",
    )
    args = parser.parse_args()

    score_and_upload(
        input_path=args.input,
        sheet_id=args.sheet_id,
        model_path=args.model,
        config_path=args.config,
        worksheet_name=args.worksheet,
        credentials_path=args.credentials,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
