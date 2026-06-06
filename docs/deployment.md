# Entrega e consumo

O projeto prioriza clientes com maior propensao a contratar seguro veicular e entrega essa lista por API ou Google Sheets.

## Canais

- **FastAPI:** endpoint `/predict` para score em lote.
- **Google Sheets:** `integrations/google_sheets_appscript.gs` cria colunas de predicao e score na planilha.
- **Lista priorizada:** `data/processed/predictions.csv` gerada pelo comando batch.

## API local

```bash
python -m pip install -r requirements.txt -r requirements-api.txt
PYTHONPATH=src python -m health_insurance_cross_sell.cli train
PYTHONPATH=src uvicorn health_insurance_cross_sell.api:app --reload
```

Teste em outro terminal:

```bash
PYTHONPATH=src python scripts/sample_api_request.py
```

## Google Sheets

1. Crie uma planilha com as colunas de clientes.
2. Abra Extensoes > Apps Script.
3. Cole `integrations/google_sheets_appscript.gs`.
4. Defina `CROSS_SELL_API_URL` com a URL publica de `/predict`.
5. Use o menu **Cross Sell** para gerar e ordenar os scores.
