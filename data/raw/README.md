# Dados

Fonte Kaggle: [Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction).

Arquivos esperados nesta pasta:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

- O treino usa `Response` como alvo e preserva `id` para ranking e submissao.

## Download via Kaggle API

```bash
mkdir -p data/raw
kaggle datasets download -d anmolkumar/health-insurance-cross-sell-prediction --unzip -p data/raw
find data/raw -maxdepth 1 -name "*.zip" -exec unzip -q -o {} -d data/raw \;
```

Mantenha arquivos grandes fora do Git quando necessario e baixe-os novamente no Colab ou no ambiente local.
