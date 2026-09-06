# MLFlow Tracking

MLFlow logs every training run with parameters, metrics, and artifacts.

## Start the dashboard

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

Then open `http://localhost:5000` in your browser.

## What gets logged?

- **Parameters**: base model, LoRA rank/alpha, learning rate, batch size, etc.
- **Metrics**: training loss per step, learning rate schedule
- **Artifacts**: the saved adapter directory

## Comparing runs

In the MLFlow UI you can:
- Compare multiple runs side-by-side
- Plot loss curves
- See which hyperparameters produced the best results
