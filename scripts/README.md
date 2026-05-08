# Python Script Exports

These files are linear Python exports of the notebooks in `notebooks/`.

They are useful for:

- Reviewing the project as source code on GitHub
- Searching model, data, and evaluation logic
- Comparing framework implementations without opening notebook outputs
- Reusing code in future scripts or apps

The notebooks remain the primary narrative artifacts because they include explanations, cell outputs, charts, and training context. The scripts are cleaned exports for portfolio readability and static review.

## Script Order

| Order | Script | Source Notebook |
|---:|---|---|
| 1 | `01_data_loading_memory_vs_generator.py` | `notebooks/01_data_loading_memory_vs_generator.ipynb` |
| 2 | `02_keras_data_pipeline.py` | `notebooks/02_keras_data_pipeline.ipynb` |
| 3 | `03_pytorch_data_pipeline.py` | `notebooks/03_pytorch_data_pipeline.ipynb` |
| 4 | `04_keras_cnn_classifier.py` | `notebooks/04_keras_cnn_classifier.ipynb` |
| 5 | `05_pytorch_cnn_classifier.py` | `notebooks/05_pytorch_cnn_classifier.ipynb` |
| 6 | `06_keras_vs_pytorch_cnn_comparison.py` | `notebooks/06_keras_vs_pytorch_cnn_comparison.ipynb` |
| 7 | `07_keras_cnn_vit_hybrid.py` | `notebooks/07_keras_cnn_vit_hybrid.ipynb` |
| 8 | `08_pytorch_cnn_vit_hybrid.py` | `notebooks/08_pytorch_cnn_vit_hybrid.ipynb` |
| 9 | `09_final_cnn_vit_evaluation.py` | `notebooks/09_final_cnn_vit_evaluation.ipynb` |

## Notes

Some scripts still depend on local data, downloaded files, and trained model artifacts. For a production-style command-line workflow, the next step would be to refactor repeated notebook code into reusable modules under `src/`.
