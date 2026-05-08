# Notebooks

The notebooks are ordered to tell the full modeling story from data handling to final model comparison.

| Order | Notebook | Purpose |
|---:|---|---|
| 1 | `01_data_loading_memory_vs_generator.ipynb` | Compare memory-based and generator-based image loading |
| 2 | `02_keras_data_pipeline.ipynb` | Build Keras data loading and augmentation workflows |
| 3 | `03_pytorch_data_pipeline.ipynb` | Build PyTorch Dataset and DataLoader workflows |
| 4 | `04_keras_cnn_classifier.ipynb` | Train and evaluate a Keras CNN baseline |
| 5 | `05_pytorch_cnn_classifier.ipynb` | Train and evaluate a PyTorch CNN baseline |
| 6 | `06_keras_vs_pytorch_cnn_comparison.ipynb` | Compare CNN results across frameworks |
| 7 | `07_keras_cnn_vit_hybrid.ipynb` | Build and train a Keras CNN-ViT hybrid |
| 8 | `08_pytorch_cnn_vit_hybrid.ipynb` | Build and train a PyTorch CNN-ViT hybrid |
| 9 | `09_final_cnn_vit_evaluation.ipynb` | Compare final hybrid models |

These notebooks were kept as the primary record of the capstone workflow. The `src/` folder contains small reusable helpers that can support future cleanup, inference scripts, or dashboard work.
