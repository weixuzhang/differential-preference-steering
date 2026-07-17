# lamp_benchmark

LaMP/LongLaMP benchmark utilities used throughout DPS: dataset loading
(`load_lamp_dataset`), RAG prompt construction (`create_prompt_generator`),
retrievers (BM25/contriever/ICR/RankGPT), and task metrics
(`create_metric`, `get_labels`).

Dataset files are resolved via the `LAMP_DATA_ROOT` environment variable
(default `/scratch/weixuz/lamp_data`), with one subdirectory per task
(`LaMP-1/` … `LongLaMP-4/`). See `scripts/prefetch_lamp_datasets.py` to
(re)download raw files.
