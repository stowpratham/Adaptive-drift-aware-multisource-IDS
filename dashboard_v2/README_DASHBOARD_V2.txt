ADAPTIVE IDS DASHBOARD V2

This package replaces the dashboard presentation/data-loading layer only.
It does not change ids_main.py or the IDS detection algorithm.

1. Back up your current dashboard folder.
2. Copy the contents of dashboard_v2/dashboard/ over your project's dashboard/ folder.
3. From the project root, activate .venv311:
   .\.venv311\Scripts\activate
4. Run:
   python -m streamlit run dashboard\app.py

New/updated pages:
- System Overview
- Stream Monitoring
- Model Comparison
- Explainable AI

The Model Comparison page uses results/baselines/final_comparison.csv for persisted
evaluation metrics and displays the eight recorded model training times separately.
