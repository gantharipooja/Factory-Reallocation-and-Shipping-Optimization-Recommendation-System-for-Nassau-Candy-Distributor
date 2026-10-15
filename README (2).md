# Nassau Candy: Shipping Route Efficiency

Factory-to-customer shipping route analysis: data cleaning, EDA, route benchmarking,
statistical testing and an interactive Streamlit dashboard.

## Files
| File | Purpose |
|---|---|
| `nassau_candy_analysis.py` | Cleaning, date calibration, route KPIs, tests, figures, insights report |
| `app.py` | Streamlit dashboard (reuses the analysis module) |
| `requirements.txt` | Python dependencies |
| `Nassau_Candy_Distributor.csv` | Dataset (add this file yourself, next to `app.py`) |
| `sample_output/` | Example output from the analysis script |

## Run
```bash
pip install -r requirements.txt

# 1. Batch analysis: cleaned data, tables, figures, insights.md
python nassau_candy_analysis.py --data Nassau_Candy_Distributor.csv --output outputs

# 2. Dashboard
streamlit run app.py
```

## Data note
Raw ship dates sit 904 to 1,642 days after order dates in three clusters (a systematic export offset).
The pipeline calibrates each cluster using Same Day shipments as the zero anchor. Lead time is
order-to-dispatch, not delivery. See the research paper for details and limitations.
