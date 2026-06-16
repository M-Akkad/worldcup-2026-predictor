# data/

The match data is **not committed to the repo**, you download it yourself
(see the main [README](../README.md#get-the-data-you-download-it-yourself)).

Place these files here:

- **`results.csv`**, *required.* Kaggle's *International football results from
  1872 to 2026* (martj42). The whole engine reads this file.
- `goalscorers.csv`, `shootouts.csv`, `former_names.csv`, optional companions
  from the same Kaggle dataset.

No Kaggle account? From the repo root run `python make_demo_data.py` to generate
a synthetic `results.csv` for testing (numbers are indicative, not real).

`wc2026.db` (the web app's SQLite database) is created automatically on first run.
