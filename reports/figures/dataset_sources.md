# Dataset Sources

| Item | Value |
|---|---|
| Dataset name | Uber and Lyft Dataset Boston, MA |
| Link | https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma |
| Kaggle owner | `brllrb` |
| Original file | `rideshare_kaggle.csv` (367 MB). Stored unchanged as `data/raw/uber_raw.csv` |
| Size | 693,071 rows × 57 columns |
| Period | 2018-11-26 → 2018-12-18, Boston, MA |
| Real or synthetic? | **Real.** Price quotes collected from the Uber and Lyft APIs, with real weather joined per area and hour |
| License / terms | See the license shown on the Kaggle page. Used here for non-commercial academic coursework, with credit to the source |
| Changes / merges | No external data was merged. The weather columns were already joined by the dataset author. The file itself was not edited |

## Why only the Lyft rows
Every Uber row has `surge_multiplier = 1.0` (385,663 of 385,663). That value was never really recorded for
Uber, so training on it would teach the model a fake target. Lyft rows have real, varying surge values:

| Surge | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|---|
| Lyft rows | 286,433 | 11,085 | 5,065 | 2,420 | 2,239 | 154 | 12 |

The project therefore predicts the surge multiplier of **Lyft** rides (307,408 rows). The problem type is the same:
ride-hailing surge regression. The ideas sheet allows "a similar dataset from the same category" as long as the
**TA is informed**. The TA was informed about this dataset and agreed to it.

## Sharing
The CSV files are larger than GitHub's 100 MB file limit, so `data/raw/` and the large processed files are listed
in `.gitignore` and shared through Google Drive. The train/test files are small enough to commit.
