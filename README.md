\# AI Data Scientist



Transparent, local-first automated machine learning for tabular classification and regression.



The application guides a user from a CSV or Excel dataset to exploratory analysis, safe preprocessing, dataset-aware model comparison, bounded tuning, explainability, downloadable model/report artifacts, and predictions from the verified fitted pipeline. It runs locally and does not require an LLM, an API key, or a cloud service.



Current release: \*\*0.10.3\*\*  

Verified project baseline: \*\*250 backend tests\*\* and \*\*53 frontend tests\*\*



\## What a user can do



1\. Upload a `.csv`, `.xlsx`, or `.xls` dataset (maximum 100 MB).

2\. Review its shape, missing values, duplicates, preview, and target suggestions.

3\. Choose the value to predict; suggestions never override the user's choice.

4\. Generate a classification or regression EDA report with insights and charts.

5\. Start training. The system recommends a small, diverse set of relevant models instead of training every registered algorithm.

6\. Compare cross-validation results, optionally tune the leaders, and select a winner without using the holdout set for selection.

7\. Review holdout metrics, model-selection reasons, feature importance, limitations, and warnings.

8\. Download the complete fitted pipeline, integrity manifest, JSON report, Markdown report, PDF report, and generated charts.

9\. Enter new raw records and generate classification or regression predictions without retraining.



\## Verified scope



| Capability | Status |

|---|---|

| CSV and Excel upload | Complete |

| Target suggestions and manual target selection | Complete |

| Classification/regression inference | Complete |

| Automated EDA, insights, and charts | Complete |

| Missing-value handling and categorical encoding | Complete |

| Outlier analysis with robust scaling | Complete |

| Leakage checks | Complete |

| Training-only feature selection | Complete |

| Dataset-aware model recommendation | Complete |

| Cross-validation and bounded tuning | Complete |

| Untouched holdout evaluation | Complete |

| Winner/runner-up/baseline rationale | Complete |

| Permutation and native importance where available | Complete |

| Verified model artifact export | Complete |

| Local Markdown and PDF reports | Complete |

| Classification and regression prediction UI | Complete |

| Authentication, multi-user jobs, deployment | Not implemented |

| Run history and automatic retention/cleanup | Not implemented |

| CSV bulk-prediction UI | Not implemented |



\## System at a glance



```mermaid

flowchart TD

&#x20;   UI\["React browser interface"] --> API\["FastAPI application"]

&#x20;   API --> DATA\["Upload and EDA pipeline"]

&#x20;   API --> ML\["Training and explanation pipeline"]

&#x20;   API --> INFER\["Verified inference pipeline"]

&#x20;   DATA --> STORE\["Local uploads and run artifacts"]

&#x20;   ML --> STORE

&#x20;   STORE --> INFER

```



The frontend runs at `http://127.0.0.1:5173` and the API at `http://127.0.0.1:8000` by default. Uploaded data and generated artifacts remain on the local computer.



\## Technology



\- Python 3.12

\- FastAPI and Uvicorn

\- pandas, NumPy, SciPy, and scikit-learn

\- Matplotlib and ReportLab

\- React 19, TypeScript, and Vite 8

\- pytest and Node's built-in test runner



The currently verified development machine uses Ubuntu 24.04 under WSL2, Python 3.12.3, Node 22.23.2, and npm 10.9.8. The project is not inherently Linux-only, but WSL is the recommended route on this machine because Windows Enterprise Application Control blocks the unsigned SciPy `\_flapack.pyd` binary.



\## Repository layout



```text

AI-Data-Scientist/

├── backend/

│   ├── app/

│   │   ├── api/                 # Upload, EDA, training, artifacts, inference

│   │   ├── core/                # Configuration and safe filesystem helpers

│   │   ├── pipeline/            # Analysis, preprocessing, ML, reports

│   │   └── main.py              # FastAPI app, CORS, health routes

│   ├── uploads/                 # Uploaded datasets (runtime data)

│   └── outputs/

│       ├── eda/                 # EDA runs by analysis ID

│       └── models/api/          # Training runs by training ID

├── frontend/

│   ├── src/                     # React/TypeScript application

│   ├── tests/                   # Frontend contract tests

│   └── dist/                    # Production build output

├── tests/                       # Backend tests

├── docs/                        # Project and milestone documentation

├── requirements.txt

└── README.md

```



`backend/uploads` and `backend/outputs` are application data, not source code. Do not delete them during a normal code update unless the data has been backed up and removal is intentional.



\## Quick start for the verified WSL setup



Open Ubuntu 24.04 from Windows PowerShell:



```powershell

wsl -d Ubuntu-24.04

```



In WSL, activate the existing Python environment and enter the project:



```bash

cd "/mnt/d/fianl year/AI-Data-Scientist"

source /home/antariksha\_yewatkar/venvs/ai-data-scientist/bin/activate

```



Load the existing nvm-managed Node installation:



```bash

export NVM\_DIR="$HOME/.nvm"

\[ -s "$NVM\_DIR/nvm.sh" ] \&\& . "$NVM\_DIR/nvm.sh"

nvm use 22

```



Start the API in Terminal 1:



```bash

MPLBACKEND=Agg python -m uvicorn \\

&#x20;   backend.app.main:app \\

&#x20;   --host 127.0.0.1 \\

&#x20;   --port 8000

```



Start the frontend in Terminal 2:



```bash

cd "/mnt/d/fianl year/AI-Data-Scientist/frontend"

export NVM\_DIR="$HOME/.nvm"

\[ -s "$NVM\_DIR/nvm.sh" ] \&\& . "$NVM\_DIR/nvm.sh"

nvm use 22

npm run dev

```



Open `http://127.0.0.1:5173` manually in a Windows browser. Keep both terminals running. Stop each server with `Ctrl+C` in its own terminal.



For a new machine or a clean reinstall, follow \[docs/SETUP\_GUIDE.md](docs/SETUP\_GUIDE.md).



\## Run all tests



Backend, from the project root with the Python environment active:



```bash

MPLBACKEND=Agg python -m pytest -v -W error

```



Expected for the verified 0.10.3 source tree:



```text

250 passed

```



Frontend:



```bash

cd frontend

npm test

```



Expected:



```text

53 passed

```



`npm test` performs strict TypeScript compilation, a Vite production build, and the frontend tests. Test totals can legitimately increase when more tests are added; no failures or warnings-as-errors should remain.



\## How model selection works



The catalog contains \*\*14 non-baseline classification candidates\*\* and \*\*14 non-baseline regression candidates\*\*, plus one dummy baseline for each problem type. The automatic workflow does not train all 14.



By default, the recommender inspects only the training split and chooses up to four diverse, compatible candidates. Cross-validation compares them with the dummy baseline. Optional tuning searches the top two candidates with 12 bounded randomized iterations. The final holdout is evaluated only after the winner has been selected.



\- Classification is ranked primarily by balanced accuracy (higher is better).

\- Regression is ranked primarily by RMSE (lower is better).

\- A model is “best” only among the successfully evaluated candidates for that dataset and configuration.

\- The selected algorithm can change when the dataset, target, software versions, random seed, or settings change.



See \[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full catalog and leakage-safe workflow.



\## Data preparation behavior



\- Rows with a missing target are removed before splitting because supervised training cannot learn from them.

\- Constant, identifier-like, and excessively missing features can be excluded by the cleaning plan.

\- Review/high-cardinality fields are not silently allowed into training.

\- Numeric missing values use median imputation.

\- Categorical missing values use most-frequent imputation.

\- Categorical values are one-hot encoded with unknown values ignored safely.

\- Standard scaling is used normally; outlier-affected numeric features can use robust scaling.

\- Outliers are reported, but rows are \*\*not automatically deleted\*\*.

\- Potential target leakage blocks training unless the user explicitly acknowledges it after review.

\- Feature selection occurs inside the fitted pipeline using training data only.



\## Artifacts and reports



A successful exported training run can include:



| Artifact | Purpose |

|---|---|

| `model\_pipeline.joblib` | Complete fitted preprocessing, feature selection, and estimator |

| `artifact\_manifest.json` | File hashes, sizes, verification state, and bundle metadata |

| `training\_report.json` | Machine-readable public training result |

| `model\_report.md` | Editable human-readable professional report |

| `model\_report.pdf` | Portable professional report |

| `\*.png` or `\*.svg` | EDA and explainability charts |



Reports are generated locally from the computed results. No LLM or external API is needed. The exported pipeline is specific to the selected target and input schema. A model trained to predict `Survived` cannot be used as a `Fare` model.



Important: joblib/pickle-based artifacts are executable serialized content. Load only artifacts created by a trusted copy of this application. SHA-256 verifies that a file has not changed; it does not make an untrusted file safe.



\## Prediction behavior



The Predict step loads the saved model schema, asks only for raw input fields, and runs the exact verified pipeline without fitting or tuning again.



\- Classifiers return a predicted class and, when supported, confidence/class probabilities.

\- Regressors return a continuous numerical estimate and no probabilities.

\- The browser supports up to 50 manually entered records per request.

\- The backend is independently bounded at 1,000 records.

\- Starting a new dataset/target/training flow clears stale prediction context.



\## API summary



| Method | Path | Purpose |

|---|---|---|

| GET | `/` | Service message |

| GET | `/health` | Health check |

| POST | `/upload` | Upload and inspect a dataset |

| POST | `/analysis/target` | Validate the selected target |

| POST | `/analysis/eda` | Generate EDA and optional plots |

| GET | `/analysis/eda/artifacts/{analysis\_id}/{artifact\_filename}` | Download one EDA artifact |

| POST | `/analysis/train` | Recommend, compare, tune, explain, report, and export |

| GET | `/analysis/train/artifacts/{training\_id}/{artifact\_filename}` | Download an approved training artifact |

| GET | `/analysis/predict/schema/{training\_id}` | Inspect verified model inputs |

| POST | `/analysis/predict` | Predict with a verified saved pipeline |



Interactive API documentation is available while the backend runs at `http://127.0.0.1:8000/docs`. See \[docs/API\_REFERENCE.md](docs/API\_REFERENCE.md) for request examples and status codes.



\## Security and privacy boundary



This is a local, single-user development application. It includes filename/path traversal checks, isolated run identifiers, controlled artifact allowlists, upload/request bounds, explicit CORS origins, model hash verification, strict JSON safety, and public-report filtering. Raw holdout rows are not included in public responses or reports.



It does \*\*not\*\* include authentication, authorization, encryption-at-rest management, malware scanning, multi-tenant isolation, a background job queue, rate limiting, or production deployment controls. Do not expose the development server to an untrusted network. Read \[docs/SECURITY\_AND\_LIMITATIONS.md](docs/SECURITY\_AND\_LIMITATIONS.md) before any deployment.



\## Documentation



\- \[Architecture](docs/ARCHITECTURE.md)

\- \[Complete setup guide](docs/SETUP\_GUIDE.md)

\- \[User guide](docs/USER\_GUIDE.md)

\- \[API reference](docs/API\_REFERENCE.md)

\- \[Testing and troubleshooting](docs/TESTING\_AND\_TROUBLESHOOTING.md)

\- \[Security and limitations](docs/SECURITY\_AND\_LIMITATIONS.md)

\- \[Project status and roadmap](docs/PROJECT\_STATUS\_AND\_ROADMAP.md)




\## Responsible interpretation



The tool automates technical comparisons; it does not prove causality, fairness, legal compliance, or suitability for a high-stakes decision. Accuracy is not the only metric, confidence is not certainty, feature importance is not causation, and a holdout score is not a production guarantee. A domain expert must validate the target definition, leakage timing, data rights, model impact, and deployment monitoring plan.



