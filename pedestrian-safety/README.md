# Pedestrian in Traffic: Agentic AI Framework for Crowd-Safety Early Warning

BITE308L Artificial Intelligence, Project Component (VIT, Fall 2026-27)

## Phases

| Phase | What gets built | Main files | Runs on | Done when |
|---|---|---|---|---|
| **0. Setup** | Folders, Python environment, data check | `00_check_data.py` | Laptop | `results/data_check.txt` looks right |
| **1. Data prep** | WiderPerson person crops; Melbourne cleaning + features (time, events, last-hour counts) + next-hour level | `image_pipeline/01_make_crops.py`, `sensor_pipeline/01_clean_features.py`, `data/events.csv` | Laptop | Crop folders filled; `features.parquet` saved |
| **2. Sensor models** | 8 ML models compared; best one saved; early-warning replay on hidden event days | `sensor_pipeline/02_train_compare.py`, `03_early_warning.py` | Laptop (CPU) | Comparison table + charts in `results/` |
| **3. Image models** | 5 backbones: extract fingerprints once, train each classifier | `image_pipeline/02_extract_features.py`, `03_train_single.py` | Kaggle/Colab GPU | 5 accuracy/F1 results |
| **4. Hybrids + ensemble** | 3 fusion models + soft voting; YOLO person detector tool | `image_pipeline/04_hybrids_ensemble.py`, `05_detector.py` | Kaggle/Colab GPU | Ensemble beats single models; `image_ensemble.pt` saved |
| **5. Agent** | 6 tools, risk rules, LLM loop + backup script, 20 test scenarios | `agent/tools.py`, `rules.py`, `agent.py`, `test_scenarios.json` | Laptop | Agent tests pass |
| **6. Dashboard** | Flask app: Plan ahead · Live watch · Camera check · Ask the agent · Model results | `app/` | Laptop | Full demo runs end-to-end |
| **7. Wrap-up** | Final results, report sections, viva/demo script | `results/` | — | Ready for review |

## Folder structure

```
pedestrian-safety/
├── config.py                  all paths, classes, thresholds (one place)
├── 00_check_data.py           Phase 0: dataset check
├── requirements.txt
├── .env.example               copy to .env, add API key (Phase 5)
├── data/
│   ├── widerperson/           put WiderPerson here (Images/, Annotations/, train.txt, val.txt)
│   ├── melbourne/             put the hourly counts CSV here
│   ├── events.csv             events calendar (filled in Phase 1)
│   ├── cameras.csv            camera registry (filled in Phase 5)
│   └── processed/
│       ├── crops/             person crops by class (Phase 1)
│       └── features/          backbone fingerprints .npy (Phase 3)
├── image_pipeline/            Phases 1, 3, 4
├── sensor_pipeline/           Phases 1, 2
├── agent/                     Phase 5
├── app/                       Phase 6 (templates/, static/)
├── saved_models/              trained models the app loads
├── results/
│   ├── figures/               charts for the report
│   └── tables/                comparison tables
└── notebooks/                 Kaggle/Colab notebooks for GPU phases
```

## Phase 0: setup (Windows PowerShell)

Requires Python 3.11 (3.10–3.12 also fine).

```powershell
cd path\to\pedestrian-safety
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas
python 00_check_data.py
```

If activation is blocked, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Install the full `requirements.txt` later (Phase 1), not now. `torch` is a large download.

## Run the dashboard (Phase 6)

```powershell
python -m pip install flask
python app\app.py
```
Open **http://127.0.0.1:5000** in Chrome. Stop the server with **Ctrl+C**.

| Screen | What it shows |
|---|---|
| Overview | Live crowd simulation (pressure follows the replay clock) + every camera's forecast |
| Plan ahead | 24-hour forecast grid for all cameras on any day (past days use real data, future days use the calendar) |
| Live watch | Replay a day hour by hour: actual crowd, usual crowd, Very High line, early warnings as they are issued |
| Camera check | Upload or pick a crowd photo: YOLO finds people, the ensemble labels them, rules give the risk |
| Ask the agent | Gemini plans which tools to use; each step streams in live |
| Model results | All scores and charts from Phases 2-5 |

The **replay clock** (top bar) drives every screen. Use *Jump to* for event days such as the AFL Grand Final parade.

## Put it online (Hugging Face Spaces, free)

```powershell
python -m pip install huggingface_hub
python deploy\make_space.py            # builds deploy\space\ (only what the website needs)
python deploy\make_space.py --push     # asks for your HF token, creates the Space, uploads
```
The Space builds itself from `deploy/Dockerfile` (CPU PyTorch, gunicorn on port 7860).
`GEMINI_API_KEY` from `.env` is stored as a hidden Space secret, never in the uploaded files.
