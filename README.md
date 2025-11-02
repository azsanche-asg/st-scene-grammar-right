# ST-SceneGrammar


## 🚀 Installation

### Option 1 — pip
```bash
python -m venv .venv
source .venv/bin/activate  # (Linux/macOS)
# or .venv\Scripts\activate  # (Windows)
pip install -r requirements.txt
```

### Option 2 — conda
```bash
conda env create -f environment.yml
conda activate st_scene_grammar
```

### Verify installation
```bash
pytest st_scene_grammar/tests/test_basic.py
jupyter lab
```

---

## 🧩 Recommended Workflow

- Use this repository as your base and sync with **GitHub** for version control.
- Launch experiments from **Colab** by cloning your GitHub repo.
- Modify modules interactively using **Codex CLI** for rapid development.
