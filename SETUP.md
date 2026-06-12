# Opening this project in VS Code and publishing to GitHub

## 1. Open in VS Code
- Unzip the folder.
- In VS Code: File -> Open Folder -> select `governance-runtime-iot`.
- When prompted, install the recommended extensions (Python, OPA).

## 2. Set up Python
Open a terminal in VS Code (Ctrl + `) and run:
```bash
pip install -r requirements.txt
```

## 3. Run / verify (or press F5 and pick a configuration)
```bash
python3 experiment_runner.py     # generate traces
python3 stats_analysis.py        # Tables 5-7
python3 verify_corrected_g2.py   # Table 10 (18.7% -> 0.0%)
python3 verify_audit_chain.py    # tamper-evident audit chain (G4)
```

## 4. Publish to GitHub (easiest: VS Code GUI)
- Open Source Control (Ctrl + Shift + G).
- Click "Publish to GitHub".
- Choose a name (e.g. governance-runtime-iot) and Public.
- Done — VS Code initializes git, commits, and pushes for you.

### Or via terminal
```bash
git init
git add .
git commit -m "Initial commit: governance runtime reproducibility package"
git branch -M main
git remote add origin https://github.com/USERNAME/governance-runtime-iot.git
git push -u origin main
```

## 5. After publishing
- Make the repository Public so reviewers can access it.
- Replace the placeholder URL in the manuscript's Data Availability Statement
  with your real repository URL.
- (Recommended) Connect the repo to Zenodo and create a release to obtain a DOI.
