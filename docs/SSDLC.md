# Secure Software Development Life‑Cycle (SSDLC)

| Phase | Activities |
|-------|------------|
| **Planning** | Collect requirements, security checklist. |
| **Design** | Threat modelling (STRIDE) – single data‑flow from SDK → disk. |
| **Implementation** | Coding in GitHub; pre‑commit lint (black, flake8). |
| **Code Review** | Mandatory PR review by repo owner. |
| **Static Analysis** | GitHub Action: `bandit -r . -f json -o SAST_Bandit.json` + Semgrep default ruleset. |
| **Testing** | manual run in Zoom sandbox. |
| **Release** | Tag + changelog; compiled in GitHub Releases. |
| **Maintenance** | Dependabot alerts weekly; CVE triage < 48 h. |
