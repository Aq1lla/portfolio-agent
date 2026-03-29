# Portfolio Agent — Automatizirani Portfolio Manager

Automatizirani sustav za upravljanje investicijskim portfeljem temeljen na pravilima (Razina A).

## Brzi start

```bash
# 1. Kloniraj i postavi okruženje
git clone <repo-url> && cd portfolio-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Konfiguriraj
cp config/secrets.example.yaml config/secrets.yaml
# Uredi secrets.yaml s Alpaca API ključevima

# 3. Provjeri setup
python scripts/setup_db.py

# 4. Pokreni testove
pytest tests/ -v
```

## Struktura projekta

```
portfolio-agent/
├── config/
│   ├── strategy.yaml          # Konfiguracija strategije
│   ├── secrets.yaml           # API ključevi (gitignore!)
│   └── secrets.example.yaml   # Template za secrets
├── src/
│   ├── models.py              # Pydantic modeli i validacija
│   ├── db.py                  # SQLite baza podataka
│   ├── data_engine.py         # Prikupljanje tržišnih podataka
│   ├── strategy_engine.py     # [Faza 2] Logika strategije
│   ├── execution_engine.py    # [Faza 3] Izvršenje naloga
│   ├── risk_manager.py        # [Faza 2] Upravljanje rizikom
│   └── reporter.py            # [Faza 3] Izvještavanje
├── tests/
│   └── test_phase1.py         # Testovi za Fazu 1
├── scripts/
│   └── setup_db.py            # Inicijalizacija sustava
├── requirements.txt
└── .gitignore
```

## Status implementacije

- [x] **Faza 1**: Infrastruktura i podatkovni sloj
- [ ] **Faza 2**: Strategy i Risk Engine
- [ ] **Faza 3**: Execution Engine i integracija
- [ ] **Faza 4**: Paper trading
- [ ] **Faza 5**: Live trading

## Napomena

Ova analiza je rezultat algoritamske obrade te joj je cilj i svrha biti
pomoćni materijal u procesu donošenja odluke. Automatizirano ulaganje nosi
rizike, uključujući mogućnost gubitka ukupnog uloženog kapitala.
