# banking-book-lab

** Under construction

> A portfolio of quantitative models covering the core disciplines 
> of Asset & Liability Management (ALM/Treasury).  
> Built with Python and standard scientific libraries.

---

## Projects

| # | Project | Key Topics | Regulation |
|---|---------|-----------|------------|
| 1 | [Interest Rate Risk Engine](./01_interest_rate_risk/) | NII at Risk, EVE, repricing gap | BCBS 368 / IRRBB |
| 2 | [Liquidity Risk & Stress Testing](./02_liquidity_risk/) | LCR, NSFR, survival horizon | Basel III CRR2 |
| 3 | [Yield Curve Modeling](./03_yield_curve/) | NSS, Hull-White 1F, Monte Carlo | — |
| 4 | [Prepayment Modeling](./04_prepayment_modeling/) | PSA, logistic regression, OAD | — |
| 5 | [FTP Framework](./05_ftp_framework/) | Matched-maturity FTP, NIM decomposition | — |

---

## Stack

| Library | Usage |
|---------|-------|
| `numpy` / `pandas` | Core data manipulation |
| `scipy` | Optimization & calibration |
| `matplotlib` / `plotly` | Visualization |
| `statsmodels` | Regression-based prepayment model |
| `jupyter` | Notebook interface |

---

## Setup
```bash
conda env create -f environment.yml
conda activate alm-env
jupyter lab
```

---

## Entry point

Start with [`portfolio_summary.ipynb`](./portfolio_summary.ipynb) 
for a one-page overview of all projects and key results.
