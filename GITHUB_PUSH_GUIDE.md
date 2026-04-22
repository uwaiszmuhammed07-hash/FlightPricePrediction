# 🚀 GitHub Push & Deploy Guide

## PART 1 — Push to GitHub

### Step 1: Create GitHub repo
1. Go to https://github.com → sign in
2. Click + → New repository
3. Name: FlightPricePrediction → Public → Create

### Step 2: Run these commands in VS Code terminal

```bash
git init
git add .
git commit -m "Initial commit — PRCP-1025 Flight Price Prediction"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/FlightPricePrediction.git
git push -u origin main
```
Replace YOUR_USERNAME with your actual GitHub username.

> For password: use a Personal Access Token
> GitHub → Settings → Developer Settings → Personal access tokens → Generate → tick repo → copy → paste as password

---

## PART 2 — Save model and push it

```bash
python save_model.py
git add model.pkl model_columns.pkl data/Flight_Fare.xlsx
git commit -m "Add trained model and dataset"
git push
```

---

## PART 3 — Deploy on Streamlit Cloud (Free)

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click New app
4. Select: Repository = FlightPricePrediction, Branch = main, File = app.py
5. Click Deploy

Your app will be live in 2-3 minutes at a public URL you can share!

---

## Future updates

```bash
git add .
git commit -m "describe your change"
git push
```
Streamlit Cloud auto-redeploys on every push.
