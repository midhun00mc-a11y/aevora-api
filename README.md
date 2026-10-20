# Aevora Fit Intelligence API

A compact API-first fit recommendation demo built with FastAPI, Pydantic, Google's official `google-genai` SDK, and SQLite. Catalog data stays in code; customer memory is stored separately in a local SQLite file.

## Windows setup (PowerShell)

Run these commands from the project folder. Set the current location first if needed:

```powershell
Set-Location "C:\Users\MIDHUN M\OneDrive\Documents\ChatGPT\aevora apt"
python -m venv .venv
Copy-Item .env.example .env
notepad .env
```

In `.env`, replace the placeholder with a Gemini API key and save the file. Create a key in [Google AI Studio](https://aistudio.google.com/app/apikey). Keep `.env` private; it is excluded from Git. Install packages and run the API without activating the virtual environment (this also avoids PowerShell script execution-policy issues):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Leave that terminal window open and browse to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). If `.env` has no usable key or Gemini is temporarily unavailable, the rules-based fallback still responds. The API response reports `recommendation_source`, `fallback_used`, and `fallback_reason`.

The key can also be set in the current PowerShell session instead of `.env`:

```powershell
$env:GEMINI_API_KEY = "paste-your-key-here"
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Do not paste a real key into chat, source code, or a committed file. The implementation uses Google's official `google-genai` SDK, its current Interactions API, structured JSON output, and a 10-second SDK request timeout. The Gemini model name defaults to `gemini-3.8-flash` and can be changed with `GEMINI_MODEL`.

## Recommendation behavior

The existing `POST /v1/fit/recommend` endpoint and sample products (`tee-001`, `shirt-001`) are preserved. The deterministic scorer remains in the backend. It ranks in-stock sizes using chest ease (preference targets: slim 6 cm, regular 12 cm, relaxed 20 cm), usual size, and optional height/garment length. Gemini receives the customer's usual size and fit preference plus the verified product size chart, garment measurements, and stock list. Chest and height measurements are used locally by the rules and are not sent to Gemini or stored.

Gemini can propose a size and a short preference-style note. The backend accepts the proposed size only when it exists in the product measurements and is currently in stock. The backend always supplies the catalog values and constructs the factual explanation; it filters AI notes that contain size labels, numbers, or measurement/availability claims. Invalid or out-of-stock proposals use the deterministic ranking.

Example request:

```json
{
  "product_id": "tee-001",
  "usual_size": "M",
  "fit_preference": "relaxed",
  "height_cm": 175,
  "chest_cm": 94
}
```

`usual_size` and `fit_preference` can be omitted after they have been explicitly saved for that `customer_id`. The default demo customer is `demo-customer-001`. Unknown products return 404; invalid input or missing required preference information returns 422; a product with no in-stock sizes returns 409.

## Demo fit memory

SQLite stores only a customer's explicit `usual_size`/`fit_preference` and categorical feedback (`too_tight`, `too_loose`, `fits_well`) associated with a product and recommended size. It does not persist body measurements. Latest feedback for that product nudges the local ease target by 4 cm for “too tight” or “too loose”; “fits well” is recorded without a size adjustment. Responses expose `memory.feedback_count`, `feedback_used`, and `preferences_used`, so an empty memory is not presented as learned behavior.

Routes (also available in Swagger UI):

- `GET /v1/customers/{customer_id}/preferences` — read saved preferences and feedback count.
- `PUT /v1/customers/{customer_id}/preferences` — explicitly save/update either or both values, for example `{"usual_size":"M","fit_preference":"regular"}`.
- `DELETE /v1/customers/{customer_id}/preferences` — delete saved preferences.
- `POST /v1/customers/{customer_id}/feedback` — save `product_id`, `recommended_size`, and a sentiment.
- `DELETE /v1/customers/{customer_id}/memory` — delete both preferences and feedback for that customer.

The default SQLite file is `aevora_memory.sqlite3`; set `AEVORA_DB_PATH` to choose another path. The file is ignored by Git. This demo has no authentication: a `customer_id` is only a data lookup key, not an identity or access-control boundary. Do not expose it publicly or use real customer data until authentication, authorization, retention, and data-protection controls are added. Google receives the usual size, fit preference, latest categorical feedback when present, and verified catalog data for a recommendation; chest and height measurements are not sent or stored. Avoid sending data you do not want processed by the Gemini API.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover recommendation output, unknown products, stock validation, Gemini structured-output handling and failures, memory retrieval/isolation, feedback influence, and preference updates/deletion. Gemini calls are mocked; tests do not require an API key or make paid API requests.

## Publish to GitHub

This workspace currently has no commits or remote configured. Create an empty repository on GitHub, then run these commands from this project folder, replacing the URL with your repository URL:

```powershell
git branch -M main
git add .gitignore .env.example README.md app.py memory_store.py requirements.txt test_app.py
git status --short
git commit -m "Build AI-assisted Aevora fit API prototype"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Before pushing, confirm `.env`, `.venv`, and local SQLite files are absent from `git status`. Never force-add secrets or the local memory database.
