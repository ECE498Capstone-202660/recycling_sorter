# FastAPI Server – Quick Start

## 1. Create & Activate Virtual Environment
```bash
cd server
python3 -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

## 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 3. Start the FastAPI Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

The API will be available at **http://localhost:8080**.
