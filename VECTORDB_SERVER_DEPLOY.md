# VectorDB — Server Deployment

Only the SEC ingestion runs on the server. All HMM/Monte Carlo/LLM tasks stay on the AI PC.

## 1. Copy the folder
```bash
rsync -av --exclude='venv/' --exclude='*.db' --exclude='chroma_db/' \
  deploy_on_ai-pc/ user@your-server:/opt/financial/deploy_on_ai-pc/
```

## 2. Create venv & install deps
```bash
cd /opt/financial/deploy_on_ai-pc
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
> `sentence-transformers` will download `all-MiniLM-L6-v2` (~90MB) on first run.

## 3. Configure `.env`
```bash
FINANCIAL_DB_PATH=/opt/financial/deploy_on_ai-pc/financial_data.db
CHROMA_DB_PATH=/opt/financial/chroma_db
EDGAR_USER_AGENT=YourName your@email.com   # required by SEC
WEBSERVER_URL=http://localhost:9875
OLLAMA_URL=http://your-ai-pc.tailnet.ts.net:11434
```

## 4. Initialise ChromaDB (once)
```bash
python vectordb/setup.py
```

## 5. Test ingest
```bash
python tasks/sec_ingestion_task.py --tickers AAPL,MSFT,NVDA --forms 10-K
```

## 6. Add cron (weekly, Monday 06:00)
```bash
crontab -e
```
```
0 6 * * 1 /opt/financial/deploy_on_ai-pc/venv/bin/python \
  /opt/financial/deploy_on_ai-pc/tasks/sec_ingestion_task.py \
  >> /var/log/sec_ingest.log 2>&1
```

## Sanity check
Verify the server DB has tickers before the first full run:
```bash
sqlite3 financial_data.db "SELECT COUNT(*) FROM dim_company WHERE ticker NOT LIKE '%.HK';"
```
