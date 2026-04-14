import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Assuming you will set these in a local .env later on the AI pc
# For Tailscale, WEBSERVER_URL would be something like http://webserver.machine-name.ts.net:8000
WEBSERVER_URL = os.getenv("WEBSERVER_URL", "http://localhost:8000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

# Tasks Scheduling config
WORK_HOUR = int(os.getenv("WORK_HOUR", 8))
WORK_MINUTE_MARKOV = int(os.getenv("WORK_MINUTE_MARKOV", 5))
WORK_MINUTE_MC = int(os.getenv("WORK_MINUTE_MC", 30))
WORK_MINUTE_LLM = int(os.getenv("WORK_MINUTE_LLM", 0))
WORK_MINUTE_TRIGGER = int(os.getenv("WORK_MINUTE_TRIGGER", 30))
