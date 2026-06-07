# LET - LLM Explanation Tool

**LET (LLM Explanation Tool)** is a comprehensive web-based platform for generating, evaluating, and comparing natural language explanations from large language models (LLMs). Built for researchers and practitioners in explainable AI, LET addresses the growing need to understand and assess the quality of AI-generated explanations across multiple dimensions.

**Please take a look at the Prototype documentation for detailed, extensive explanations on how to use and understand every function of the tool.**

---

## Overview

While most existing explainability frameworks focus on feature attribution methods (e.g., LIME, SHAP), LET emphasizes **self-explanations** and **post-hoc explanations** expressed in natural language. This reflects the growing importance of LLMs in human-AI interaction and the need for explanations that are both **faithful** (accurately reflecting model reasoning) and **plausible** (convincing to human users).

LET enables:
- Multi-provider LLM integration (OpenAI, Gemini, DeepSeek, Groq, Ollama)
- Traditional transformer classifiers (BERT) with SHAP-based explanations
- Systematic evaluation of explanation quality using the LExT framework
- Interactive and batch processing of benchmark datasets
- Side-by-side comparison of explanation types and providers

---

## Key Features

✅ **Provider-agnostic design**: Automatically supports new models from connected providers  
✅ **Dual explanation modes**: Self-explanations and post-hoc explanations  
✅ **Traditional baselines**: BERT + SHAP for comparison  
✅ **Rigorous evaluation**: LExT framework for faithfulness and plausibility  
✅ **Flexible datasets**: Built-in benchmarks + custom upload support  
✅ **Chain-of-Thought prompting**: Elicit step-by-step reasoning  
✅ **Interactive exploration**: Both batch processing and instance-level analysis  
✅ **User rating system**: Collect human feedback on explanation quality  
✅ **Privacy-preserving option**: Local deployment with Ollama  

---

## Installation and Setup

### 🐳 Docker Installation (Recommended)

The easiest way to run the application is using Docker Compose. This method automatically sets up MongoDB, backend, and frontend with a single command.

#### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

#### Quick Start

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd thesisXNLP
   ```

2. **Start all services**:
   ```bash
   docker compose watch
   ```
   
   Or for standard mode:
   ```bash
   docker compose up --build
   ```

3. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000
   - MongoDB: localhost:27017

4. **Stop services**:
   ```bash
   docker compose down
   ```

#### What Docker Sets Up

- **MongoDB**: Automatically started in a container with persistent data storage
- **Backend**: Flask application with all dependencies installed
- **Frontend**: React development server with hot reload
- **Public Datasets**: Automatically seeded on backend startup (casehold.csv, imdb.csv, etc.)

#### Development Mode with Watch

Using `docker compose watch` provides:
- ✅ Automatic file syncing (no rebuild needed for code changes)
- ✅ Flask auto-reload on Python file changes
- ✅ React hot-reload on frontend changes
- ✅ Automatic rebuilds when dependencies change

#### Viewing Logs

```bash
# View all logs
docker compose logs -f

# View backend logs (Flask HTTP requests)
docker logs -f backend

# View frontend logs
docker logs -f frontend
```

#### Environment Variables (Optional)

Create a `.env` file in the project root to customize settings:
```env
FLASK_SECRET_KEY=your-secure-secret-key-here
```

For more Docker details, see [DOCKER.md](DOCKER.md).

---

### Manual Installation (Alternative)

If you prefer to run services manually without Docker:

#### Prerequisites
- Python 3.10+
- Node.js 16+
- MongoDB 4.4+

### Backend Setup

#### Install Conda

If you don't have Conda installed:

**macOS / Linux:**
```bash
# Download Miniconda (recommended) or Anaconda
# Miniconda: https://docs.conda.io/en/latest/miniconda.html
# Anaconda: https://www.anaconda.com/products/distribution

# After installation, restart your terminal or run:
source ~/.bashrc  # or source ~/.zshrc for zsh
```

**Windows:**
- Download and install Miniconda or Anaconda from:
  - Miniconda: https://docs.conda.io/en/latest/miniconda.html
  - Anaconda: https://www.anaconda.com/products/distribution
- Use Anaconda Prompt for terminal commands

#### Create Environment and Setup

```bash
conda env create -f environment.yml        # Run once to create the environment
conda activate let                         # Run in every new shell before using the backend

cd backend
cp .env.example .env                       # On Windows use: copy .env.example .env
# Edit .env with your MongoDB URI and other settings
```

### MongoDB Setup

#### Local MongoDB (development)

**macOS**
1. Install MongoDB Community Edition:  
   ```bash
   brew tap mongodb/brew
   brew install mongodb-community@7.0
   ```
2. Start the service:  
   ```bash
   brew services start mongodb/brew/mongodb-community
   ```
3. Verify it is running:  
   ```bash
   mongosh
   ```

**Ubuntu / Debian**
1. Import the MongoDB public key and add the repository (example for 7.0):  
   ```bash
   curl -fsSL https://pgp.mongodb.com/server-7.0.asc | \
     sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
   echo "deb [signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg] \
     https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | \
     sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
   sudo apt-get update
   sudo apt-get install -y mongodb-org
   ```
2. Start and enable the service:  
   ```bash
   sudo systemctl start mongod
   sudo systemctl enable mongod
   ```
3. Check status:  
   ```bash
   sudo systemctl status mongod
   ```

**Windows**
1. Download the MSI installer from <https://www.mongodb.com/try/download/community> (choose the latest stable version).
2. Run the installer and select **Install MongoDB as a Service** (default settings are fine).
3. After installation, open **Command Prompt** and run:  
   ```cmd
   mongosh
   ```
   If it opens the shell, the server is running. If not, start the service via **Services** → **MongoDB Server** → **Start**.

**Optional seeding** (all platforms)
- Import seed data with `mongorestore --uri "<MONGO_URI>" dump/` or execute scripts via `mongosh`.

### Environment Variables
Create or update `backend/.env` so Flask points to the correct database:

```
FLASK_SECRET_KEY=change-me
MONGO_URI=mongodb://localhost:27017/auth_app           # local development
SESSION_COOKIE_NAME=let_session
UPLOAD_FOLDER=uploads
```

Flask initializes the MongoDB client through `mongo.init_app(app)` on startup, so as long as the URI is reachable the database and collections will be created automatically on first write. Make sure the MongoDB service is running before launching Flask.

### Frontend Setup
```bash
conda activate let
cd explainable-nlp
npm install
```

### API Keys
Configure provider API keys either:
1. During registration via the Settings panel
2. In the Settings page after login

At least one provider key is required to run classifications or generate explanations.

### Local Model Support (Optional)
To enable Ollama for local model deployment:
```bash
# Install Ollama (see https://ollama.ai)
ollama pull llama2  # Or any other model

# Ensure sufficient GPU memory for your chosen model
```

**Note**: Ollama is only available in local deployments, not hosted versions.

### Running the App

#### Using Docker (Recommended)
If you installed using Docker, simply run:
```bash
docker compose watch
```

This starts all services (MongoDB, backend, frontend) automatically. Access the app at http://localhost:3000.

#### Manual Setup (Alternative)
If you installed manually, make sure the Conda environment is active (`conda activate let`) and run the appropriate script from the project root:

- **macOS / Linux**  
  ```bash
  ./start.sh
  ```

- **Windows (Anaconda Prompt / PowerShell)**  
  ```cmd
  start_windows.bat
  ```
  The script opens a new terminal window for the frontend (`npm start`) and keeps the backend (`python app.py`) in the current window. Stop the backend with `Ctrl+C` and close the frontend window when you are done.

---

## Result Scripts and Analysis for the User Study

The `result scripts/` directory contains a comprehensive analysis pipeline for processing experimental data and generating statistical results. This includes:

- **Data processing**: Combines raw data from Prolific and friends/family sources
- **Hypothesis testing**: 16 hypotheses with cluster-robust standard errors
- **Demographic analyses**: Age, gender, CS/AI expertise, education level, and NLP experience
- **Visualization**: Automated generation of plots and statistical summaries

**Quick Start:**
```bash
cd "result scripts"
pip install -r requirements.txt
python main.py  # Run all analyses
```

For detailed documentation, usage examples, and workflow instructions, see **[result scripts/README.md](result%20scripts/README.md)**.

---

## Documentation

For comprehensive technical documentation, see **[PROTOTYPE_DOCUMENTATION.md](PROTOTYPE_DOCUMENTATION.md)** which includes:
- System architecture
- Supported models and datasets
- Evaluation methodology
- User interface design
- Complete prompt templates

For Docker-specific documentation, see **[DOCKER.md](DOCKER.md)**.

---

## Citation

If you use LET in your research, please cite:

```bibtex
@mastersthesis{eren2025let,
  author = {Yarkin Eren},
  title = {LET: LLM Explanation Tool for Evaluating Faithfulness and Plausibility},
  school = {Technical University of Munich},
  year = {2025}
}
```
