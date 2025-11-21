# Docker Setup Guide

This project uses Docker Compose to orchestrate three services: MongoDB, Flask backend, and React frontend.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose (usually included with Docker Desktop)

## Quick Start

### Standard Mode (without watch)

1. **Build and start all services:**
   ```bash
   docker compose up --build
   ```

2. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000
   - MongoDB: localhost:27017

3. **Stop all services:**
   ```bash
   docker compose down
   ```

### Development Mode with Watch (Recommended)

Use `docker compose watch` for automatic file syncing and rebuilding:

```bash
docker compose watch
```

**Benefits:**
- Automatically syncs file changes to containers
- Rebuilds containers when dependencies change (requirements.txt, package.json)
- Flask auto-reloads on Python file changes (with FLASK_DEBUG=1)
- React hot-reloads on source file changes
- No need to manually rebuild containers

**Note:** Requires Docker Compose v2.22+ (included in Docker Desktop 4.15+)

## Services

### MongoDB
- **Container name**: `mongodb`
- **Port**: 27017
- **Data persistence**: Named volume `mongodb_data`
- **Database**: `auth_app`

### Backend (Flask)
- **Container name**: `backend`
- **Port**: 5000
- **Environment variables**:
  - `MONGO_URI`: MongoDB connection string (set automatically)
  - `FLASK_SECRET_KEY`: Secret key for Flask sessions (default: `your-secret-key-change-in-production`)
  - `FLASK_DEBUG`: Debug mode (default: 1)
  - `FRONTEND_ORIGIN`: Frontend origin for CORS (default: `http://localhost:3000`)
- **Public Datasets**: On startup, the backend automatically seeds public datasets from `backend/uploads/`:
  - `casehold/casehold.csv`
  - `cqa_data.csv`
  - `deceptive-opinion.csv`
  - `imdb.csv`
  - `maveriq/bigbenchhard.csv`
  - `qiaojin/PubMedQA.csv`
  
  These datasets are flagged as `is_public: true` and are available to all users.

### Frontend (React)
- **Container name**: `frontend`
- **Port**: 3000
- **Environment variables**:
  - `REACT_APP_API_URL`: Backend API URL (default: `http://localhost:5000`)
  - `CHOKIDAR_USEPOLLING`: Enable file watching in Docker (set to `true`)

## Environment Variables

You can customize environment variables by:

1. **Creating a `.env` file** in the project root:
   ```env
   FLASK_SECRET_KEY=your-secret-key-here
   REACT_APP_API_URL=http://localhost:5000
   ```

2. **Modifying `docker-compose.yml`** directly

## Volumes

- **MongoDB data**: Persisted in Docker volume `mongodb_data`
- **Backend uploads**: Mounted from `./backend/uploads` (bind mount)
- **Frontend source**: Mounted from `./explainable-nlp/src` for hot reload in development

## Development vs Production

The current setup runs in **development mode**:
- Frontend uses `npm start` with hot reload
- Backend runs with `FLASK_DEBUG=1`
- Source code is mounted as volumes for live editing

For production, you would need to:
- Build the frontend (`npm run build`) and serve static files
- Set `FLASK_DEBUG=0`
- Use a production WSGI server (e.g., gunicorn)

## Troubleshooting

### MongoDB connection issues
- Ensure MongoDB container is healthy: `docker compose ps`
- Check logs: `docker compose logs mongodb` or `docker logs mongodb`

### Frontend can't connect to backend
- Verify backend is running: `docker compose logs backend` or `docker logs backend`
- Check CORS settings in `backend/app.py`
- Ensure `REACT_APP_API_URL` is set correctly

### Port conflicts
If ports 3000, 5000, or 27017 are already in use:
- Stop conflicting services, or
- Modify port mappings in `docker-compose.yml`

## Viewing Logs

### Flask Backend Logs (HTTP Requests & Server Output)

To see Flask HTTP requests, responses, and server output in real-time:

```bash
# View backend logs (follow mode - shows new logs as they appear)
docker compose logs -f backend

# Or using docker directly (also works)
docker logs -f backend

# View last 100 lines of backend logs
docker compose logs --tail=100 backend

# View logs with timestamps
docker compose logs -f --timestamps backend
```

**What you'll see:**
- HTTP request logs (method, path, status codes)
- Flask debug output
- Python print statements and errors
- Database queries (if enabled)
- Any application logs

### All Services Logs

```bash
# View logs from all services
docker compose logs -f

# View logs from specific services
docker compose logs -f backend frontend
```

### In Watch Mode

When running `docker compose watch`, logs are automatically displayed in the terminal. To view logs separately in another terminal:

```bash
# In a new terminal window (use docker compose or docker)
docker compose logs -f backend
# OR
docker logs -f backend
```

## Useful Commands

```bash
# Start with watch mode (recommended for development)
docker compose watch

# Standard start
docker compose up --build

# View logs (use either command - both work)
docker compose logs -f backend
docker logs -f backend

# View logs for all services
docker compose logs -f

# Rebuild specific service
docker compose build backend

# Stop and remove containers (keeps volumes)
docker compose down

# Stop and remove containers and volumes
docker compose down -v

# Execute command in running container
docker compose exec backend python --version
docker compose exec frontend npm --version

# Access MongoDB shell
docker compose exec mongodb mongosh
```

**Note:** Modern Docker Desktop uses `docker compose` (without hyphen) as a CLI plugin. The older `docker-compose` (with hyphen) standalone binary may not be installed. Both `docker compose` and `docker` commands work for viewing logs.

## Watch Configuration

The `docker-compose.yml` includes watch configuration that:

- **Backend**: Rebuilds when `requirements.txt` changes
- **Frontend**: Rebuilds when `package.json` changes

**Note:** File syncing is handled automatically by bind mount volumes (`./backend:/app` and `./explainable-nlp/src:/app/src`), so we only use watch for rebuilds when dependencies change. This eliminates warnings about duplicate path monitoring.

Watch actions:
- `rebuild`: Rebuilds the container image when dependency files change (requirements.txt, package.json)

