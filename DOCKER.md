# Docker Setup Guide

This project uses Docker Compose to orchestrate three services: MongoDB, Flask backend, and React frontend.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose (usually included with Docker Desktop)

**Note:** Modern Docker uses `docker compose` (without hyphen). If that doesn't work, use `docker-compose` (with hyphen) instead.

## Development Mode

### Watch Mode (Recommended)

```bash
docker compose watch
# OR
docker-compose watch
```

Automatically syncs file changes, rebuilds on dependency changes, and enables hot reload.

### Standard Development Mode

```bash
docker compose up --build
# OR
docker-compose up --build
```

**Access:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- MongoDB: localhost:27017

## Production Mode

### Quick Start

**Build and start:**
   ```bash
   docker compose -f docker-compose.prod.yml build
   docker compose -f docker-compose.prod.yml up -d
   # OR
   docker-compose -f docker-compose.prod.yml build
   docker-compose -f docker-compose.prod.yml up -d
   ```

**Access:**
- Frontend: http://localhost:80
- Backend API: http://localhost:5000 (direct) or via frontend at http://localhost/api

**Note:** All environment variables have defaults, so `.env` file is optional. Create one only if you want to customize values (recommended for production security).

### Production Features

- Multi-stage builds for smaller images
- Pre-downloaded ML models (BERT, DistilBERT)
- Lazy model loading (loads on first use)
- Gunicorn WSGI server
- Nginx serving React build and proxying API requests
- Resource limits and health checks
- Code baked into images (no source volumes)

## Services

### MongoDB
- **Dev container**: `mongodb`
- **Prod container**: `mongodb-prod`
- **Port**: 27017
- **Database**: `auth_app`

### Backend (Flask)
- **Dev container**: `backend`
- **Prod container**: `backend-prod`
- **Port**: 5000
- **Dev**: Flask dev server with hot reload
- **Prod**: Gunicorn WSGI server

### Frontend (React)
- **Dev container**: `frontend`
- **Prod container**: `frontend-prod`
- **Dev port**: 3000
- **Prod port**: 80 (Nginx)

## Useful Commands

### Development

```bash
# Start with watch
docker compose watch

# Standard start
docker compose up --build

# View logs
docker compose logs -f backend

# Stop services
docker compose down

# Execute command in container
docker compose exec backend python --version
docker compose exec mongodb mongosh
```

### Production

```bash
# Build images
docker compose -f docker-compose.prod.yml build

# Start services
docker compose -f docker-compose.prod.yml up -d

# View logs
docker compose -f docker-compose.prod.yml logs -f

# Stop services
docker compose -f docker-compose.prod.yml down

# Rebuild and restart
docker compose -f docker-compose.prod.yml up -d --build

# Check service status
docker compose -f docker-compose.prod.yml ps
```

**Note:** Replace `docker compose` with `docker-compose` if you have an older Docker installation.

## Troubleshooting

### MongoDB connection issues
```bash
docker compose ps
docker compose logs mongodb
```

### Frontend can't connect to backend
- **Dev**: Check CORS settings in `backend/app.py`
- **Prod**: Ensure nginx is proxying `/api` requests correctly

### Port conflicts
Stop conflicting services or modify port mappings in docker-compose files.

### Production build fails
- Check Docker has enough resources
- Verify internet connection for model downloads
- Check logs: `docker compose -f docker-compose.prod.yml logs`

## Environment Variables (Optional)

All environment variables have defaults, so `.env` file is optional. Create one only to customize values.

**Development defaults:**
- `FLASK_SECRET_KEY`: `your-secret-key-change-in-production`
- `REACT_APP_API_URL`: `http://localhost:5000`
- `FRONTEND_ORIGIN`: `http://localhost:3000`

**Production defaults:**
- `FLASK_SECRET_KEY`: `your-secret-key-change-in-production`
- `SECRET_KEY`: Same as `FLASK_SECRET_KEY`
- `FRONTEND_ORIGIN`: `http://localhost:80`
- `REACT_APP_API_URL`: `/api`

**Note:** For production, it's recommended to set strong secret keys in `.env` file for security.
