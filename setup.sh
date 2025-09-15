#!/bin/bash

# Setup script for Spend Data Management Platform

echo "🚀 Setting up Spend Data Management Platform..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv is installed"

# Install dependencies
echo "📦 Installing dependencies..."
uv sync

# Ensure a local virtualenv exists; create it if missing, then activate
if [ ! -d ".venv" ]; then
    echo "No .venv found — creating virtual environment..."
    python3 -m venv .venv || { echo "Failed to create .venv"; exit 1; }
fi

# Activate the venv for the remainder of the script
# shellcheck disable=SC1091
source .venv/bin/activate

# Resolve configured DB path (pre-flight) and confirm before destructive ops
DB_PATH=$(python3 - <<'PY'
import sys
sys.path.insert(0, '.')
from src.config import config
print(config.database.path)
PY
)

echo "Resolved database path: ${DB_PATH}"
if [ "${SKIP_CONFIRM}" != "yes" ]; then
    read -r -p "This will initialize or overwrite data at ${DB_PATH}. Continue? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "Continuing..."
            ;;
        *)
            echo "Aborted by user. No changes made.";
            exit 0
            ;;
    esac
fi

echo "📦 Installing dependencies..."
uv sync

# Initialize database
echo "🗄️ Initializing database..."
uv run python scripts/init_db.py

# Load sample data
echo "📊 Loading sample data..."
uv run python scripts/load_sample_data.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To start the application:"
echo "   uv run streamlit run src/app.py"
echo ""
echo "Then open your browser and go to: http://localhost:8501"
echo ""
echo "Demo Credentials:"
echo "   Admin: admin / admin1234"
echo "   Spend Manager: manager / manager1234" 
echo "   Data Analyst: analyst / analyst1234"
echo ""
