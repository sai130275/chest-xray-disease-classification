#!/usr/bin/env bash
# ==============================================================================
#  ██████╗   █████╗  ██████╗         ██╗  ██╗
#  ██╔══██╗ ██╔══██╗ ██╔══██╗        ╚██╗██╔╝
#  ██████╔╝ ███████║ ██║  ██║  █████╗ ╚███╔╝ 
#  ██╔══██╗ ██╔══██║ ██║  ██║  ╚════╝ ██╔██╗ 
#  ██║  ██║ ██║  ██║ ██████╔╝        ██╔╝ ██╗
#  ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═════╝         ╚═╝  ╚═╝
#
# RAD-X Application Supercharged Startup Script
# Starts both the FastAPI inference backend and the React frontend concurrently.
# Gracefully terminates all background processes on Ctrl+C.
# ==============================================================================

set -eo pipefail

# ANSI color codes
BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[38;5;51m'
BLUE='\033[38;5;39m'
MAGENTA='\033[38;5;201m'
PURPLE='\033[38;5;141m'
GREEN='\033[38;5;48m'
YELLOW='\033[38;5;220m'
RED='\033[38;5;196m'
NC='\033[0m' # No Color

# Determine project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Variables to store child process IDs
BACKEND_PID=""
FRONTEND_PID=""

# Flashy shutdown handler
cleanup() {
    echo ""
    echo -e "${MAGENTA}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║${YELLOW}  ⚡ [RAD-X] SYSTEM SHUTDOWN INITIATED...                             ${MAGENTA}║${NC}"
    echo -e "${MAGENTA}║${NC}     Terminating active background services cleanly                   ${MAGENTA}║${NC}"
    echo -e "${MAGENTA}╚══════════════════════════════════════════════════════════════════════╝${NC}"

    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "  ${PURPLE}▸${NC} Stopping React Frontend  (PID: ${CYAN}$FRONTEND_PID${NC})..."
        kill -SIGTERM "$FRONTEND_PID" 2>/dev/null || true
    fi

    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "  ${PURPLE}▸${NC} Stopping FastAPI Backend (PID: ${CYAN}$BACKEND_PID${NC})..."
        kill -SIGTERM "$BACKEND_PID" 2>/dev/null || true
    fi

    wait "$BACKEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true

    echo -e "\n  ${GREEN}✓ All services safely halted.${NC} ${CYAN}Have a productive day!${NC}\n"
    exit 0
}

# Trap signals for graceful exit
trap cleanup SIGINT SIGTERM EXIT

# ------------------------------------------------------------------------------
# Flashy Header Banner
# ------------------------------------------------------------------------------
clear 2>/dev/null || true
echo -e "${CYAN}"
cat << 'EOF'
  ██████╗   █████╗  ██████╗         ██╗  ██╗
  ██╔══██╗ ██╔══██╗ ██╔══██╗        ╚██╗██╔╝
  ██████╔╝ ███████║ ██║  ██║  █████╗ ╚███╔╝ 
  ██╔══██╗ ██╔══██║ ██║  ██║  ╚════╝ ██╔██╗ 
  ██║  ██║ ██║  ██║ ██████╔╝        ██╔╝ ██╗
  ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═════╝         ╚═╝  ╚═╝
EOF
echo -e "${NC}"
echo -e "${BOLD}${MAGENTA}  ✦ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✦${NC}"
echo -e "  ${BOLD}${BLUE}⚡ RAD-X Clinical Research & Deep Learning Diagnostic Platform ⚡${NC}"
echo -e "  ${DIM}DenseNet121 Multi-Label Thoracic Pathology Classification System${NC}"
echo -e "${BOLD}${MAGENTA}  ✦ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✦${NC}\n"

# ------------------------------------------------------------------------------
# 1. Pre-flight Environment Checks
# ------------------------------------------------------------------------------
echo -e "${BOLD}${CYAN}▶ [1/4] Inspecting Environment & Runtime...${NC}"

# Python virtual environment check
if [ -d "$SCRIPT_DIR/venv" ] && [ -f "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
    echo -e "  ${GREEN}✓${NC} Python Virtual Environment : ${PURPLE}venv/${NC} ($("$PYTHON_BIN" --version))"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
    echo -e "  ${YELLOW}!${NC} Fallback Python Runtime    : $PYTHON_BIN"
else
    echo -e "  ${RED}✗ Error: Python 3 not detected. Please install Python 3.10+.${NC}"
    exit 1
fi

# Node.js & npm check
if command -v npm &>/dev/null; then
    NODE_VER="$(node --version 2>/dev/null || echo 'unknown')"
    NPM_VER="$(npm --version 2>/dev/null || echo 'unknown')"
    echo -e "  ${GREEN}✓${NC} Node Engine & NPM Manager  : ${PURPLE}Node $NODE_VER${NC} / ${PURPLE}npm $NPM_VER${NC}"
else
    echo -e "  ${RED}✗ Error: npm is not installed or not in PATH. Please install Node.js.${NC}"
    exit 1
fi

# Frontend node_modules check
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo -e "  ${YELLOW}! frontend/node_modules missing. Running 'npm install'...${NC}"
    (cd "$SCRIPT_DIR/frontend" && npm install)
    echo -e "  ${GREEN}✓${NC} Frontend dependencies installed."
fi

# Model checkpoint check
MODEL_PATH="$SCRIPT_DIR/models/chest_xray_finetuned_best.pth"
if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE="$(ls -lh "$MODEL_PATH" | awk '{print $5}')"
    echo -e "  ${GREEN}✓${NC} Active DenseNet121 Weights : ${GREEN}chest_xray_finetuned_best.pth${NC} ${DIM}($MODEL_SIZE)${NC}"
else
    echo -e "  ${YELLOW}! Checkpoint missing at $MODEL_PATH${NC}"
    echo -e "    Backend will launch in heuristic fallback mode."
fi

# ------------------------------------------------------------------------------
# 2. Port Availability Checks
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${CYAN}▶ [2/4] Verifying Network Ports...${NC}"

check_port() {
    local port=$1
    local name=$2
    if command -v lsof &>/dev/null; then
        local pid
        pid=$(lsof -ti tcp:"$port" || true)
        if [ -n "$pid" ]; then
            echo -e "  ${YELLOW}⚠ Port $port ($name) already in use by PID $pid${NC}"
        else
            echo -e "  ${GREEN}✓${NC} Port $port ($name) is clear"
        fi
    fi
}

check_port 8000 "Backend API"
check_port 5173 "Vite Frontend"

# ------------------------------------------------------------------------------
# 3. Start Backend Service
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${CYAN}▶ [3/4] Launching FastAPI Inference Engine...${NC}"

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
"$PYTHON_BIN" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload > /dev/null 2>&1 &
BACKEND_PID=$!
echo -e "  ${GREEN}✓${NC} Background worker spawned (PID: ${CYAN}$BACKEND_PID${NC})"

# Wait for backend health probe
echo -n -e "  ${PURPLE}▸${NC} Probing backend health..."
for i in {1..25}; do
    if curl -s http://127.0.0.1:8000/health 2>/dev/null | grep -q "status"; then
        echo -e " ${GREEN}Online!${NC}"
        break
    fi
    echo -n "•"
    sleep 0.4
    if [ $i -eq 25 ]; then
        echo -e " ${YELLOW}(probe timed out; continuing)${NC}"
    fi
done

# ------------------------------------------------------------------------------
# 4. Start Frontend Service
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${CYAN}▶ [4/4] Launching React 19 + Vite Dashboard...${NC}"

(cd "$SCRIPT_DIR/frontend" && npm run dev > /dev/null 2>&1) &
FRONTEND_PID=$!
echo -e "  ${GREEN}✓${NC} Vite dev server spawned (PID: ${CYAN}$FRONTEND_PID${NC})"
sleep 1

# ------------------------------------------------------------------------------
# Flashy Ready Dashboard Card
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}╭──────────────────────────────────────────────────────────────────────╮${NC}"
echo -e "${BOLD}${GREEN}│${NC}   ${BOLD}${GREEN}⚡ RAD-X SERVICES ACTIVE & READY TO USE                           ${NC}${BOLD}${GREEN}│${NC}"
echo -e "${BOLD}${GREEN}├──────────────────────────────────────────────────────────────────────┤${NC}"
echo -e "${BOLD}${GREEN}│${NC}                                                                      ${BOLD}${GREEN}│${NC}"
echo -e "${BOLD}${GREEN}│${NC}   ${BOLD}✦ Web Dashboard   :${NC}  ${CYAN}http://localhost:5173${NC}                         ${BOLD}${GREEN}│${NC}"
echo -e "${BOLD}${GREEN}│${NC}   ${BOLD}✦ API Service     :${NC}  ${CYAN}http://127.0.0.1:8000${NC}                         ${BOLD}${GREEN}│${NC}"
echo -e "${BOLD}${GREEN}│${NC}   ${BOLD}✦ Swagger Docs    :${NC}  ${CYAN}http://127.0.0.1:8000/docs${NC}                    ${BOLD}${GREEN}│${NC}"
echo -e "${BOLD}${GREEN}│${NC}   ${BOLD}✦ Health Endpoint :${NC}  ${CYAN}http://127.0.0.1:8000/health${NC}                  ${BOLD}${GREEN}│${NC}"
echo -e "${BOLD}${GREEN}│${NC}                                                                      ${BOLD}${GREEN}│${NC}"
echo -e "${BOLD}${GREEN}├──────────────────────────────────────────────────────────────────────┤${NC}"
echo -e "${BOLD}${GREEN}│${NC}   ${YELLOW}⚡ Press Ctrl+C anytime to terminate all services cleanly.         ${NC}${BOLD}${GREEN}│${NC}"
echo -e "${BOLD}${GREEN}╰──────────────────────────────────────────────────────────────────────╯${NC}\n"

# Keep the script running and wait for background processes
wait "$BACKEND_PID" "$FRONTEND_PID"
