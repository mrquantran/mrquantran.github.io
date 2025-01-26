#!/bin/bash
set -euo pipefail

echo "Entry point script running"

CONFIG_FILE=_config.yml
DEFAULT_PORT=4000
ALTERNATE_PORT=4001

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        echo "Port $port is in use, killing existing process..."
        lsof -ti :$port | xargs kill -9
        sleep 1
    fi
}

# Function to manage Gemfile.lock
manage_gemfile_lock() {
    git config --global --add safe.directory '*'
    if command -v git &> /dev/null && [ -f Gemfile.lock ]; then
        if git ls-files --error-unmatch Gemfile.lock &> /dev/null; then
            echo "Gemfile.lock is tracked by git, keeping it intact"
            git restore Gemfile.lock 2>/dev/null || true
        else
            echo "Gemfile.lock is not tracked by git, removing it"
            rm -f Gemfile.lock
        fi
    fi
}

# Check and kill processes on both potential ports
check_port $DEFAULT_PORT
check_port $ALTERNATE_PORT

# Rest of your script...
manage_gemfile_lock

# Start Jekyll with fallback port
if ! bundle exec jekyll serve --port $DEFAULT_PORT --livereload; then
    echo "Trying alternate port..."
    bundle exec jekyll serve --port $ALTERNATE_PORT --livereload
fi