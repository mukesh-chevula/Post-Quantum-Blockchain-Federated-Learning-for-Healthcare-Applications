#!/bin/bash

# PQBFL Prototype - Quick Start Script
# This script sets up the environment and runs the prototype

set -e  # Exit on error

echo "======================================================================"
echo "PQBFL Prototype - Post-Quantum Blockchain Federated Learning"
echo "======================================================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
echo "Installing dependencies..."
echo "  (This may take a few minutes on first run)"
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"

echo ""
echo "======================================================================"
echo "Running Component Tests"
echo "======================================================================"
echo ""

# Run component tests
python test_components.py

echo ""
echo "======================================================================"
echo "Running Main Prototype"
echo "======================================================================"
echo ""

# Run main prototype
python main.py --n_clients 3 --rounds 5 --ratchet_threshold 3

echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "You can now:"
echo "  • Run main.py with custom parameters"
echo "  • Modify config.yaml for different settings"
echo "  • Check results/ directory for outputs"
echo ""
echo "Example commands:"
echo "  python main.py --n_clients 5 --rounds 10"
echo "  python main.py --dataset healthcare --condition \"Diabetes\""
echo "  python test_components.py  # Re-run tests"
echo ""
echo "======================================================================"
