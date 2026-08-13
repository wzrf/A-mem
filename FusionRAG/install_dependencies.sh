#!/bin/bash

# Installation script for ktransformers dependencies
# This script installs required packages using conda and pip

set -e  # Exit on error

echo "========================================"
echo "Installing ktransformers dependencies"
echo "========================================"

# Check if conda is available
if ! command -v conda &> /dev/null
then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install conda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Install faiss using conda
echo ""
echo "Installing faiss-cpu via conda..."
conda install -c pytorch faiss-cpu=1.13.1 -y

# Install pip packages
echo ""
echo "Installing pip packages..."
pip install rouge
pip install FlagEmbedding
pip install transformers==4.53.3

echo ""
echo "========================================"
echo "Installation completed successfully!"
echo "========================================"
echo ""
echo "Installed packages:"
echo "  - faiss-cpu=1.13.1 (via conda)"
echo "  - rouge (via pip)"
echo "  - FlagEmbedding (via pip)"
echo "  - transformers==4.53.3 (via pip)"
echo ""
