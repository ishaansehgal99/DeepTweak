#!/bin/bash

# Create a virtual environment (optional)
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
else
  echo "Using existing virtual environment..."
  source venv/bin/activate
fi

# Run the visualization script
echo "Running visualization script..."
python visualize.py

# Display the results
echo ""
echo "Visualization complete!"
echo "Results are available in:"
echo "  - plots/ directory (visualizations)"
echo "  - results/ directory (summary data)"

# Open the summary CSV file if possible
if command -v open &> /dev/null; then
  if [ -f "results/experiment_summary.csv" ]; then
    echo ""
    echo "Opening experiment summary..."
    open results/experiment_summary.csv
  fi
fi 