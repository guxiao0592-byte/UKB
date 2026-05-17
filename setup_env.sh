#!/bin/bash
# ============================================================================
# UKB-DRP Imaging Pipeline — Environment Setup
# ============================================================================
set -euo pipefail

echo "Setting up UKB-DRP environment..."
echo ""

# Check what's already installed
MISSING=""
python3 -c "import numpy"       2>/dev/null || MISSING="$MISSING numpy"
python3 -c "import pandas"      2>/dev/null || MISSING="$MISSING pandas"
python3 -c "import scipy"       2>/dev/null || MISSING="$MISSING scipy"
python3 -c "import sklearn"     2>/dev/null || MISSING="$MISSING scikit-learn"
python3 -c "import lightgbm"    2>/dev/null || MISSING="$MISSING lightgbm"
python3 -c "import matplotlib"  2>/dev/null || MISSING="$MISSING matplotlib"

if [ -z "$MISSING" ]; then
    echo "All packages already installed."
else
    echo "Missing packages:$MISSING"
    echo ""
    echo "Installing with pip..."
    pip install $MISSING
fi

echo ""
echo "Verifying..."
python3 -c "
import numpy;     print(f'  numpy:      {numpy.__version__}')
import pandas;    print(f'  pandas:     {pandas.__version__}')
import scipy;     print(f'  scipy:      {scipy.__version__}')
import sklearn;   print(f'  sklearn:    {sklearn.__version__}')
import lightgbm;  print(f'  lightgbm:   {lightgbm.__version__}')
import matplotlib;print(f'  matplotlib: {matplotlib.__version__}')
"

echo ""
echo "Done. Now run: bash run_imaging_pipeline.sh"
