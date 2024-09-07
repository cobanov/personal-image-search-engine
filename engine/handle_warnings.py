# handle_warnings.py

import warnings

# Suppress specific warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Torch was not compiled with flash attention",
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*weights_only=False.*"
)

# Add more warning filters as needed
