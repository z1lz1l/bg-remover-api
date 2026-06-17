"""
Configuration module for the BG Remover service.

The FAL.AI API key is embedded here as a default. It can be overridden
by setting the FAL_KEY environment variable before starting the server.
"""
import os

# FAL.AI API key. Override with the FAL_KEY environment variable if needed.
FAL_KEY: str = os.environ.get(
    "FAL_KEY",
    "709a2045-58b5-4d97-9477-635d0a018386:5ac3a696de187881e98ffc02b0e06794",
).strip()
