"""Real-time MPFB ECG beat classification (N / S / V) for the Jetson Nano."""
from .model import CLASSES, MPFBModel
from .pipeline import EcgPipeline, OraclePeaks
