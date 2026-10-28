import os
import sys

# Import the ``hi3d`` package directly; the repository root is a ComfyUI node
# package whose __init__ needs ComfyUI.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
