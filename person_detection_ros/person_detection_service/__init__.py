import os
import sys

# Automatically append the torchreid path relative to this file
reid_path = os.path.join(os.path.dirname(__file__), 'submodules', 'keypoint_promptable_reidentification')
if reid_path not in sys.path:
    sys.path.insert(0, reid_path)
