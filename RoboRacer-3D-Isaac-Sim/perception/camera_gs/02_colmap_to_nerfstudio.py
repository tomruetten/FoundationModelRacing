"""
Convert a COLMAP sparse reconstruction (sparse/0) into a nerfstudio
transforms.json + sparse_pc.ply, using nerfstudio's own colmap_to_json
helper (same conversion ns-process-data uses internally).

Usage:
    conda run -n nerfstudio-py310 python3 colmap_to_transforms.py <recon_dir> <output_dir>
"""
import sys
from pathlib import Path
from nerfstudio.process_data.colmap_utils import colmap_to_json

recon_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)

n = colmap_to_json(recon_dir=recon_dir, output_dir=output_dir, ply_filename="sparse_pc.ply")
print(f"registered {n} images, wrote transforms.json to {output_dir}")
