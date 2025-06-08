# 3D Surface Segmentation using Spherical SOM

This project implements an unsupervised 3D surface segmentation method based on **Spherical Self-Organizing Maps (S-SOM)**.
It clusters face normals and geometric information from a 3D mesh using a spherical topology, then refines the segmentation through post-processing steps.
Input: 3D model file (*.obj), and (*.sdf) in case using SDF feature.
 
## Features

- Load a 3D mesh (.obj) and compute face normals
- Apply Spherical SOM to segment the surface based on normals
- Post-process to:
  - Separate disconnected regions
  - Merge small regions
  - Merge similar regions based on surface orientation
- Visualize segmentation results using `pyvista`
- 
## Requirements

- Python 3.7+
- `numpy`
- `pyvista`
- `matplotlib`
- scikit-learn

Install dependencies via bash
pip install -r requirements.txt

## Run Facet segmentation using normal vector as feature descriptor
python .\main.py --obj_file='./Models/brick_part01.obj'

## Run Part segmentation using precomputed SDF (in a *.sdf file with same name and in same folder with *.obj) and curvature
### Precompute the sdf if need
python cal_sdf.py --obj_file='./Models/181.obj'
### Run segmentation
python .\main2D.py --obj_file='./Models/181.obj'
