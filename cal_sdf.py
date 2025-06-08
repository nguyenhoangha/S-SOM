import pymeshlab
import numpy as np
import pyvista as pv
import argparse
import os

def main(obj_file):
    # Load mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_file)

    # Compute the Shape Diameter Function (SDF)
    ms.apply_filter('compute_scalar_by_shape_diameter_function_per_vertex', 
                    cone_amplitude = 150)

    # The SDF values are stored per-vertex as a scalar attribute
    sdf_values = ms.current_mesh().vertex_scalar_array()

    print("SDF values per vertex:", sdf_values)

    # Create a PyVista mesh from the original OBJ file
    mesh = pv.read(obj_file)

    # Convert vertex-based SDF values to face-based values by averaging
    face_centers = mesh.cell_centers().points
    face_sdf = np.zeros(mesh.n_faces)

    for i, cell in enumerate(mesh.faces.reshape((-1, 4))):
        vertex_ids = cell[1:]  # Skip the first number which is the number of vertices
        face_sdf[i] = np.mean(sdf_values[vertex_ids])

    # Assign SDF values to faces
    mesh.cell_data['sdf'] = face_sdf

    # Generate output filename based on input filename
    base_name = os.path.splitext(obj_file)[0]
    output_file = f"{base_name}.sdf"
    
    with open(output_file, 'w') as f:
        for face_id, sdf_value in enumerate(face_sdf):
            f.write(f"{face_id},{sdf_value}\n")

    # Create a plotter
    plotter = pv.Plotter()
    # Add the mesh to the plotter with SDF values as colors
    plotter.add_mesh(mesh, scalars='sdf', cmap='jet', show_edges=True)
    # Add a colorbar
    plotter.add_scalar_bar(title='SDF Values')
    # Show the visualization
    plotter.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate SDF for a 3D mesh file')
    parser.add_argument('--obj_file', type=str, required=True,
                      help='Path to the OBJ file to process')
    
    args = parser.parse_args()
    main(args.obj_file)