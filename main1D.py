import numpy as np
import pyvista as pv
from SSOM1D import *
import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def get_color_map(class_count):
    # 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
    #                   'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
    #                   'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
    #                   'turbo', 'nipy_spectral', 'gist_ncar']
    cmap = plt.get_cmap('gist_rainbow')
    colors = []
    i = 0
    while len(colors) < class_count:
        c = cmap(i / class_count)[:3]  # Use n+1 to avoid endpoints
        if not np.allclose(c, (0, 0, 0)):  # Exclude black
            colors.append(mcolors.to_hex(c))
        i += 1
    return colors
def plot(plotter, mesh, scalar_name, class_count):
    # 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
    #                   'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
    #                   'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
    #                   'turbo', 'nipy_spectral', 'gist_ncar']
    assert(class_count > 0)
    colors = get_color_map(class_count)
    if(class_count <20):
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": class_count})
    else:
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": 20})


def main(normal_weight, lr, radius, obj_file, min_region_rate, threshold_similarity_merge):
    # print("Running segmentation with the following parameters:")
    # print(f"Learning Rate: {lr}")
    # print(f"Radius: {radius}")
    # print(f"Object File: {obj_file}")
    # print(f"Min Region Face Count: {min_region_face_count}")
    # print(f"Similarity Merge Threshold: {threshold_similarity_merge}")
    
    obj_mesh = load_obj_with_face_normals(obj_file)
    face_adjacency = build_face_adjacency(obj_mesh)

    sdf = []
    with open(obj_file.replace('.obj', '.sdf'), 'r') as f: #suppose there is *.sdf file in the same folder
        for line in f:
            face_id, value = line.strip().split(',')
            sdf.append(float(value))
    sdf = np.array(sdf).reshape(-1, 1)  # Convert to 2D array with shape (n_samples, 1)
    obj_mesh.cell_data['sdf'] = sdf

    # # Load mesh and assign 6D features, train SOM
    #data_for_som = np.concatenate((translated_xyz*(1-normal_weight), obj_mesh.cell_data['Normals']*normal_weight), axis=1)
    data_for_som = sdf
    spherical_mesh = pv.read("regular_sphere.obj")
    som = SphereSOM(spherical_mesh, lr=lr, radius=radius)
    before = som.weights.copy()
    som.train(data_for_som, n_epochs=2000, n_rings=2)
    devi = (som.weights - before).T
    # Set numpy print options for 4 decimal places
    np.set_printoptions(precision=4, floatmode='fixed')
    print("Beforetraining", before.T)
    print("After training", som.weights.T)    

    #Predict labels
    raw_labels = som.predict(data_for_som)
    # Get the weights corresponding to each raw label
    label_weights = som.weights[raw_labels]*100
    obj_mesh.cell_data["label_weights"] = label_weights # Assign cluster labels to each face  
    print("Label weights:", label_weights)

    raw_labels, raw_labels_count = remap_labels(raw_labels)  # Convert to face labels 0-based indices
    print("SOM clustering: there are {} clusters".format(raw_labels_count))
    obj_mesh.cell_data["raw_labels"] = label_weights # Assign cluster labels to each face  
    
    #Separate disconnected components
    separated_region_labels = separate_disconnected_components(obj_mesh, face_adjacency, raw_labels)
    obj_mesh.cell_data["separated_region_labels"] = separated_region_labels 
    
    #Merge small regions to their biggest neighbor
    region_labels = merge_small_regions(obj_mesh, separated_region_labels, face_adjacency, min_region_rate)
    merged_small_region_labels, merged_region_labels_count = remap_labels(region_labels)  # Convert to face labels 0-based indices
    obj_mesh.cell_data["merged_small_region_labels"] = merged_small_region_labels


    # plotter = pv.Plotter()
    # obj_mesh10 = obj_mesh.copy()
    # plot(plotter,obj_mesh10, "merged_small_region_labels", merged_region_labels_count)  
    # # Create a copy of the mesh for displaying sparse labels
    
    # plotter.show()
    
    # Merge similar regions based on sdf
    merged_region_label_temp = merged_small_region_labels.copy()
    #for i in range(1):
    merged_region_label_temp, face_avg_values = merge_similar_direction_regions(obj_mesh, merged_region_label_temp, face_adjacency, threshold_similarity_merge)
    merged_similar_region_labels, merged_similar_region_labels_count = remap_labels(merged_region_label_temp)  # Convert to face labels 0-based indices
    obj_mesh.cell_data["merged_similar_region_labels"] = merged_similar_region_labels

    # Create a 2x2 grid plotter
    plotter = pv.Plotter(shape=(2, 2))

    plotter.subplot(0, 0)
    plot(plotter, obj_mesh, "raw_labels", raw_labels_count)

    plotter.subplot(0, 1)
    # som_mesh = som.get_mesh()
    # plot(plotter, som_mesh, raw_labels_count)
    obj_mesh01 = obj_mesh.copy()
    plot(plotter,obj_mesh01, "separated_region_labels", merged_region_labels_count)
    
    plotter.subplot(1, 0)
    obj_mesh10 = obj_mesh.copy()
    plot(plotter,obj_mesh10, "merged_small_region_labels", merged_region_labels_count)
    
    sparse_mesh = obj_mesh.copy()    
    # Randomly select 20% of faces to display labels
    n_faces = sparse_mesh.n_faces
    n_display = int(n_faces * 0.01)
    display_indices = np.random.choice(n_faces, n_display, replace=False)    
    # Create a mask for faces to display labels
    display_mask = np.zeros(n_faces, dtype=bool)
    display_mask[display_indices] = True    
    # Get face centers for label placement
    face_centers = sparse_mesh.cell_centers().points    
    # Create points for labels
    label_points = face_centers[display_mask]
    label_values = merged_small_region_labels[display_mask]    
    # Create point cloud for labels
    label_cloud = pv.PolyData(label_points)    
    # Add labels to the plotter
    plotter.add_point_labels(
        label_points,
        label_values,
        always_visible=True,
        text_color='black',
        font_size=12,
        shape_color='white',
        shape_opacity=0.8,
        show_points=False
    )

    
    plotter.subplot(1, 1)
    obj_mesh11 = obj_mesh.copy()
    plot(plotter, obj_mesh11, "merged_similar_region_labels", merged_similar_region_labels_count)
    
    plotter.show()
    # plotter = pv.Plotter()
    # mesh_som = som.get_mesh() #.plot(show_edges=True)
    # plotter.add_mesh(mesh_som, show_edges=True, color="white", opacity=0.5)
    
    # #plotter.add_scalar_bar(title="Cluster ID", n_labels=len(unique_ids), vertical=False)
    # highlight_points = mesh_som.points[raw_labels]
    # highlight_cloud = pv.PolyData(highlight_points)
    # glyphs = highlight_cloud.glyph(scale=True, geom=pv.Sphere(radius=0.02))
    # colors = get_color_map(raw_labels_count)


    # colors_glyphs = np.array([colors[cid] for cid in raw_labels], dtype=np.uint8)
    # glyphs.point_data["colors"] = np.repeat(colors_glyphs, glyphs.n_points // len(colors_glyphs), axis=0)
    # plotter.add_mesh(glyphs, scalars="colors", cmap = get_color_map(raw_labels_count), rgb=True, render_points_as_spheres=True, point_size=30)

    # plotter.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spherical SOM surface segmentation.")
    parser.add_argument("--normal_weight", type=float, default = 1, help="Weight for normal direction")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate for SOM training")
    parser.add_argument("--radius", type=float, default=0.3, help="Neighborhood radius for SOM")
    parser.add_argument("--obj_file", type=str, required=True, default='../FastSDF/181.obj', help="Path to the OBJ file")
    parser.add_argument("--min_region_face_count", type=float, default=0.01, help="Minimum proportion of faces per region")
    parser.add_argument("--threshold_similarity_merge", type=float, default=0.6, help="Similarity threshold for region merging")

    args = parser.parse_args()
    main(args.normal_weight, args.lr, args.radius, args.obj_file, args.min_region_face_count, args.threshold_similarity_merge)
