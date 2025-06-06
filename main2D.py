import numpy as np
import pyvista as pv
from SSOM2D import *
import argparse
import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import io


def get_color_map(class_count):
    # 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
    #                   'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
    #                   'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
    #                   'turbo', 'nipy_spectral', 'gist_ncar']
    cmap = plt.get_cmap('jet')
    colors = []
    i = 0
    while len(colors) < class_count:
        c = cmap(i / class_count)[:3]  # Use n+1 to avoid endpoints
        if not np.allclose(c, (0, 0, 0)):  # Exclude black
            colors.append(mcolors.to_hex(c))
        i += 1
    return colors
def plot(plotter, mesh, scalar_name, class_count):
    assert(class_count > 0)
    colors = get_color_map(class_count)
    if(class_count <20):
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": class_count})
    else:
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": 20})
def plot_with_title(plotter, mesh, scalar_name, class_count, title):
    assert(class_count > 0)
    colors = get_color_map(class_count)
    if(class_count <20):
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": class_count, "title": title})
    else:
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": 20, "title": title})

def comp_cur(mesh):
    # Compute mean curvature (default per vertex)
    curvature_point = mesh.curvature(curv_type='Mean')

    # Assign it as a point data array
    mesh.point_data["Mean Curvature (Points)"] = curvature_point
    # Apply Gaussian smoothing to the curvature values
    from scipy.ndimage import gaussian_filter

    # Reshape curvature values to match mesh dimensions
    curvature_point_smoothed = gaussian_filter(curvature_point, sigma=2)

    # Assign smoothed curvature as point data
    mesh.point_data["Smoothed Mean Curvature"] = curvature_point_smoothed


    # Approximate face-based curvature: average curvature at the face's vertices
    face_curvature = np.zeros(mesh.n_faces)

    for i, cell in enumerate(mesh.faces.reshape((-1, 4))):  # assuming triangle mesh (3 vertices + 1 size)
        ids = cell[1:]
        face_curvature[i] = np.mean(curvature_point[ids])


    # Apply sigmoid normalization
    percentile = 99
    threshold = np.percentile(face_curvature, percentile)

    a = threshold  # controls steepness
    b = np.mean(face_curvature)  # center of sigmoid
    # Clip values to prevent overflow in exp
    clipped_input = np.clip(a * (face_curvature - b), -500, 500)
    sigmoid_normalized = 1 / (1 + np.exp(-clipped_input))

    return sigmoid_normalized


def main(normal_weight, lr, radius, obj_file, min_region_rate, threshold_similarity_merge):
    # print("Running segmentation with the following parameters:")
    # print(f"Learning Rate: {lr}")
    # print(f"Radius: {radius}")
    # print(f"Object File: {obj_file}")
    # print(f"Min Region Face Count: {min_region_face_count}")
    # print(f"Similarity Merge Threshold: {threshold_similarity_merge}")
    
    obj_mesh = load_obj_with_face_normals(obj_file)
    face_adjacency = build_face_adjacency(obj_mesh)

    sdf_raw = []
    with open(obj_file.replace('.obj', '.sdf'), 'r') as f: #suppose there is *.sdf file in the same folder
        for line in f:
            face_id, value = line.strip().split(',')
            sdf_raw.append(float(value))
    sdf_raw = np.array(sdf_raw).reshape(-1, 1)  # Convert to 2D array with shape (n_samples, 1)
    
    # Normalize SDF values using min-max normalization
    sdf = sdf_raw/np.max(sdf_raw)
    cur = comp_cur(obj_mesh)
    # Combine SDF and curvature into 2D data
    features = np.concatenate((sdf, cur.reshape(-1, 1)), axis=1)    
    obj_mesh.cell_data['features'] = features


    # # Load mesh and assign 6D features, train SOM
    #data_for_som = np.concatenate((translated_xyz*(1-normal_weight), obj_mesh.cell_data['Normals']*normal_weight), axis=1)
    spherical_mesh = pv.read("regular_sphere.obj")
    som = SphereSOM(spherical_mesh, lr=lr, radius=radius)
    som.train(features, n_epochs=2000, n_rings=2)
    
    #Predict labels
    raw_labels = som.predict(features)
    raw_labels, raw_labels_count = remap_labels(raw_labels)  # Convert to face labels 0-based indices
    print("SOM clustering: there are {} clusters".format(raw_labels_count))
    obj_mesh.cell_data["raw_labels"] = raw_labels # Assign cluster labels to each face  
    
    #Separate disconnected components
    separated_region_labels = separate_disconnected_components(obj_mesh, face_adjacency, raw_labels)
    obj_mesh.cell_data["separated_region_labels"] = separated_region_labels 
    
    #Merge small regions to their biggest neighbor
    region_labels = merge_small_regions(obj_mesh, separated_region_labels, face_adjacency, min_region_rate)
    merged_small_region_labels, merged_region_labels_count = remap_labels(region_labels)  # Convert to face labels 0-based indices
    obj_mesh.cell_data["merged_small_region_labels"] = merged_small_region_labels
    
    # Merge similar regions based on normal direction
    merged_region_label_temp = merged_small_region_labels.copy()
    #for i in range(1):
    merged_region_label_temp = merge_similar_neighbor_regions(obj_mesh, merged_region_label_temp, face_adjacency, threshold_similarity_merge)
    merged_similar_region_labels, merged_similar_region_labels_count = remap_labels(merged_region_label_temp)  # Convert to face labels 0-based indices
    obj_mesh.cell_data["merged_similar_region_labels"] = merged_similar_region_labels



    # Create a scatter plot of features
    # plt.figure(figsize=(10, 6))
    # colors = get_color_map(raw_labels_count)
    # custom_cmap = mcolors.ListedColormap(colors)
    # plt.scatter(features[:, 0], features[:, 1], c=raw_labels, cmap=custom_cmap, alpha=0.6)
    # plt.xlabel('SDF Values')
    # plt.ylabel('Curvature Values')
    # plt.title('Feature Space Visualization')
    # plt.colorbar(label='Region Labels')
    # plt.show()

    # Create a 2x3 grid plotter #################################################################
    plotter = pv.Plotter(shape=(2, 3))

    plotter.subplot(0, 0)
    plotter.add_text("Initial SOM Clustering", font_size=12)
    plot(plotter, obj_mesh, "raw_labels", raw_labels_count)

    plotter.subplot(0, 1)
    plotter.add_text("After Disconnected Component Separation", font_size=12)
    obj_mesh01 = obj_mesh.copy()
    plot(plotter,obj_mesh01, "separated_region_labels", merged_region_labels_count)
    
    plotter.subplot(0, 2)
    plotter.add_text("After Small Region Merging", font_size=12)
    obj_mesh10 = obj_mesh.copy()
    plot(plotter,obj_mesh10, "merged_small_region_labels", merged_region_labels_count)
    
    plotter.subplot(1, 0)
    plotter.add_text("Final Segmentation", font_size=12)
    obj_mesh11 = obj_mesh.copy()
    plot_with_title(plotter, obj_mesh11, "merged_similar_region_labels", merged_similar_region_labels_count, "Segment ID")

    # Add SDF visualization
    plotter.subplot(1, 1)
    plotter.add_text("SDF Values", font_size=12)
    obj_mesh_sdf = obj_mesh.copy()
    obj_mesh_sdf["sdf"] = sdf
    plotter.add_mesh(obj_mesh_sdf, scalars='sdf', show_scalar_bar=True, show_edges=True, edge_opacity=0.2)

    # Add curvature visualization
    plotter.subplot(1, 2)
    plotter.add_text("Curvature Values", font_size=12)
    obj_mesh_curv = obj_mesh.copy()
    plotter.add_mesh(obj_mesh_curv, scalars='Smoothed Mean Curvature', show_scalar_bar=True, show_edges=True, edge_opacity=0.2)

    plotter.show()
    # plotter = pv.Plotter()
    # mesh_som = som.get_mesh() #.plot(show_edges=True)
    # plotter.add_mesh(mesh_som, show_edges=True, color="white", opacity=0.5)
    
    # Create glyphs for visualization
    # highlight_points = obj_mesh.points[raw_labels]
    # highlight_cloud = pv.PolyData(highlight_points)
    # glyphs = highlight_cloud.glyph(scale=False, geom=pv.Sphere(radius=0.02))
    
    # # Create color array for glyphs
    # colors = get_color_map(raw_labels_count)
    # # Convert hex colors to RGB values (0-1 range)
    # colors_rgb = np.array([mcolors.to_rgb(c) for c in colors])
    # colors_array = colors_rgb[raw_labels]
    # glyphs.point_data["colors"] = colors_array
    
    # # Add glyphs to plotter with proper color mapping
    # plotter.add_mesh(glyphs, scalars="colors", rgb=True, render_points_as_spheres=True, point_size=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spherical SOM surface segmentation.")
    parser.add_argument("--normal_weight", type=float, default = 1, help="Weight for normal direction")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate for SOM training")
    parser.add_argument("--radius", type=float, default=1, help="Neighborhood radius for SOM")
    parser.add_argument("--obj_file", type=str, required=True, help="Path to the OBJ file")
    parser.add_argument("--min_region_face_count", type=float, default=0.01, help="Minimum proportion of faces per region")
    parser.add_argument("--threshold_similarity_merge", type=float, default=0.8, help="Similarity threshold for region merging")

    args = parser.parse_args()
    main(args.normal_weight, args.lr, args.radius, args.obj_file, args.min_region_face_count, args.threshold_similarity_merge)
