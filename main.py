from turtle import title
import numpy as np
import pyvista as pv
from SSOM import *
import argparse
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
    while (len(colors) < class_count):
        c = cmap(i / class_count)[:3]  # Use n+1 to avoid endpoints
        if not np.allclose(c, (0, 0, 0)):  # Exclude black
            colors.append(mcolors.to_hex(c))
        i += 1
    return colors
def plot_with_title(plotter, mesh, scalar_name, class_count, title):
    assert(class_count > 0)
    colors = get_color_map(class_count)
    if(class_count <20):
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": class_count, "title": title})
    else:
        plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.2, scalar_bar_args={"fmt": "%.0f", "n_labels": 20, "title": title})
def plot(plotter, mesh, scalar_name, class_count):    
    assert(class_count > 0)
    colors = get_color_map(class_count)
    plotter.add_mesh(mesh, scalars=scalar_name,  cmap = colors, show_scalar_bar=True, show_edges=True, edge_opacity=0.3, scalar_bar_args={"fmt": "%.0f", "n_labels": class_count})


def main(normal_weight, lr, radius, obj_file, min_region_rate, threshold_similarity_merge):
   
    obj_mesh = load_obj_with_face_normals(obj_file)
    face_adjacency = build_face_adjacency(obj_mesh)

    normed_xyz = normalize(obj_mesh.cell_data["face_center"]) 
    n = len(normed_xyz)
    cx = sum(p[0] for p in normed_xyz) / n
    cy = sum(p[1] for p in normed_xyz) / n
    cz = sum(p[2] for p in normed_xyz) / n
    translated_xyz = np.array([(x - cx, y - cy, z - cz) for x, y, z in normed_xyz])# Translate all points

    import time
    start_time = time.time()
    # # Load mesh and assign 6D features, train SOM
    #data_for_som = np.concatenate((translated_xyz*(1-normal_weight), obj_mesh.cell_data['Normals']*normal_weight), axis=1)
    data_for_som = obj_mesh.cell_data['Normals']
    spherical_mesh = pv.read("regular_sphere.obj")
    som = SphereSOM(spherical_mesh, normal_weight=normal_weight, lr=lr, radius=radius)
    som.train(data_for_som, n_epochs=2000, n_rings=-1)
    
    #Predict labels
    raw_labels = som.predict(data_for_som)
    unique_raw_labels = np.unique(raw_labels)
    raw_labels, raw_labels_count = remap_labels(raw_labels)  # Convert to face labels 0-based indices
    #raw_labels_count = len(np.unique(raw_labels))    
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
    merged_region_label_temp = merge_similar_direction_regions(obj_mesh, merged_region_label_temp, face_adjacency, threshold_similarity_merge)
    merged_similar_region_labels, merged_similar_region_labels_count = remap_labels(merged_region_label_temp)  # Convert to face labels 0-based indices
    obj_mesh.cell_data["merged_similar_region_labels"] = merged_similar_region_labels

    end_time = time.time()
    running_time = end_time - start_time
    print(f"Total running time: {running_time:.2f} seconds")

    # Start plotting, Create a 2x3 grid plotter ##################################################
    plotter = pv.Plotter(shape=(2, 3))

    plotter.subplot(0, 0) #-----------------------------------------------
    plotter.add_mesh(obj_mesh, color='grey', show_edges=True, edge_opacity=0.2)    

    plotter.subplot(0, 1) #-----------------------------------------------
    plot(plotter, obj_mesh, "raw_labels", raw_labels_count)   

    plotter.subplot(0, 2) #-----------------------------------------------
    spherical_ori = pv.read("regular_sphere.obj")
    plotter.add_mesh(spherical_ori, color='white', opacity=0.3, show_edges=True)   
    vertex_points = spherical_ori.points     # Create point cloud for vertices
    vertex_cloud = pv.PolyData(vertex_points)    
    # Create spheres at vertex positions
    glyphs = vertex_cloud.glyph(scale=True, geom=pv.Sphere(radius=0.02))    
    # Add the spheres to the plotter
    plotter.add_mesh(glyphs, color='grey', render_points_as_spheres=True, point_size=10)
    
    highlight_points = spherical_mesh.points[unique_raw_labels]
    highlight_cloud = pv.PolyData(highlight_points)
    glyphs = highlight_cloud.glyph(scale=True, geom=pv.Sphere(radius=0.05))
    colors = get_color_map(raw_labels_count)
    colors_rgb = np.array([mcolors.to_rgb(c) for c in colors])    # Convert hex colors to RGB values (0-255)
    colors_glyphs = (colors_rgb * 255).astype(np.uint8)
    glyphs.point_data["colors"] = np.repeat(colors_glyphs, glyphs.n_points // len(colors_glyphs), axis=0)
    plotter.add_mesh(glyphs, scalars="colors", cmap=get_color_map(len(unique_raw_labels)), opacity= 1, show_scalar_bar=True, rgb=True, point_size=30,  scalar_bar_args={"fmt": "%.0f", "n_labels": raw_labels_count})   
    plotter.add_scalar_bar(title="Cluster ID", n_labels=raw_labels_count, vertical=False)

    plotter.subplot(1, 0) #-----------------------------------------------
    obj_mesh01 = obj_mesh.copy()
    plot(plotter,obj_mesh01, "separated_region_labels",  len(np.unique(separated_region_labels)))   
    
    plotter.subplot(1, 1) #-----------------------------------------------
    obj_mesh10 = obj_mesh.copy()
    plot(plotter,obj_mesh10, "merged_small_region_labels", merged_region_labels_count)   

    plotter.subplot(1, 2) #-----------------------------------------------
    obj_mesh11 = obj_mesh.copy()
    plot(plotter, obj_mesh11, "merged_similar_region_labels", merged_similar_region_labels_count)
    plotter.show()
     #End plotting               ############################################################    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spherical SOM surface segmentation.")
    parser.add_argument("--normal_weight", type=float, default = 1, help="Weight for normal direction")
    parser.add_argument("--lr", type=float, default=0.2, help="Learning rate for SOM training")
    parser.add_argument("--radius", type=float, default=0.1, help="Neighborhood radius for SOM")
    parser.add_argument("--obj_file", type=str, default = "./Models/brick_part01.obj",  help="Path to the OBJ file")
    parser.add_argument("--min_region_face_count", type=float, default=0.01, help="Minimum proportion of faces per region")
    parser.add_argument("--threshold_similarity_merge", type=float, default=0.8, help="Similarity threshold for region merging")

    args = parser.parse_args()
    main(args.normal_weight, args.lr, args.radius, args.obj_file, args.min_region_face_count, args.threshold_similarity_merge)
