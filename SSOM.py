import numpy as np
from sklearn.preprocessing import normalize
import pyvista as pv
from collections import defaultdict
import itertools
from collections import Counter

class SphereSOM:
    def __init__(self, mesh: pv.PolyData, normal_weight = 0.5, lr: float = 0.1, radius: float = 0.2):
        self.mesh = mesh
        self.positions = normalize(mesh.points.copy())           # (N, 3) fixed node positions        
        self.normal_weight = normal_weight
        self.n_nodes = mesh.n_points
        self.lr = lr
        self.radius = radius

    def _initialize_nodes(self, n_dim: int = 6):
        # Initialize SOM weights randomly in 6D space
        #self.weights = normalize(np.random.rand(self.n_nodes, self.input_dim).astype(np.float32))/3 # Randomly initialize weights in 6D space

        # normals = self.positions / np.linalg.norm(self.positions, axis=1, keepdims=True)
        #norm_ran = normalize(np.random.rand(self.n_nodes, 3).astype(np.float32)) 
        #self.weights = np.concatenate((self.positions*(1-self.normal_weight), np.random.rand(self.n_nodes, 3)*self.normal_weight), axis=1)/4
        #self.weights = self.positions / np.linalg.norm(self.positions, axis=1, keepdims=True)
        #self.mesh.compute_normals(point_normals=True, cell_normals=False, inplace=True)
        
        #self.positions = self.mesh.points/5
        if(n_dim == 3):
            self.weights = self.mesh.point_normals/5
        elif(n_dim == 6):
            self.weights = np.concatenate((self.positions*(1-self.normal_weight), np.random.rand(self.n_nodes, 3)*self.normal_weight), axis=1)/5
        else:
            #self.weights = np.ones((self.n_nodes, n_dim))
            self.weights = np.ones((self.n_nodes, n_dim), dtype=np.float32)*5
    def get_n_ring_neighbors(self, mu_idx, n):
        visited = set()
        current_ring = set([mu_idx])
        all_neighbors = set()
        for _ in range(n):
            next_ring = set()
            for vid in current_ring:
                neighbors = self.mesh.point_neighbors(vid)
                next_ring.update(neighbors)
            next_ring -= visited
            all_neighbors.update(next_ring)
            visited.update(current_ring)
            current_ring = next_ring
        return all_neighbors
    
    def train(self, data: np.ndarray, n_epochs: int = 1000, use_topology: bool = True, n_rings: int = -1):
        self._initialize_nodes(data.shape[1])
        for epoch in range(n_epochs):
            x = data[np.random.randint(0, len(data))]

            # Find BMU (closest in weight space)
            bmu_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
            bmu_pos = self.positions[bmu_idx]

            searched_nodes = self.get_n_ring_neighbors(bmu_idx, n_rings) if n_rings >= 0 else np.arange(self.n_nodes)
            for i in searched_nodes:
                dist = np.linalg.norm(self.positions[i] - bmu_pos)
                if dist <= self.radius:
                    influence = np.exp(-dist**2 / (2 * self.radius**2))
                    self.weights[i] += self.lr * influence * (x - self.weights[i])   

    def predict(self, data):
        #print(self.weights.shape, data.shape)
        dot_products = np.dot(data, self.weights.T)
        return np.argmax(dot_products, axis=1)
    
    def get_weights(self) -> np.ndarray:
        return self.weights
    
    def get_mesh(self) -> pv.PolyData:
        self.mesh.points = self.weights[:, :3]
        return self.mesh
    

def load_obj_with_face_normals(obj_path):
    import pyvista as pv
    import numpy as np

    mesh = pv.read(obj_path)    # Load the mesh

    # Calculate per-face center and normal
    face_centers = mesh.cell_centers().points  # shape: (n_faces, 3)
    mesh.cell_data["face_center"] = face_centers

    mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    #face_normals = mesh.cell_data['Normals']

    # Normalize face normals
    # norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    # face_normals = face_normals / np.where(norms == 0, 1, norms)
    
    #mesh.cell_data['Normals'] = face_normals

    return mesh

def build_face_adjacency(mesh):
    """Returns a dictionary: face index → list of adjacent face indices (via shared edge)"""
    edge_to_faces = defaultdict(list)
    faces = mesh.faces.reshape((-1, 4))[:, 1:]  # assuming triangles or quads

    for face_idx, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            edge = frozenset((face[i], face[(i + 1) % n]))
            edge_to_faces[edge].append(face_idx)

    adjacency = defaultdict(set)
    for edge, face_list in edge_to_faces.items():
        if len(face_list) > 1:
            for i in face_list:
                for j in face_list:
                    if i != j:
                        adjacency[i].add(j)
    return adjacency

def remap_labels(labels):
    """Remaps face labels to be continuous integers starting from 0."""
    unique_labels = np.unique(labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    return np.vectorize(label_map.get)(labels), len(unique_labels)

def compute_region_adjacency(face_adjacency, face_labels):     # Build a reverse map: face_id → region_id
    region_adjacency = defaultdict(set)
    # For each face, look at its neighbors
    for face, neighbors in face_adjacency.items():
        region = face_labels[face]
        if region is None:
            continue  # skip if face isn't part of any region

        for neighbor in neighbors:
            neighbor_region = face_labels[neighbor]
            if neighbor_region is None or neighbor_region == region:
                continue  # skip if same region or unknown
            region_adjacency[region].add(neighbor_region)
    return dict(region_adjacency)

def compute_region_to_faces(region_labels):
    region_to_faces = defaultdict(list)
    for face, region in enumerate(region_labels):
        if region != -1:  # Ignore unassigned faces
            region_to_faces[region].append(face)
    region_to_faces = dict(sorted(region_to_faces.items()))
    return region_to_faces  

def separate_disconnected_components(mesh, face_adjacency, raw_cluster_labels):
    unique_classes = np.unique(raw_cluster_labels)
    class_regions = {}
    visited = set()
    region_labels = -np.ones(mesh.n_cells, dtype=int) # Initialize region labels to -1
    region_id = 0

    for cls in unique_classes:
        class_faces = np.where(raw_cluster_labels == cls)[0]
        class_face_set = set(class_faces)

        regions = []
        for face in class_faces:
            if face in visited:
                continue
            # BFS to find all connected faces of same class
            queue = [face]
            region = []
            while queue: 
                current = queue.pop()
                if current in visited or current not in class_face_set:
                    continue
                visited.add(current)
                region.append(current)
                queue.extend(face_adjacency[current])
            regions.append(region)
            for f in region:
                region_labels[f] = region_id
            region_id += 1
        class_regions[cls] = regions

    region_labels_count = len(np.unique(region_labels))
    print("After separate_disconnected_components, There are {} regions having data".format(region_labels_count))

    return region_labels



def merge_small_regions(mesh, region_labels, face_adjacency, min_region_rate):      
    FACE_COUNT_THRESHOLD = len(region_labels)*min_region_rate
    print("FACE_COUNT_THRESHOLD: ", FACE_COUNT_THRESHOLD)
    merged_labels = region_labels.copy() # Initialize merged labels with region labels

    region_to_faces = compute_region_to_faces(merged_labels)
    small_regions = [region for region, faces in region_to_faces.items() if len(faces) < FACE_COUNT_THRESHOLD]
    round = 0
    while len(small_regions) > 0 and round < 100:
        round += 1
        print("In merge_small_regions, Round {}, there are {} small regions".format(round, len(small_regions)))
                
        region_adjacency = compute_region_adjacency(face_adjacency, merged_labels)
        avg_normals = {}
        for region_id, face_indices in region_to_faces.items(): # Calculate average of normals for each region
            region_normals = mesh.cell_data['Normals'][face_indices]  # shape (n_faces_in_region, 3)
            avg_normals[region_id] = np.mean(region_normals, axis=0)
            
        small_regions.sort(key=lambda region: len(region_to_faces[region]))
        target_regions = set() # regions that another have been merged to, so should not be merged again in this round
        for rid in small_regions:
            if rid in target_regions:
                continue
            reg_norm = avg_normals[rid]
            most_similar_region_id = -1
            most_similarity = -1.001
            for nr_id in region_adjacency.get(rid, set()):
                nr_norm = avg_normals[int(nr_id)]
                similarity = np.dot(reg_norm, nr_norm)  # cosine similarity
                if similarity > most_similarity:
                    most_similarity = similarity
                    most_similar_region_id = nr_id
            if most_similar_region_id != -1:
                #print("Region {} is merged into {}".format(rid, most_similar_region_id))
                faces = region_to_faces[rid]
                for f in faces: 
                    merged_labels[f] = most_similar_region_id    
                target_regions.add(most_similar_region_id)
            else:
                print("Similarity: {}, No similar region found for region {}".format(most_similarity, rid))
        
        region_to_faces = compute_region_to_faces(merged_labels)
        small_regions = [region for region, faces in region_to_faces.items() if len(faces) < FACE_COUNT_THRESHOLD]
                
    return merged_labels

def merge_small_regions_area_weighted(mesh, region_labels, face_adjacency, min_region_rate):      
    FACE_COUNT_THRESHOLD = len(region_labels)*min_region_rate
    print("FACE_COUNT_THRESHOLD: ", FACE_COUNT_THRESHOLD)
    merged_labels = region_labels.copy() # Initialize merged labels with region labels

    # Calculate areas for all faces
    face_areas = np.zeros(mesh.n_cells)
    for i in range(mesh.n_cells):
        face = mesh.faces[4*i+1:4*i+4]  # Get vertices of the face
        v1, v2, v3 = mesh.points[face]
        # Calculate area using cross product
        face_areas[i] = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    region_to_faces = compute_region_to_faces(merged_labels)
    small_regions = [region for region, faces in region_to_faces.items() if len(faces) < FACE_COUNT_THRESHOLD]
    round = 0
    while len(small_regions) > 0 and round < 100:
        round += 1
        print("In merge_small_regions_area_weighted, Round {}, there are {} small regions".format(round, len(small_regions)))
                
        region_adjacency = compute_region_adjacency(face_adjacency, merged_labels)
        avg_normals = {}
        for region_id, face_indices in region_to_faces.items():
            # Calculate area-weighted average normal for the region
            region_normals = mesh.cell_data['Normals'][face_indices]  # shape (n_faces_in_region, 3)
            region_areas = face_areas[face_indices]  # shape (n_faces_in_region,)
            total_area = np.sum(region_areas)
            if total_area > 0:  # Avoid division by zero
                weighted_normals = region_normals * region_areas[:, np.newaxis]
                avg_normals[region_id] = np.sum(weighted_normals, axis=0) / total_area
            else:
                avg_normals[region_id] = np.mean(region_normals, axis=0)  # Fallback to simple average
            
        small_regions.sort(key=lambda region: len(region_to_faces[region]))
        target_regions = set() # regions that another have been merged to, so should not be merged again in this round
        for rid in small_regions:
            if rid in target_regions:
                continue
            reg_norm = avg_normals[rid]
            most_similar_region_id = -1
            most_similarity = -1.001
            for nr_id in region_adjacency.get(rid, set()):
                nr_norm = avg_normals[int(nr_id)]
                similarity = np.dot(reg_norm, nr_norm)  # cosine similarity
                if similarity > most_similarity:
                    most_similarity = similarity
                    most_similar_region_id = nr_id
            if most_similar_region_id != -1:
                faces = region_to_faces[rid]
                for f in faces: 
                    merged_labels[f] = most_similar_region_id    
                target_regions.add(most_similar_region_id)
            else:
                print("Similarity: {}, No similar region found for region {}".format(most_similarity, rid))
        
        region_to_faces = compute_region_to_faces(merged_labels)
        small_regions = [region for region, faces in region_to_faces.items() if len(faces) < FACE_COUNT_THRESHOLD]
                
    return merged_labels

def merge_similar_direction_regions(mesh, region_labels, face_adjacency, threshold_similarity_merge):
    region_to_faces = compute_region_to_faces(region_labels)   
    # Calculate average and variance of normals for each region
    avg_normals = {}
    for region_id, face_indices in region_to_faces.items(): # Calculate average of normals for each region
        region_normals = mesh.cell_data['Normals'][face_indices]  # shape (n_faces_in_region, 3)
        # var = np.var(region_normals, axis=0)
        # print("region_id, var: ", region_id, np.linalg.norm(var))
        avg_normals[region_id] = np.mean(region_normals, axis=0)
        
    merged_region_labels = region_labels.copy()
    count_face_region = Counter(merged_region_labels)  # Count faces in each region
    
    round = 0
    still_having_similar_regions = True
    while still_having_similar_regions:
        round += 1
        print("In merge_similar_direction_regions, Round {}, there are {} regions".format(round, len(region_to_faces)))
        still_having_similar_regions = False
        for (r1, n1), (r2, n2) in itertools.combinations(avg_normals.items(), 2):  # Find similar adjacent pairs and merge them
            region_adjacency = compute_region_adjacency(face_adjacency, region_labels)
            if  r1 in region_adjacency.get(r2, set()):
                similarity = np.dot(n1, n2)  # cosine similarity
                #print(r1, r2, similarity)
                if similarity > threshold_similarity_merge:
                    print(f"Region {r1} and Region {r2} have similar normals (cosine similarity = {similarity:.3f})")
                    merged_id = -1
                    more_common_id = -1
                    if count_face_region[r1] < count_face_region[r2]:  # Merge into the smaller ID
                        merged_id = r1
                        more_common_id = r2
                    else:
                        merged_id = r2
                        more_common_id = r1
                    for f in region_to_faces[merged_id]:        
                        merged_region_labels[f] = more_common_id          
                    still_having_similar_regions = True
        region_to_faces = compute_region_to_faces(merged_region_labels)
        avg_normals = {}
        for region_id, face_indices in region_to_faces.items(): # Calculate average of normals for each region
            region_normals = mesh.cell_data['Normals'][face_indices]  # shape (n_faces_in_region, 3)
            avg_normals[region_id] = np.mean(region_normals, axis=0)
    return merged_region_labels

def merge_similar_direction_regions_area_weighted(mesh, region_labels, face_adjacency, threshold_similarity_merge):
    region_to_faces = compute_region_to_faces(region_labels)   
    
    # Calculate areas for all faces
    face_areas = np.zeros(mesh.n_cells)
    for i in range(mesh.n_cells):
        face = mesh.faces[4*i+1:4*i+4]  # Get vertices of the face
        v1, v2, v3 = mesh.points[face]
        # Calculate area using cross product
        face_areas[i] = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    # Calculate area-weighted average normals for each region
    avg_normals = {}
    for region_id, face_indices in region_to_faces.items():
        region_normals = mesh.cell_data['Normals'][face_indices]  # shape (n_faces_in_region, 3)
        region_areas = face_areas[face_indices]  # shape (n_faces_in_region,)
        total_area = np.sum(region_areas)
        if total_area > 0:  # Avoid division by zero
            weighted_normals = region_normals * region_areas[:, np.newaxis]
            avg_normals[region_id] = np.sum(weighted_normals, axis=0) / total_area
        else:
            avg_normals[region_id] = np.mean(region_normals, axis=0)  # Fallback to simple average
        
    merged_region_labels = region_labels.copy()
    count_face_region = Counter(merged_region_labels)  # Count faces in each region
    
    round = 0
    still_having_similar_regions = True
    while still_having_similar_regions:
        round += 1
        print("In merge_similar_direction_regions_area_weighted, Round {}, there are {} regions".format(round, len(region_to_faces)))
        still_having_similar_regions = False
        for (r1, n1), (r2, n2) in itertools.combinations(avg_normals.items(), 2):  # Find similar adjacent pairs and merge them
            region_adjacency = compute_region_adjacency(face_adjacency, region_labels)
            if r1 in region_adjacency.get(r2, set()):
                similarity = np.dot(n1, n2)  # cosine similarity
                if similarity > threshold_similarity_merge:
                    print(f"Region {r1} and Region {r2} have similar normals (cosine similarity = {similarity:.3f})")
                    merged_id = -1
                    more_common_id = -1
                    if count_face_region[r1] < count_face_region[r2]:  # Merge into the smaller ID
                        merged_id = r1
                        more_common_id = r2
                    else:
                        merged_id = r2
                        more_common_id = r1
                    for f in region_to_faces[merged_id]:        
                        merged_region_labels[f] = more_common_id          
                    still_having_similar_regions = True
        
        # Recompute region mappings and normals after merging
        region_to_faces = compute_region_to_faces(merged_region_labels)
        avg_normals = {}
        for region_id, face_indices in region_to_faces.items():
            region_normals = mesh.cell_data['Normals'][face_indices]
            region_areas = face_areas[face_indices]
            total_area = np.sum(region_areas)
            if total_area > 0:
                weighted_normals = region_normals * region_areas[:, np.newaxis]
                avg_normals[region_id] = np.sum(weighted_normals, axis=0) / total_area
            else:
                avg_normals[region_id] = np.mean(region_normals, axis=0)
                
    return merged_region_labels
        