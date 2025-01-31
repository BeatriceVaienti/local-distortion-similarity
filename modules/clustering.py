from collections import defaultdict
import sys
sys.path.append('../')
import modules.transform as transform
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
from skimage import measure
from scipy.cluster.hierarchy import linkage, fcluster
import networkx as nx
from collections import defaultdict


def identify_shared_clusters_2d(clustering_results, grid_shape):
    """
    Identifies shared clusters between neighboring cells in a 2D grid.
    
    A cluster is considered shared if at least two maps from one cluster
    in a cell also appear in a neighboring cell's cluster.

    Parameters:
    - clustering_results (dict): Mapping of (i, j) grid positions to clustering results.
    - grid_shape (tuple): Shape of the grid (rows, cols).

    Returns:
    - shared_clusters (dict): Dictionary where keys are ((i, j), (ni, nj)) pairs representing neighboring cells,
      and values are lists of frozensets containing shared maps.
    """
    shared_clusters = defaultdict(list)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    rows, cols = grid_shape

    for (i, j), result in clustering_results.items():
        clusters, map_indices = result['clusters'], result['map_indices']

        for di, dj in directions:
            ni, nj = i + di, j + dj  # Compute neighbor indices

            # Ensure neighbor exists within grid bounds and is in clustering_results
            if not (0 <= ni < rows and 0 <= nj < cols and (ni, nj) in clustering_results):
                continue

            neighbor_result = clustering_results[(ni, nj)]
            neighbor_clusters, neighbor_map_indices = neighbor_result['clusters'], neighbor_result['map_indices']

            # Convert clusters to sets for quick lookups
            cluster_maps = {
                cluster: {map_indices[idx] for idx, c in enumerate(clusters) if c == cluster}
                for cluster in set(clusters) if cluster != -1
            }

            neighbor_cluster_maps = {
                cluster: {neighbor_map_indices[idx] for idx, c in enumerate(neighbor_clusters) if c == cluster}
                for cluster in set(neighbor_clusters) if cluster != -1
            }

            # Compare clusters between current cell and neighbor
            for cluster, maps_in_cluster in cluster_maps.items():
                for neighbor_cluster, maps_in_neighbor_cluster in neighbor_cluster_maps.items():
                    intersection = maps_in_cluster & maps_in_neighbor_cluster
                    if len(intersection) >= 2:
                        shared_clusters[((i, j), (ni, nj))].append(frozenset(intersection))

    return shared_clusters


def filter_deformed_tokens(padded_maps, grid_shape, normalized_undeformed_vector, threshold_undeformed, comparison):
    """
    Identifies tokens whose deformation exceeds a given similarity threshold.

    Parameters:
    - padded_maps (list): List of processed maps.
    - grid_shape (tuple): Shape of the map grid.
    - undeformed_vector (numpy.array): The reference undeformed vector.
    - similarity_threshold (float): Threshold for considering a token as deformed.
    - comparison (module): Module containing similarity functions.

    Returns:
    - token_vectors_dict (dict): Dictionary mapping (i, j) coordinates to a list of (map_idx, normalized_vector).
    """
    token_vectors_dict = {}
    
    for map_idx, curr_map in enumerate(padded_maps):
        token_vectors = curr_map.grid.graph.alpha_masked_token_vectors
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                if not np.isnan(token_vectors[i, j]).any():
                    normalized_vector = comparison.normalize_vector(token_vectors[i, j])
                    distance_from_undeformed = comparison.ratio_distance(normalized_vector, normalized_undeformed_vector)
                    if distance_from_undeformed > threshold_undeformed:
                        if (i, j) not in token_vectors_dict:
                            token_vectors_dict[(i, j)] = []
                        token_vectors_dict[(i, j)].append((map_idx, normalized_vector))
    return token_vectors_dict

def cluster_tokens(token_vectors_dict, max_distance, comparison):
    """
    Clusters token vectors using hierarchical clustering (Complete Linkage) based on their ratio distances.

    Parameters:
    - token_vectors_dict (dict): Dictionary mapping (i, j) coordinates to token vectors.
    - max_distance (float): Maximum distance for hierarchical clustering to form clusters.
    - comparison (module): Module containing distance functions.

    Returns:
    - clustering_results (dict): Dictionary storing clustering results for each token.
    """
    clustering_results = {}

    for token, vectors in token_vectors_dict.items():
        map_indices, vectors = zip(*vectors)
        vectors = np.array(vectors)

        if len(vectors) < 2:
            continue  # Skip if not enough data for clustering

        # Normalize vectors and compute pairwise distances
        normalized_vectors = np.array([comparison.normalize_vector(vec) for vec in vectors])
        pairwise_distances = pdist(normalized_vectors, metric=comparison.ratio_distance)

        # Perform hierarchical clustering (Complete Linkage)
        Z = linkage(pairwise_distances, method='complete')
        clusters = fcluster(Z, max_distance, criterion='distance')

        # Store clustering results
        clustering_results[token] = {
            "clusters": clusters,
            "vectors": vectors,
            "map_indices": map_indices
        }

    return clustering_results


def visualize_clustering(selected_tokens, clustering_results, plot=True):
    """
    Visualizes the clustering results for selected tokens.

    Parameters:
    - selected_tokens (list): List of tokens to visualize.
    - clustering_results (dict): Dictionary storing clustering results for each token.
    - plot (bool): Whether to generate plots.
    """
    if not plot:
        return

    for token in selected_tokens:
        result = clustering_results[token]
        clusters, vectors, map_indices = result["clusters"], result["vectors"], result["map_indices"]
        unique_clusters = np.unique(clusters)
        num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)  # Exclude noise

        # Skip plotting if no valid clusters exist
        if num_clusters == 0:
            print(f"Skipping visualization for token {token} (only noise detected)")
            continue

        # If there's only one cluster, use a single plot
        if num_clusters == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            cluster = unique_clusters[0]  # The only valid cluster
            cluster_indices = np.where(clusters == cluster)[0]
            cluster_vectors = vectors[cluster_indices]
            cluster_maps = [map_indices[idx] for idx in cluster_indices]

            for vec in cluster_vectors:
                ax.plot(vec, alpha=0.5)

            print(f'Token {token} - Cluster {cluster}')
            ax.set_title(f'Token {token} - Cluster {cluster}\nMaps: {cluster_maps}')
            ax.set_xlabel('Vector Index')
            ax.set_ylabel('Vector Value')
        else:
            fig, axes = plt.subplots(1, num_clusters, figsize=(15, 6))

            if num_clusters == 1:
                axes = [axes]

            cluster_index = 0
            for cluster in unique_clusters:
                if cluster == -1:
                    continue  # Skip noise

                cluster_indices = np.where(clusters == cluster)[0]
                cluster_vectors = vectors[cluster_indices]
                cluster_maps = [map_indices[idx] for idx in cluster_indices]

                for vec in cluster_vectors:
                    axes[cluster_index].plot(vec, alpha=0.5)

                print(f'Token {token} - Cluster {cluster}')
                axes[cluster_index].set_title(f'Token {token} - Cluster {cluster}\nMaps: {cluster_maps}')
                axes[cluster_index].set_xlabel('Vector Index')
                axes[cluster_index].set_ylabel('Vector Value')
                cluster_index += 1

        plt.suptitle(f'Clustering of Token {token} Vectors')
        plt.show()


def generate_cluster_grid(clustering_results, grid_shape):
    """
    Generates grid representations of cluster distributions.

    Parameters:
    - clustering_results (dict): Dictionary storing clustering results for each token.
    - grid_shape (tuple): Shape of the map grid.

    Returns:
    - cluster_grid (numpy.array): Grid with the count of distinct clusters per token.
    - enhanced_cluster_grid (numpy.array): Grid including noise clusters.
    - cluster_grid_normalized (numpy.array): Grid normalized by the number of maps per token.
    """
    cluster_grid = np.full(grid_shape, np.nan)
    enhanced_cluster_grid = np.full(grid_shape, np.nan)
    cluster_grid_normalized = np.full(grid_shape, np.nan)

    for (i, j), result in clustering_results.items():
        clusters = result["clusters"]
        valid_clusters = clusters[clusters != -1]
        noise_clusters = clusters[clusters == -1]

        if len(valid_clusters) > 0:
            cluster_grid[i, j] = len(set(valid_clusters))  # Count valid clusters
        enhanced_cluster_grid[i, j] = len(set(valid_clusters)) + len(noise_clusters)  # Include noise
        cluster_grid_normalized[i, j] = len(set(valid_clusters)) / len(result["map_indices"])

    return cluster_grid, enhanced_cluster_grid, cluster_grid_normalized


def visualize_cluster_grid(enhanced_cluster_grid, cmap, plot=True):
    """
    Visualizes the cluster grid if enabled.

    Parameters:
    - enhanced_cluster_grid (numpy.array): Grid representing the number of clusters per token.
    - cmap (matplotlib colormap): Colormap for visualization.
    - plot (bool): Whether to generate plots.
    """
    if not plot:
        return

    plt.figure(figsize=(10, 10))
    plt.imshow(enhanced_cluster_grid, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Number of Clusters (Including Single Maps)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()
    plt.title('Number of Clusters per Token (Including Noise)')
    plt.show()



def build_shared_cluster_graphs(shared_clusters):
    """
    Constructs a graph for each unique shared cluster.

    Parameters:
    - shared_clusters (dict): A dictionary mapping point pairs to sets of shared clusters.

    Returns:
    - graph_dict (dict): A dictionary mapping cluster sets to their corresponding NetworkX graph.
    """
    unique_sets = set()
    for pairs in shared_clusters.values():
        for cluster_set in pairs:
            unique_sets.add(cluster_set)

    graph_dict = {cluster_set: nx.Graph() for cluster_set in unique_sets}

    for (point1, point2), clusters in shared_clusters.items():
        for cluster_set in clusters:
            graph_dict[cluster_set].add_edge(point1, point2)  # Implicitly adds nodes

    return graph_dict


def extract_regions_from_graphs(graph_dict):
    """
    Identifies connected components (regions) from cluster graphs.

    Parameters:
    - graph_dict (dict): A dictionary mapping cluster sets to their corresponding NetworkX graph.

    Returns:
    - regions (dict): A dictionary mapping cluster sets to lists of connected components.
    """
    regions = defaultdict(list)
    for cluster_set, graph in graph_dict.items():
        connected_components = list(nx.connected_components(graph))
        regions[cluster_set].extend(connected_components)
    return regions


def sort_regions_by_size(regions):
    """
    Sorts the identified regions by size in descending order.

    Parameters:
    - regions (dict): A dictionary mapping cluster sets to lists of connected components.

    Returns:
    - sorted_regions (list): A list of (component, cluster_set) tuples sorted by component size.
    """
    region_list = [(component, cluster_set) for cluster_set, components in regions.items() for component in components]
    return sorted(region_list, key=lambda x: len(x[0]), reverse=True)


import networkx as nx
import time
import tqdm

def are_touching(component1, component2):
    """
    Check if two sets of points are touching (adjacent or overlapping).

    Parameters:
    - component1 (set): Set of (x, y) points in the first region.
    - component2 (set): Set of (x, y) points in the second region.

    Returns:
    - bool: True if the two sets of points are adjacent or overlap.
    """
    return any(abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1 for x1, y1 in component1 for x2, y2 in component2)


def merge_regions(region_list):
    """
    Merge regions based on adjacency and similarity of cluster sets.

    Parameters:
    - region_list (list of dicts): List of regions where each region contains:
        - "component": Set of (x, y) points in the region.
        - "cluster_set": A set representing clusters.

    Returns:
    - region_graph (networkx.DiGraph): Graph where nodes represent regions,
      and edges represent merged connections.
    """
    region_graph = nx.DiGraph()
    checked_pairs = set()
    merge_count = 0

    # Add regions as nodes in the graph
    for idx, region in enumerate(region_list):
        region_graph.add_node(idx, component=region["component"], cluster_set=region["cluster_set"])

    changes_made = True
    while changes_made:
        changes_made = False
        with tqdm.tqdm(total=len(region_graph.nodes), desc="Merging regions") as pbar:
            for i in list(region_graph.nodes):
                if not region_graph.has_node(i):  # Skip removed regions
                    pbar.update(1)
                    continue

                region_i = region_graph.nodes[i]
                component_i, cluster_set_i = region_i["component"], region_i["cluster_set"]

                for j in list(region_graph.nodes):
                    if i == j or not region_graph.has_node(j) or (i, j) in checked_pairs:
                        continue

                    checked_pairs.add((i, j))
                    checked_pairs.add((j, i))
                    region_j = region_graph.nodes[j]
                    component_j, cluster_set_j = region_j["component"], region_j["cluster_set"]

                    if are_touching(component_i, component_j):
                        changes_made = True
                        if (cluster_set_i.issubset(cluster_set_j) or cluster_set_j.issubset(cluster_set_i)) and cluster_set_i != cluster_set_j:
                            # Extend the smaller component
                            if len(cluster_set_i) < len(cluster_set_j):
                                region_graph.nodes[i]["component"].update(component_j)
                                region_graph.add_edge(i, j)
                            else:
                                region_graph.nodes[j]["component"].update(component_i)
                                region_graph.add_edge(j, i)
                        elif cluster_set_i == cluster_set_j:
                            # Merge components
                            region_graph.nodes[i]["component"].update(component_j)
                            region_graph.add_edge(i, j)
                            region_graph.remove_node(j)
                            merge_count += 1

                pbar.update(1)

    print(f"Total merges: {merge_count}")
    return region_graph


def filter_graph(region_graph, min_elements):
    """
    Filters out regions with fewer than min_elements points.

    Parameters:
    - region_graph (networkx.Graph): Graph containing region nodes.
    - min_elements (int): Minimum number of points required to keep a region.

    Returns:
    - networkx.Graph: Filtered graph with small regions removed.
    """
    filtered_graph = region_graph.copy()
    nodes_to_remove = [node for node, data in filtered_graph.nodes(data=True) if len(data["component"]) < min_elements]
    filtered_graph.remove_nodes_from(nodes_to_remove)
    return filtered_graph


def find_directed_reachability_subgraphs(region_graph):
    """
    Identifies all directed reachability subgraphs.

    Parameters:
    - region_graph (networkx.DiGraph): The region graph.

    Returns:
    - list: A list of subgraphs where each subgraph represents a connected component of reachability.
    """
    reachability_subgraphs = []
    visited = set()

    for node in region_graph.nodes:
        if node not in visited:
            # Perform a depth-first search to find reachable nodes
            reachable_nodes = list(nx.dfs_postorder_nodes(region_graph, source=node))
            subgraph = region_graph.subgraph(reachable_nodes).copy()

            # Check if any existing subgraph already contains this one
            if not any(set(reachable_nodes).issubset(set(existing_subgraph.nodes)) for existing_subgraph in reachability_subgraphs):
                reachability_subgraphs.append(subgraph)
                visited.update(reachable_nodes)

    return reachability_subgraphs

def initialize_components(region_graph):
    """
    Ensures all nodes in the region graph have a 'component' attribute.

    Parameters:
    - region_graph (networkx.Graph): The graph containing region nodes.

    Returns:
    - networkx.Graph: Updated graph with initialized components.
    """
    for node in region_graph.nodes:
        if "component" not in region_graph.nodes[node] or region_graph.nodes[node]["component"] is None:
            region_graph.nodes[node]["component"] = set()
    return region_graph


def process_reachability_subgraphs(region_graph):
    """
    Iteratively updates reachability subgraphs by merging connected components.

    Parameters:
    - region_graph (networkx.DiGraph): The graph containing region nodes.

    Returns:
    - networkx.DiGraph: Updated graph after merging reachability subgraphs.
    """
    print("Processing reachability subgraphs")
    changes_made = True

    while changes_made:
        changes_made = False
        for node in tqdm.tqdm(list(region_graph.nodes), desc="Updating components"):
            if region_graph.has_node(node):
                initial_size = len(region_graph.nodes[node]["component"])
                component = set(region_graph.nodes[node]["component"])
                for successor in nx.descendants(region_graph, node):
                    component.update(region_graph.nodes[successor]["component"])
                region_graph.nodes[node]["component"] = component
                if len(component) > initial_size:
                    changes_made = True

    return region_graph


def merge(region_graph):
    """
    Merges adjacent nodes in a region graph if they share the same cluster set.

    Parameters:
    - region_graph (networkx.DiGraph): The graph containing region nodes.

    Returns:
    - networkx.DiGraph: Updated graph after merging.
    """
    changes_made = True
    while changes_made:
        changes_made = False
        nodes_to_check = list(region_graph.nodes)
        with tqdm.tqdm(total=len(nodes_to_check), desc="Merging regions") as pbar:
            for i in nodes_to_check:
                if not region_graph.has_node(i):
                    pbar.update(1)
                    continue
                
                component_i = region_graph.nodes[i]["component"]
                cluster_set_i = region_graph.nodes[i]["cluster_set"]
                
                for j in nodes_to_check:
                    if i != j and region_graph.has_node(j):
                        cluster_set_j = region_graph.nodes[j]["cluster_set"]
                        if cluster_set_i == cluster_set_j and are_touching(component_i, region_graph.nodes[j]["component"]):
                            # Merge components
                            component_i.update(region_graph.nodes[j]["component"])
                            region_graph.nodes[i]["component"] = component_i
                            # Add edges
                            for successor in list(region_graph.successors(j)):
                                region_graph.add_edge(i, successor)
                            for predecessor in list(region_graph.predecessors(j)):
                                region_graph.add_edge(predecessor, i)
                            # Remove merged node
                            region_graph.remove_node(j)
                            changes_made = True
                
                pbar.update(1)
    
    return region_graph

def iterative_process_and_merge(region_graph):
    """
    Iteratively processes reachability subgraphs and merges them until no changes occur.

    Parameters:
    - region_graph (networkx.DiGraph): The graph containing region nodes.

    Returns:
    - networkx.DiGraph: Updated graph after iterative merging.
    """
    changes_made = True
    while changes_made:
        changes_made = False
        region_graph_before = region_graph.copy()
        region_graph = process_reachability_subgraphs(region_graph)
        region_graph = merge(region_graph)

        if region_graph.nodes != region_graph_before.nodes or region_graph.edges != region_graph_before.edges:
            changes_made = True

    return region_graph


def finalize_merges_new(region_graph):
    """
    Finalizes merging by ensuring all nodes with the same cluster set are merged.

    Parameters:
    - region_graph (networkx.DiGraph): The graph containing region nodes.

    Returns:
    - networkx.DiGraph: Updated graph after finalizing merges.
    """
    nodes_to_check = list(region_graph.nodes)
    nodes_to_remove = set()

    def process_removals():
        """Removes nodes marked for deletion."""
        for node in tqdm.tqdm(nodes_to_remove, desc="Removing nodes"):
            if region_graph.has_node(node):
                region_graph.remove_node(node)
        nodes_to_remove.clear()

    with tqdm.tqdm(total=len(nodes_to_check), desc="Final merging") as pbar:
        for i in nodes_to_check:
            if i in nodes_to_remove or not region_graph.has_node(i):
                pbar.update(1)
                continue

            component_i = region_graph.nodes[i]["component"]
            cluster_set_i = region_graph.nodes[i]["cluster_set"]

            for j in nodes_to_check:
                if i == j or j in nodes_to_remove or not region_graph.has_node(j):
                    continue

                cluster_set_j = region_graph.nodes[j]["cluster_set"]

                if cluster_set_i == cluster_set_j and are_touching(component_i, region_graph.nodes[j]["component"]):
                    component_i.update(region_graph.nodes[j]["component"])
                    region_graph.nodes[i]["component"] = component_i
                    for successor in list(region_graph.successors(j)):
                        region_graph.add_edge(i, successor)
                    for predecessor in list(region_graph.predecessors(j)):
                        region_graph.add_edge(predecessor, i)
                    nodes_to_remove.add(j)

            if len(nodes_to_remove) >= 10:
                process_removals()

            pbar.update(1)

        process_removals()

    return region_graph

def build_directed_graph(region_list):
    """
    Constructs a directed graph where nodes represent regions, 
    and edges represent subset relationships between cluster sets.

    Parameters:
    - region_list (list of dicts): List of regions where each region contains:
        - "component": Set of (x, y) points in the region.
        - "cluster_set": A set representing clusters.

    Returns:
    - region_graph (networkx.DiGraph): Directed graph where nodes are regions and edges indicate subset relationships.
    """
    region_graph = nx.DiGraph()

    # Add nodes to the graph
    for idx, region in enumerate(region_list):
        region_graph.add_node(idx, component=region["component"], cluster_set=region["cluster_set"])

    # Build directed edges
    with tqdm.tqdm(total=len(region_list), desc="Building graph") as pbar:
        for i, region_i in enumerate(region_list):
            component_i, cluster_set_i = region_i["component"], region_i["cluster_set"]

            for j, region_j in enumerate(region_list[i + 1:], start=i + 1):
                component_j, cluster_set_j = region_j["component"], region_j["cluster_set"]

                if are_touching(component_i, component_j):
                    # Add directed edges based on subset relationships
                    if cluster_set_i.issubset(cluster_set_j) and cluster_set_i != cluster_set_j:
                        region_graph.add_edge(i, j)
                    elif cluster_set_j.issubset(cluster_set_i) and cluster_set_i != cluster_set_j:
                        region_graph.add_edge(j, i)

            pbar.update(1)

    return region_graph


def add_edges_after_merge(region_graph):
    """
    Adds edges between regions after merging to maintain subset relationships.

    Parameters:
    - region_graph (networkx.DiGraph): Graph where nodes represent merged regions.

    Returns:
    - networkx.DiGraph: Updated graph with added edges where necessary.
    """
    with tqdm.tqdm(total=len(region_graph.nodes()), desc="Adding edges after merge") as pbar:
        nodes = list(region_graph.nodes())

        for i, node_i in enumerate(nodes):
            component_i = region_graph.nodes[node_i]["component"]
            cluster_set_i = region_graph.nodes[node_i]["cluster_set"]

            for node_j in nodes[i + 1:]:
                cluster_set_j = region_graph.nodes[node_j]["cluster_set"]

                # Avoid duplicate edges
                if region_graph.has_edge(node_i, node_j) or region_graph.has_edge(node_j, node_i):
                    continue

                # Add edges if cluster sets are subset-related
                if (cluster_set_i.issubset(cluster_set_j) or cluster_set_j.issubset(cluster_set_i)) and cluster_set_i != cluster_set_j:
                    component_j = region_graph.nodes[node_j]["component"]
                    if are_touching(component_i, component_j):
                        if len(cluster_set_i) < len(cluster_set_j):
                            region_graph.add_edge(node_i, node_j)
                        else:
                            region_graph.add_edge(node_j, node_i)

            pbar.update(1)

    return region_graph



def calculate_map_center(region_graph, padded_maps):
    """
    Calculate the center of the map based on the average location of all tokens.

    Parameters:
    - region_graph (networkx.Graph): Graph of clustered regions.
    - padded_maps (list): List of maps containing grid data.

    Returns:
    - tuple (lat, lon): The calculated center of the map.
    """
    all_points_x, all_points_y = [], []

    for node in region_graph.nodes:
        for (i, j) in region_graph.nodes[node]["component"]:
            x = padded_maps[0].grid.base_grid_x[i, j]
            y = padded_maps[0].grid.base_grid_y[i, j]
            if not np.isnan(x) and not np.isnan(y):
                all_points_x.append(x)
                all_points_y.append(y)

    if not all_points_x or all(np.isnan(all_points_x)) or all(np.isnan(all_points_y)):
        raise ValueError("No valid points found for calculating map center.")

    mean_x, mean_y = np.nanmean(all_points_x), np.nanmean(all_points_y)
    return transform.token_to_latlon(mean_x, mean_y)




def nodes_intersecting_point(region_graph, padded_maps, lat, lon):
    """
    Find all nodes in the region graph that intersect with a given lat/lon point.

    Parameters:
    - region_graph (networkx.Graph): The graph of clustered regions.
    - padded_maps (list): List of maps containing grid data.
    - lat (float): Latitude of the point.
    - lon (float): Longitude of the point.

    Returns:
    - list: Nodes that intersect the given coordinate.
    """
    x, y = transform.latlon_to_token(lat, lon)
    intersecting_nodes = []

    for node in region_graph.nodes:
        points = set(region_graph.nodes[node]["component"])
        boundary_contours = find_boundary_contours(
            points,
            padded_maps[0].grid.base_grid_x.shape,
            padded_maps[0].grid.base_grid_x,
            padded_maps[0].grid.base_grid_y
        )

        for contour in boundary_contours:
            polygon = Polygon(contour)
            if polygon.contains(Point(lat, lon)):
                intersecting_nodes.append(node)
                break

    return intersecting_nodes

def group_connected_components(region_graph, intersecting_nodes):
    """
    Group intersecting nodes into connected components.

    Parameters:
    - region_graph (networkx.Graph): The graph of clustered regions.
    - intersecting_nodes (list): List of nodes that intersect a given location.

    Returns:
    - list of sets: Groups of connected nodes.
    """
    node_groups = []
    visited = set()

    for node in intersecting_nodes:
        if node not in visited:
            connected_component = set()
            queue = [node]

            while queue:
                current_node = queue.pop()
                if current_node not in visited:
                    visited.add(current_node)
                    connected_component.add(current_node)
                    for neighbor in region_graph.neighbors(current_node):
                        if neighbor in intersecting_nodes and neighbor not in visited:
                            queue.append(neighbor)

            node_groups.append(connected_component)

    return node_groups

def find_boundary_contours(points, grid_shape, base_grid_x, base_grid_y):
    """
    Compute the boundary contours of clustered regions using the Marching Squares algorithm.

    Parameters:
    - points (set): Set of (i, j) points representing a region.
    - grid_shape (tuple): Shape of the base grid.
    - base_grid_x (numpy.ndarray): X-coordinates of the base grid.
    - base_grid_y (numpy.ndarray): Y-coordinates of the base grid.

    Returns:
    - list: List of boundary contours in lat/lon coordinates.
    """
    mask = np.zeros(grid_shape, dtype=bool)
    for i, j in points:
        mask[i, j] = 1

    contours = measure.find_contours(mask, level=0.5)
    boundary_contours = []

    for contour in contours:
        latlon_contour = [
            transform.token_to_latlon(base_grid_x[int(i), int(j)], base_grid_y[int(i), int(j)])
            for i, j in contour if not np.isnan(base_grid_x[int(i), int(j)]) and not np.isnan(base_grid_y[int(i), int(j)])
        ]
        if latlon_contour:
            boundary_contours.append(latlon_contour)

    return boundary_contours



def clean_geometry(geom):
    """
    Ensure that a geometry is valid and return a cleaned version.

    Parameters:
    - geom (Polygon, MultiPolygon, or GeometryCollection): The geometry to validate.

    Returns:
    - Valid geometry (Polygon, MultiPolygon, or None if invalid)
    """
    if not geom.is_valid:
        geom = make_valid(geom)  # Ensure validity

    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom  # Return cleaned Polygon or MultiPolygon

    elif isinstance(geom, GeometryCollection):
        # Extract only Polygon/MultiPolygon elements
        valid_polygons = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if valid_polygons:
            return MultiPolygon(valid_polygons) if len(valid_polygons) > 1 else valid_polygons[0]

    return None  # Return None if no valid geometry found
