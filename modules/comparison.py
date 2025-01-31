from dataclasses import dataclass
import pandas as pd
import numpy as np
from skimage.transform import estimate_transform
import rasterio
import matplotlib.pyplot as plt 
from scipy.interpolate import RBFInterpolator
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as col
from colorspacious import cspace_convert
from collections import deque
import cv2  # OpenCV library
from pyproj import Transformer
import folium
from shapely.geometry import Polygon, Point
import alphashape
import networkx as nx
import os
import time
from scipy.spatial.distance import euclidean







red = '#FB4C59'
blue = '#489FEE'
orange = '#EC802F'
green = '#3CBB39'
aqua = '#3CC5BE'
purple = '#7B60EE'
color_center = "#FFFFFF"  # White
color_dissimilar = purple
color_similar = aqua
positive_displacement = blue
negative_displacement = red

cmap_similarity = LinearSegmentedColormap.from_list("custom_divergent_cmap", [color_dissimilar, color_center, color_similar])
cmap_displacement = LinearSegmentedColormap.from_list("custom_divergent_cmap_2", [negative_displacement, color_center, positive_displacement])
cmap1= cmap_similarity
cmap2 = cmap_displacement


def create_map_object(map_info, grid_size):
    folder = map_info['folder']
    folder_path = map_info['folder_path']
    image_path = map_info['image_path']
    gcp_df = map_info['points']
    metadata = map_info['metadata']
    epsg = int(map_info['epsg'])
    base_x, base_y = 172119.73,1131710.35
    current_map = Map(name = folder, gcp_df= gcp_df, grid_size = grid_size, base_x = base_x, base_y = base_y, metadata = metadata, folder_path = folder_path, image_path = image_path, epsg = epsg)
    return current_map


def ratio_distance(vec1, vec2):
    """
    Computes the distance metric as the difference between the maximum 
    and minimum ratio value of two vectors.
    
    The denominator is chosen as the vector with the higher maximum value, 
    ensuring that the ratio is always <= 1.
    """
    max_1 = np.max(vec1)
    max_2 = np.max(vec2)
    ratio_vec = vec1 / vec2 if max_1 < max_2 else vec2 / vec1
    return np.max(ratio_vec) - np.min(ratio_vec)

def normalize_vector(vector):
    """
    Normalizes a vector to unit length. Returns the original vector if its norm is zero.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm 

def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.
    """
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_ratios_and_distances(vectors):
    """
    Computes pairwise ratio vectors and ratio distances for a set of vectors.

    Parameters:
    - vectors (ndarray): A (n, d) matrix where n is the number of vectors, and d is the vector dimension.

    Returns:
    - pairwise_ratios (ndarray): An (n, n, d) array containing ratio vectors for each pair (i, j).
    - pairwise_distances (ndarray): An (n, n) array containing ratio distances for each pair (i, j).
    """
    n = len(vectors)
    pairwise_ratios = np.zeros((n, n, vectors.shape[1]))
    pairwise_distances = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                ratio_vector = vectors[i] / vectors[j]
                pairwise_ratios[i, j] = ratio_vector
                pairwise_distances[i, j] = np.max(ratio_vector) - np.min(ratio_vector)

    return pairwise_ratios, pairwise_distances


### functions to compare grids
def process_grid(grid):
    base_grid_x = grid.base_grid_x
    base_grid_y = grid.base_grid_y
    
    # Initialize coordinates with NaN, will update with actual coordinates
    coordinates = np.full((base_grid_x.shape[0], base_grid_x.shape[1], 2), np.nan)  # Assuming 2D coordinates

    # Populate the coordinates
    coordinates[:, :, 0] = base_grid_x
    coordinates[:, :, 1] = base_grid_y

    return coordinates

def find_bounding_box(grids):
    min_x, min_y = np.inf, np.inf
    max_x, max_y = -np.inf, -np.inf

    for grid in grids:
        coordinates = process_grid(grid)
        valid_coords = coordinates[~np.isnan(coordinates[:, :, 0])]

        if valid_coords.size > 0:
            min_x = min(min_x, valid_coords[:, 0].min())
            max_x = max(max_x, valid_coords[:, 0].max())
            min_y = min(min_y, valid_coords[:, 1].min())
            max_y = max(max_y, valid_coords[:, 1].max())


    return min_x, max_x, min_y, max_y

def pad_grid_and_deformed(grid, token_vectors, min_x, max_x, min_y, max_y):
    base_grid_x = grid.base_grid_x
    base_grid_y = grid.base_grid_y
    deformed_grid_x = grid.graph.deformed_grid_x
    deformed_grid_y = grid.graph.deformed_grid_y

    # Calculate the padding required
    dx = np.nanmean(np.diff(base_grid_x, axis=1))
    dy = np.nanmean(np.diff(base_grid_y, axis=0))

    if np.isnan(dx) or dx == 0:
        dx = 1
    if np.isnan(dy) or dy == 0:
        dy = 1

    new_cols = int(np.round((max_x - min_x) / dx)) + 1
    new_rows = int(np.round((max_y - min_y) / dy)) + 1

    # Create new padded grids filled with NaNs
    new_base_grid_x = np.full((new_rows, new_cols), np.nan)
    new_base_grid_y = np.full((new_rows, new_cols), np.nan)
    new_token_vectors = np.full((new_rows, new_cols, token_vectors.shape[2]), np.nan)
    new_deformed_grid_x = np.full((new_rows, new_cols), np.nan)
    new_deformed_grid_y = np.full((new_rows, new_cols), np.nan)

    row_offset = int((base_grid_y[0, 0] - min_y) / dy)
    col_offset = int((base_grid_x[0, 0] - min_x) / dx)

    old_rows, old_cols = base_grid_x.shape
    target_row_slice = slice(row_offset, row_offset + old_rows)
    target_col_slice = slice(col_offset, col_offset + old_cols)

    new_base_grid_x[target_row_slice, target_col_slice] = base_grid_x
    new_base_grid_y[target_row_slice, target_col_slice] = base_grid_y
    new_token_vectors[target_row_slice, target_col_slice, :] = token_vectors
    new_deformed_grid_x[target_row_slice, target_col_slice] = deformed_grid_x
    new_deformed_grid_y[target_row_slice, target_col_slice] = deformed_grid_y

    return new_base_grid_x, new_base_grid_y, new_token_vectors, new_deformed_grid_x, new_deformed_grid_y


def pad_grid_and_vectors(grid, token_vectors, min_x, max_x, min_y, max_y):
    base_grid_x = grid.base_grid_x
    base_grid_y = grid.base_grid_y

    # Calculate the padding required
    dx = np.nanmean(np.diff(base_grid_x, axis=1))
    dy = np.nanmean(np.diff(base_grid_y, axis=0))

    if np.isnan(dx) or dx == 0:
        dx = 1
    if np.isnan(dy) or dy == 0:
        dy = 1

    new_cols = int(np.round((max_x - min_x) / dx)) + 1
    new_rows = int(np.round((max_y - min_y) / dy)) + 1

    new_grid_x = np.full((new_rows, new_cols), np.nan)
    new_grid_y = np.full((new_rows, new_cols), np.nan)
    new_token_vectors = np.full((new_rows, new_cols, token_vectors.shape[2]), np.nan)

    row_offset = int((base_grid_y[0, 0] - min_y) / dy)
    col_offset = int((base_grid_x[0, 0] - min_x) / dx)

    old_rows, old_cols = base_grid_x.shape
    target_row_slice = slice(row_offset, row_offset + old_rows)
    target_col_slice = slice(col_offset, col_offset + old_cols)

    new_grid_x[target_row_slice, target_col_slice] = base_grid_x
    new_grid_y[target_row_slice, target_col_slice] = base_grid_y
    new_token_vectors[target_row_slice, target_col_slice, :] = token_vectors

    return new_grid_x, new_grid_y, new_token_vectors

def get_deviation(token_vectors):
    # Define the baseline vector and compute its norm
    baseline_vector = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)])
    baseline_norm = np.linalg.norm(baseline_vector)
    
    # Compute the norms of the token vectors
    original_norms = np.linalg.norm(token_vectors, axis=2)
    
    # Flatten the norms and remove NaN values for RANSAC fitting
    flat_original_norms = original_norms.flatten()
    valid_mask = ~np.isnan(flat_original_norms)
    valid_norms = flat_original_norms[valid_mask]
    
    # Find the optimal scaling factor F
    L = baseline_norm
    sum_d = np.sum(valid_norms)
    sum_d_squared = np.sum(valid_norms ** 2)
    
    scaling_factor = (L * sum_d) / sum_d_squared
    
    # Scale the token vectors
    scaled_token_vectors = token_vectors * scaling_factor
    
    # Compute the norms of the scaled token vectors
    scaled_norms = np.linalg.norm(scaled_token_vectors, axis=2)
    
    # Compute the deviation from the baseline norm
    deviation_from_baseline = scaled_norms - baseline_norm
    
    return deviation_from_baseline

def plot_padded_grid_old(padded_grid_x, padded_grid_y, token_vectors, min_x, max_x, min_y, max_y):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    norms = np.linalg.norm(token_vectors, axis=2)
    
    # Flatten the arrays for plotting
    flat_x = padded_grid_x.flatten()
    flat_y = padded_grid_y.flatten()
    flat_norms = norms.flatten()
    
    # Create a mask for NaN values
    nan_mask = np.isnan(flat_norms)
    
    
    # Plot all non-NaN points with the norm values
    sc = ax.scatter(flat_x[~nan_mask], flat_y[~nan_mask], c=flat_norms[~nan_mask], cmap='viridis', alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.colorbar(sc, label='Token Vector Norm')
    
    # Set axis limits based on the bounding box
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Padded Grid with Token Vector Norms')
    plt.show()





#####
def calculate_iqr(series):
    return series.quantile(0.75) - series.quantile(0.25)



def create_gcp_list_from_df(gcp_df):
    gcp_list = []
    for index, row in gcp_df.iterrows():
        # Check if the point is enabled (column "enable")
        if row['enable'] == 1:
            gcp_list.append((index, row['mapX'], row['mapY'], row['sourceX'], row['sourceY'], row['enable'], row['dX'], row['dY'], row['residual']))
    return gcp_list

def create_grid(bounds, base_x, base_y, grid_size): 
    #todo: for the moment this function is used both by the field class and by the pair, fix it and then put it inside the field dataclass
    try:
        x_min, x_max = bounds.left, bounds.right
        y_min, y_max = bounds.bottom, bounds.top
    except:
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        
    # Calculate how many grid points we need to extend from the base point to each boundary
    grid_points_left = int(np.ceil((base_x - x_min) / grid_size))
    grid_points_right = int(np.ceil((x_max - base_x) / grid_size))
    grid_points_bottom = int(np.ceil((base_y - y_min) / grid_size))
    grid_points_top = int(np.ceil((y_max - base_y) / grid_size))

    # Calculate the actual start and end points of the grid
    start_x = base_x - grid_points_left * grid_size
    end_x = base_x + grid_points_right * grid_size
    start_y = base_y - grid_points_bottom * grid_size
    end_y = base_y + grid_points_top * grid_size

    # Generate the grid
    x_grid, y_grid = np.meshgrid(
        np.arange(start_x, end_x, grid_size),
        np.arange(start_y, end_y, grid_size)
    )
    return x_grid, y_grid

def create_displacement_vectors(aligned_map_points, real_points, x_grid, y_grid):
    """ todo: same as the previous function: make it so that it will be used just by the field dataclass and insert inside """
    # Combine aligned_map_points into a 2D array
    y = np.column_stack((aligned_map_points[:, 0], aligned_map_points[:, 1]))

    # Displacement in x direction
    dx = aligned_map_points[:, 0] - real_points[:, 0]
    rbf_dx = RBFInterpolator(y, dx[:, np.newaxis], kernel='thin_plate_spline')

    # Displacement in y direction
    dy = aligned_map_points[:, 1] - real_points[:, 1]
    rbf_dy = RBFInterpolator(y, dy[:, np.newaxis], kernel='thin_plate_spline')

    # Create a grid for interpolation
    grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    # Calculate the displacement vectors using Rbf functions
    displacement_dx = rbf_dx(grid_points).reshape(x_grid.shape)
    displacement_dy = rbf_dy(grid_points).reshape(y_grid.shape)

    return displacement_dx, displacement_dy

def extract_epsg(crs_info_str):
    try:
        # Search for EPSG code in the crs_info string
        epsg_code_start = crs_info_str.find('ID["EPSG".6')
        epsg_code_end = crs_info_str.find(']]', epsg_code_start)
        epsg_code = crs_info_str[epsg_code_start:epsg_code_end].split("'")[2].strip()
        return epsg_code
    except Exception as e:
        print(f"Error extracting EPSG code: {e}")
        return '28193'

def find_alpha_shape(points, alpha=None):
    """
    Create an alpha shape (concave hull) of the given points. If the resulting
    alpha shape is a MultiPolygon or does not contain all points, return the convex hull instead.

    Parameters:
    - points: A list of (x, y) tuples representing the points.
    - alpha: Optional. A parameter that controls the tightness of the hull around the points.
             If None, an optimal alpha value will be determined automatically.

    Returns:
    - A single polygon that represents the alpha shape or the convex hull of the given points.
    """
    # Create the alpha shape
    alpha_shape = alphashape.alphashape(points, alpha)
    
    # Check if the alpha_shape is a single Polygon and contains all points
    if isinstance(alpha_shape, Polygon) and all(alpha_shape.contains(Point(p)) or alpha_shape.touches(Point(p)) for p in points):
        result_shape = alpha_shape
    else:
        # Compute the convex hull if alpha_shape is not a single Polygon or does not contain all points
        result_shape = alphashape.alphashape(points, 0)

    
    return result_shape

@dataclass
class Grid:
    """ Class to compute and store the undeformed grid of a map """
    gcp_df: pd.DataFrame
    grid_size: int
    base_x: int
    base_y: int
    real_points: np.array = None
    pixel_points: np.array = None
    bounds: tuple = None
    alpha_shape: Polygon = None
    alpha_shape_mask : np.array = None
    base_grid_x: np.array = None
    base_grid_y: np.array = None

    def __post_init__(self):
        # Estimating affine transformation and bounding box
        real_points = self.get_real_points() 
        self.get_pixel_points() 
        if self.bounds is None:
            min_x = min(real_points, key=lambda p: p[0])[0]
            max_x = max(real_points, key=lambda p: p[0])[0]
            min_y = min(real_points, key=lambda p: p[1])[1]
            max_y = max(real_points, key=lambda p: p[1])[1]

            self.bounds = rasterio.coords.BoundingBox(left=min_x, bottom=min_y, right=max_x, top=max_y)
        self.get_grid()
        self.get_alpha_shape()
        self.get_alpha_shape_mask()

    def get_real_points(self):
        """gets the real points from the mapX and mapY columns of the df"""
        if self.real_points is None:
            real_points = np.array(self.gcp_df[['mapX', 'mapY']])
            self.real_points = real_points
        else:
            real_points = self.real_points
        return real_points
    
    def get_pixel_points(self):
        """gets the pixel points from sourceX and sourceY columns of the df"""
        if self.pixel_points is None:
            self.pixel_points = np.array(self.gcp_df[['sourceX', 'sourceY']])
        return self.pixel_points

    def get_alpha_shape(self, alpha=0.001):
        """Gets the alpha shape of the real_points."""
        if self.alpha_shape is None:
            self.alpha_shape = find_alpha_shape(self.real_points, alpha)
        
    def get_alpha_shape_mask(self):
        base_grid_x, base_grid_y = self.base_grid_x, self.base_grid_y 
        mask_inside_alpha_shape = np.zeros_like(base_grid_x, dtype=bool)
        for i in range(base_grid_x.shape[0]):
            for j in range(base_grid_x.shape[1]):
                point = Point(base_grid_x[i, j], base_grid_y[i, j])
                if self.alpha_shape.contains(point):
                    mask_inside_alpha_shape[i, j] = True
        self.alpha_shape_mask = mask_inside_alpha_shape

    def get_grid(self):
        if self.base_grid_x is None:
            base_grid_x, base_grid_y = create_grid(self.bounds, self.base_x, self.base_y, self.grid_size)
            self.base_grid_x, self.base_grid_y = base_grid_x, base_grid_y

    ### plot functions
    def plot_alpha_shape(self):
        """Plots the alpha shape of the real points."""
        alpha_shape = self.alpha_shape
        if alpha_shape is not None:
            x,y = alpha_shape.exterior.xy
            plt.figure()
            plt.plot(x, y, color='#6699cc', alpha=0.7,
                     linewidth=3, solid_capstyle='round', zorder=2)
            plt.fill(x, y, color='#6699cc', alpha=0.3)
            plt.scatter(self.real_points[:,0], self.real_points[:,1], color="red")
            plt.title('Alpha Shape of Affine Aligned Pixel Points')
            plt.show()
        else:
            print("No alpha shape to plot.")

@dataclass
class Graph:
    """ Class to compute and store the geometric deformations of a grid"""
    deformed_grid_x: np.array = None
    deformed_grid_y: np.array = None
    token_vectors : np.array = None
    alpha_masked_token_vectors : np.array = None

    def __init__(self, grid):
        self.deformed_grid_x = None
        self.deformed_grid_y = None
        self.token_vectors = None
        self.alpha_masked_token_vectors = None
        self.__post_init__(grid)

    def __post_init__(self, grid):
        # using a Grid object, perform a thin plate spline interpolation of the base_grid_x and base_grid_y to the deformed_grid_x and deformed_grid_y using real_points to pixel_points as control points
        base_grid_x = grid.base_grid_x
        base_grid_y = grid.base_grid_y
        base_grid = np.stack((base_grid_x, base_grid_y), axis=-1)
        gcp_df = grid.gcp_df
        alpha_shape_mask = grid.alpha_shape_mask

        transformed_grid = self.apply_rbf_to_grid(gcp_df, base_grid)
        self.deformed_grid_x = transformed_grid[:, :, 0]
        self.deformed_grid_y = transformed_grid[:, :, 1]
        self.token_vectors = self.compute_token_vectors(transformed_grid)
        self.alpha_masked_token_vectors = self.token_vectors.copy()
        self.alpha_masked_token_vectors[~alpha_shape_mask] = np.nan
    
    def apply_rbf_to_grid(self, gcp_df, grid):
        # Extract real-world and pixel coordinates from GCPs
        real_world_x = gcp_df['mapX'].values
        real_world_y = gcp_df['mapY'].values
        pixel_x = gcp_df['sourceX'].values
        pixel_y = -gcp_df['sourceY'].values  # invert the y axis for image coordinates

        # Initialize RBF interpolator
        rbf = RBFInterpolator(np.column_stack((real_world_x, real_world_y)),
                              np.column_stack((pixel_x, pixel_y)),
                              kernel='thin_plate_spline',
                              smoothing=100, # ,
                              neighbors=70)

        # Initialize a new grid for transformed coordinates
        transformed_grid = np.zeros_like(grid)
        # Apply RBF to each point in the grid
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                geo_x, geo_y = grid[y, x]
                # Apply RBF interpolation to transform to pixel space
                pixel_location = rbf(np.array([[geo_x, geo_y]]))
                # Store the transformed location
                transformed_grid[y, x] = pixel_location[0]
        
        return transformed_grid

    def compute_token_vectors(self, transformed_grid):
        token_vectors = np.full((transformed_grid.shape[0], transformed_grid.shape[1], 16), np.nan) #changed from 16 to 20
        for i in range(transformed_grid.shape[0]):
            for j in range(transformed_grid.shape[1]):
                token_vectors[i, j] = self.compute_lengths(i, j, transformed_grid)
        return token_vectors

    def compute_lengths(self, i, j, grid):
        # Directions for 8 neighbors
        directions = [
            (0, 1),  (0, -1), (1, 0), (-1, 0),  
            (1, 1),  (-1, -1), (1, -1), (-1, 1)  
        ]
        center = grid[i, j]
        lengths = []

        # Compute lengths to 8 surrounding neighbors
        neighbors = []
        for direction in directions:
            neighbor = self.get_neighbor(i + direction[0], j + direction[1], grid)
            if neighbor is not None:
                lengths.append(euclidean(center, neighbor))
                neighbors.append(neighbor)
            else:
                lengths.append(np.nan)
                neighbors.append(None)

        lengths_1 = lengths[:4]
        lengths_2 = lengths[4:]

        # Compute lengths for the 12 neighbor-to-neighbor distances
        neighbor_pairs = [ 
            (0, 4), (4,2), (2,6), (6,1),  
            (1,5), (5,3), (3,7), (7,0) # ,
            # (0, 1), (1, 2), (2, 3), (3, 0),    
        ]
        for (n1, n2) in neighbor_pairs:
            if neighbors[n1] is not None and neighbors[n2] is not None:
                lengths_1.append(euclidean(neighbors[n1], neighbors[n2]))
            else:
                lengths_1.append(np.nan)

        
        for leng in lengths_2:
            lengths_1.append(leng)
        

        lengths_final = lengths_1

        # get the lengths of the diagonals connecting the 4 neighbors in direction up, right, down, left
        #neighbor_pairs_diagonals = [
        #    (0, 2), (2, 1), (1, 3), (3, 0)
        #]
        #for (n1, n2) in neighbor_pairs_diagonals:
        #    if neighbors[n1] is not None and neighbors[n2] is not None:
        #        lengths_final.append(euclidean(neighbors[n1], neighbors[n2]))
        #    else:
        #        lengths_final.append(np.nan)
        
    
        if len(lengths_final) != 16: #changed from 16 to 20
            raise ValueError(f"Lengths array has incorrect size: {len(lengths_final)} instead of 20")

        return lengths_final

    def get_neighbor(self, i, j, grid):
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            return grid[i, j]
        return None


    def plot_token_vector_norms(self):
        grid_x = self.deformed_grid_x
        grid_y = self.deformed_grid_y
        token_vectors = self.alpha_masked_token_vectors
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
    
        num_rows, num_cols = grid_x.shape
        norms = np.linalg.norm(token_vectors, axis=2)  # Compute the norm of each token vector

        sc = ax.scatter(grid_x, grid_y, c=norms, cmap='viridis', s=50)
        plt.colorbar(sc, label='Token Vector Norm')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Token Vector Norms Visualization')
        plt.show()


@dataclass
class Map:
    """Class to store a map info and its deformative field"""
    name: str
    gcp_df: pd.DataFrame
    grid_size: int
    base_x: int
    base_y: int
    metadata: dict
    folder_path: str
    image_path: str
    grid : Grid = None
    epsg: int = None

    def __post_init__(self):
        # Estimating affine transformation and bounding box
        self.grid = Grid(self.gcp_df, self.grid_size, self.base_x, self.base_y)
        #this can be used to create a field, decide how to create it: self.field = Field(grid)

    def plot_affine_transformation(self):
        fig, ax = plt.subplots(figsize=(10,10))
        plt.scatter(self.field.real_points[:,0], self.field.real_points[:,1], c='r', label='real points')
        plt.scatter(self.field.affine_aligned_pixel_points[:,0], (self.field.affine_aligned_pixel_points)[:,1], c='b', label='transformed points (affine)')
        ax.legend()
        plt.show()


@dataclass 
class MapPair:
    original_grid_1: Grid
    original_grid_2: Grid
    original_map_1: Map = None # inserted if available by the user
    original_map_2: Map = None # inserted if available by the user
    common_extent: tuple = None # (min_x, max_x, min_y, max_y)
    common_gcp_df_1: pd.DataFrame = None
    common_gcp_df_2: pd.DataFrame = None
    cropped_grid_1: Grid = None
    cropped_grid_2: Grid = None
    common_alpha_shape_mask : np.array = None

    def __post_init__(self):
        #we check that the grid size is the same and that the base point is the same as well, otherwise we calculate again the grid for one of the two
        if self.original_grid_1.grid_size != self.original_grid_2.grid_size:
            if self.original_grid_1.base_x != self.original_grid_2.base_x or self.original_grid_1.base_y != self.original_grid_2.base_y:
                print('we need to calculate one of the two maps with the other ones grid size and base point TODO')
            else:
                print('we need to calculate one of the two with the other map gridsize')
        self.get_common_gcp_df()


        start = time.time()
        self.get_cropped_grids()
        self.common_extent = self.cropped_grid_1.bounds.left, self.cropped_grid_1.bounds.right, self.cropped_grid_1.bounds.bottom, self.cropped_grid_1.bounds.top

        end = time.time()
        self.get_common_alpha_shape_mask()

        if False:
            print(f"Time elapsed: {end - start} seconds")

    def find_common_extent(self, cropped_base_grid_1_x, cropped_base_grid_1_y, cropped_base_grid_2_x, cropped_base_grid_2_y):

        min_x_1, max_x_1, min_y_1, max_y_1  = cropped_base_grid_1_x.min(),cropped_base_grid_1_x.max(), cropped_base_grid_1_y.min(), cropped_base_grid_1_y.max()
        min_x_2, max_x_2, min_y_2, max_y_2  = cropped_base_grid_2_x.min(), cropped_base_grid_2_x.max(), cropped_base_grid_2_y.min(), cropped_base_grid_2_y.max()

        # Check that there is an overlap
        if (min_x_1 > max_x_2 or min_x_2 > max_x_1) or (min_y_1 > max_y_2 or min_y_2 > max_y_1):
            raise ValueError('The two fields do not overlap')
        
        # Find the common extent
        min_x = max(min_x_1, min_x_2)
        max_x = min(max_x_1, max_x_2)
        min_y = max(min_y_1, min_y_2)
        max_y = min(max_y_1, max_y_2)

        self.common_extent = min_x, max_x, min_y, max_y

    
    def find_common_extent_gcp_df(self, gcp_df_1, gcp_df_2):
        min_x_1, max_x_1 = gcp_df_1['mapX'].min(), gcp_df_1['mapX'].max()
        min_y_1, max_y_1 = gcp_df_1['mapY'].min(), gcp_df_1['mapY'].max()

        min_x_2, max_x_2 = gcp_df_2['mapX'].min(), gcp_df_2['mapX'].max()
        min_y_2, max_y_2 = gcp_df_2['mapY'].min(), gcp_df_2['mapY'].max()
        # Check that there is an overlap
        if (min_x_1 > max_x_2 or min_x_2 > max_x_1) or (min_y_1 > max_y_2 or min_y_2 > max_y_1):
            raise ValueError('The two sets of cropped GCPs do not overlap')

        # Find the common extent
        min_x = max(min_x_1, min_x_2)
        max_x = min(max_x_1, max_x_2)
        min_y = max(min_y_1, min_y_2)
        max_y = min(max_y_1, max_y_2)

        self.common_extent = min_x, max_x, min_y, max_y


    def get_common_alpha_shape_mask(self):
        alpha_shape_mask_1 = self.cropped_grid_1.alpha_shape_mask
        alpha_shape_mask_2 = self.cropped_grid_2.alpha_shape_mask

        # Combine the two masks by using the logical AND operation
        common_alpha_shape_mask = alpha_shape_mask_1 & alpha_shape_mask_2
        self.common_alpha_shape_mask = common_alpha_shape_mask
    
    def crop_gcp_df(self, gcp_df, min_x, max_x, min_y, max_y):
        # Return the cropped GCP data frame based on given bounds
        return gcp_df[
            (gcp_df['mapX'] >= min_x) & (gcp_df['mapX'] <= max_x) &
            (gcp_df['mapY'] >= min_y) & (gcp_df['mapY'] <= max_y)
        ]
    
    def get_common_gcp_df(self):
        # Access the GCP data frames
        gcp_df_1 = self.original_grid_1.gcp_df
        gcp_df_2 = self.original_grid_2.gcp_df

        # Compute common mapX and mapY extents by finding the overlapping regions
        min_x = max(gcp_df_1['mapX'].min(), gcp_df_2['mapX'].min())
        max_x = min(gcp_df_1['mapX'].max(), gcp_df_2['mapX'].max())
        min_y = max(gcp_df_1['mapY'].min(), gcp_df_2['mapY'].min())
        max_y = min(gcp_df_1['mapY'].max(), gcp_df_2['mapY'].max())

        # Crop both data frames to the common extents
        cropped_gcp_df1 = self.crop_gcp_df(gcp_df_1, min_x, max_x, min_y, max_y)
        cropped_gcp_df2 = self.crop_gcp_df(gcp_df_2, min_x, max_x, min_y, max_y)

        # Assign the cropped data frames to the respective attributes
        self.common_gcp_df_1 = cropped_gcp_df1
        self.common_gcp_df_2 = cropped_gcp_df2
        self.common_extent = min_x, max_x, min_y, max_y

    def get_cropped_array(self, array, keep_x_indices, keep_y_indices): #unused?
        cropped_array = array[keep_y_indices, :][:, keep_x_indices]
        return cropped_array
    
    def crop_to_common_extent(self, grid):
        min_x, max_x, min_y, max_y = self.common_extent
        # Generate boolean arrays for rows and columns to keep
        keep_rows = (grid.base_grid_y >= min_y) & (grid.base_grid_y <= max_y)
        # Columns to keep: X-coordinates within [min_x, max_x]
        keep_columns = (grid.base_grid_x >= min_x) & (grid.base_grid_x <= max_x)
        # Ensure we're checking along the correct axis for both rows and colum
        
        keep_rows = np.any(keep_rows, axis=1)  # Any row with Y within bounds
        keep_columns = np.any(keep_columns, axis=0)  # Any column with X within bounds
        # Apply row and column masks to crop
        cropped_base_grid_x = grid.base_grid_x[keep_rows][:, keep_columns]
        cropped_base_grid_y = grid.base_grid_y[keep_rows][:, keep_columns]
        grid.base_grid_x = cropped_base_grid_x
        grid.base_grid_y = cropped_base_grid_y
        grid.get_alpha_shape_mask()
        return grid

    def get_cropped_grids(self):
        min_x, max_x, min_y, max_y = self.common_extent
        bounds = rasterio.coords.BoundingBox(left=min_x, bottom=min_y, right=max_x, top=max_y)
        self.cropped_grid_1 = Grid(self.common_gcp_df_1, self.original_grid_1.grid_size, self.original_grid_1.base_x, self.original_grid_1.base_y, bounds = bounds)
        self.cropped_grid_2 = Grid(self.common_gcp_df_2, self.original_grid_2.grid_size, self.original_grid_2.base_x, self.original_grid_2.base_y, bounds = bounds)


@dataclass
class Field:
    """Class to store a deformative field"""
    grid: Grid
    aligned_pixel_points: np.array = None  #this is probably the one taking a lot of time
    displacement_x: np.array = None
    displacement_y: np.array = None
    deformed_x: np.array = None
    deformed_y: np.array = None
    intensity_mask: np.array = None

    def __post_init__(self):
        # Estimating affine transformation and bounding box
        if self.displacement_x is None:
            self.get_aligned_pixel_points()
            self.get_deformed_grid()  
            self.get_intensity_mask()

    def get_aligned_pixel_points(self): # this could be moved in the field 
        """aligns the pixel points"""
        if self.aligned_pixel_points is None:
            transform = estimate_transform('polynomial', self.grid.pixel_points, self.grid.real_points, 2) 
            self.aligned_pixel_points = transform(self.grid.pixel_points)
        return self.aligned_pixel_points
    
    def get_deformed_grid(self):

        self.displacement_x, self.displacement_y = create_displacement_vectors(self.aligned_pixel_points, self.grid.real_points, self.grid.base_grid_x, self.grid.base_grid_y)
        self.deformed_x = self.grid.base_grid_x + self.displacement_x
        self.deformed_y = self.grid.base_grid_y + self.displacement_y 

    def get_intensity_mask(self, intensity_threshold=0.1):
        displacement_x_filtered = np.copy(self.displacement_x)
        displacement_y_filtered = np.copy(self.displacement_y)
        mask = self.grid.alpha_shape_mask
        displacement_x_filtered[~mask] = np.nan
        displacement_y_filtered[~mask] = np.nan
        magnitudes = np.sqrt(displacement_x_filtered**2 + displacement_y_filtered**2) #calculated over the points in the alpha shape
        quartile = np.nanquantile(magnitudes, intensity_threshold)
        self.intensity_mask = magnitudes >= quartile
## plotting functions

    def plot_affine_transformation(self):
        fig, ax = plt.subplots(figsize=(10,10))
        plt.scatter(self.grid.real_points[:,0], self.grid.real_points[:,1], c='r', label='real points')
        plt.scatter(self.aligned_pixel_points[:,0], (self.aligned_pixel_points)[:,1], c='b', label='transformed points (affine)')
        ax.legend()
        plt.show()

    def plot_vector_field(self):
        #plot the two vector fields
        X_component, Y_component =self.displacement_x, self.displacement_y
        X, Y = np.meshgrid(np.arange(X_component.shape[1]), np.arange(Y_component.shape[0]))
        # Correct the magnitude calculation
        color_circle = np.ones((256,3))*60
        color_circle[:,1] = np.ones((256))*45
        color_circle[:,2] = np.arange(0,360,360/256)
        color_circle_rgb = cspace_convert(color_circle, "JCh", "sRGB1")
        cm = col.ListedColormap(color_circle_rgb)
        
        direction = np.arctan2(Y_component, X_component)
        direction_norm = (direction + np.pi) / (2 * np.pi)
        colors = cm(direction_norm.flatten())
        plt.quiver(X, Y, X_component, Y_component, scale = 800, width = 0.002, color=colors)
        #remove axis
        plt.xticks([])
        plt.yticks([])
        #set aspect to equal
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        plt.close()

    def plot_vector_field_and_affine_transformation(self):
        fig, ax = plt.subplots(figsize=(10,10))
        plt.scatter(self.grid.real_points[:,0], self.grid.real_points[:,1], c='r', label='real points')
        plt.scatter(self.grid.affine_aligned_pixel_points[:,0], (self.grid.affine_aligned_pixel_points)[:,1], c='b', label='transformed points (affine)')
        
        base_grid_x, base_grid_y = self.grid.base_grid_x, self.grid.base_grid_y
        #plot a grid that has the starting point in base_grid_x, base_grid_y and the displacement in displacement_x, displacement_y
        X_component, Y_component = self.displacement_x, self.displacement_y
        # Correct the magnitude calculation
        color_circle = np.ones((256,3))*60
        color_circle[:,1] = np.ones((256))*45
        color_circle[:,2] = np.arange(0,360,360/256)
        color_circle_rgb = cspace_convert(color_circle, "JCh", "sRGB1")
        cm = col.ListedColormap(color_circle_rgb)
        
        direction = np.arctan2(Y_component, X_component)
        direction_norm = (direction + np.pi) / (2 * np.pi)
        colors = cm(direction_norm.flatten())
        plt.quiver(base_grid_x, base_grid_y, X_component, Y_component, scale = 600, width = 0.001, color=colors)
        ax.legend()
        plt.show()

    def plot_deformed_grid_with_points(self):
        fig, ax = plt.subplots(figsize=(10,10))
               
        alpha_shape = self.grid.alpha_shape
        if alpha_shape is not None:
            x,y = alpha_shape.exterior.xy
            
            plt.plot(x, y, color='#6699cc', alpha=0.7,
                     linewidth=3, solid_capstyle='round', zorder=2)
            plt.fill(x, y, color='#6699cc', alpha=0.3)
        else:
            print("No alpha shape to plot.")
        # Plot real points and affine-aligned points
        plt.scatter(self.grid.real_points[:,0], self.grid.real_points[:,1], c='r', label='real points')
        plt.scatter(self.grid.affine_aligned_pixel_points[:,0], self.grid.affine_aligned_pixel_points[:,1], c='b', label='transformed points (affine)')
        
        # Generate the base grid
        base_grid_x, base_grid_y = self.grid.base_grid_x, self.grid.base_grid_y
        
        # Apply displacement to the grid points to get the deformed grid
        deformed_grid_x = base_grid_x + self.displacement_x
        deformed_grid_y = base_grid_y + self.displacement_y
        
        # Plot the original grid (optional, for reference)
        for i in range(base_grid_x.shape[0]):
            plt.plot(base_grid_x[i, :], base_grid_y[i, :], 'k-', linewidth=0.5, alpha=0.3)
        for j in range(base_grid_x.shape[1]):
            plt.plot(base_grid_x[:, j], base_grid_y[:, j], 'k-', linewidth=0.5, alpha=0.3)
        
        # Plot the deformed grid
        for i in range(deformed_grid_x.shape[0]):
            plt.plot(deformed_grid_x[i, :], deformed_grid_y[i, :], 'g-', linewidth=1)
        for j in range(deformed_grid_x.shape[1]):
            plt.plot(deformed_grid_x[:, j], deformed_grid_y[:, j], 'g-', linewidth=1)
        
        ax.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()



@dataclass 
class GraphPair:
    map_pair: MapPair #we should not store the map pair, but just the two grids
    common_grid_map_1 : np.ndarray = None
    common_grid_map_2 : np.ndarray = None
    base_grid_nodes_edges : tuple = None 
    deformed_grid_1_nodes_edges : tuple = None 
    deformed_grid_2_nodes_edges : tuple = None 
    ratios_seeds : tuple = None 
    common_combined_mask : np.array = None # for now we directly don't store the two intensity masks but just their combination.
    name_1 : str = None
    name_2 : str = None
    def __post_init__(self):
        #self.project_common_grid()  # this is not needed, unless we want to have the grid in the epsg 4362 for folium
        self.project_common_grid_to_map_space()

        start = time.time()
        self.get_base_grid_nodes_edges()
        end = time.time()
        if False:
            print(f"Time elapsed get_base_grid_nodes_edges: {end - start} seconds")

        start = time.time()    
        self.get_deformed_grid_nodes_edges()
        end = time.time()
        if False:
            print(f"Time elapsed get_deformed_grid_nodes_edges: {end - start} seconds")
        
        start = time.time()
        self.get_ratio_graph_from_dicts()
        end = time.time()
        if False:
            print(f"Time elapsed get_ratio_graph_from_dicts: {end - start} seconds")
        start = time.time()
        self.get_combined_mask()
        end = time.time()   
        if False:    
            print(f"Time elapsed get_combined_mask: {end - start} seconds")
        
        # delete the map pair
        self.name_1 = self.map_pair.original_map_1.name
        self.name_2 = self.map_pair.original_map_2.name


    def apply_rbf_to_grid(self, gcp_df, grid):
        # Extract real-world and pixel coordinates from GCPs
        real_world_x = gcp_df['mapX'].values
        real_world_y = gcp_df['mapY'].values
        pixel_x = gcp_df['sourceX'].values
        pixel_y = -gcp_df['sourceY'].values  # invert the y axis for image coordinates

        # Initialize RBF interpolator
        rbf = RBFInterpolator(np.column_stack((real_world_x, real_world_y)),
                              np.column_stack((pixel_x, pixel_y)),
                              kernel='thin_plate_spline')

        # Initialize a new grid for transformed coordinates
        transformed_grid = np.zeros_like(grid)
        # Apply RBF to each point in the grid
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                geo_x, geo_y = grid[y, x]
                # Apply RBF interpolation to transform to pixel space
                pixel_location = rbf(np.array([[geo_x, geo_y]]))
                # Store the transformed location
                transformed_grid[y, x] = pixel_location[0]
        
        return transformed_grid

    def project_common_grid_to_map_space(self):
        # Apply RBF to the local EPSG grid for both maps
        cropped_grid_1 = self.map_pair.cropped_grid_1
        cropped_grid_2 = self.map_pair.cropped_grid_2
        common_grid_local_epsg = np.stack((cropped_grid_1.base_grid_x, cropped_grid_1.base_grid_y), axis=-1)
        self.common_grid_map_1 = self.apply_rbf_to_grid(cropped_grid_1.gcp_df, common_grid_local_epsg)
        self.common_grid_map_2 = self.apply_rbf_to_grid(cropped_grid_2.gcp_df, common_grid_local_epsg)

    def get_base_grid_nodes_edges(self):
        #base grid graph
        grid = np.stack((self.map_pair.cropped_grid_1.base_grid_x, self.map_pair.cropped_grid_1.base_grid_y), axis=-1)
        nodes, edges = self.create_grid_nodes_edges(grid)
        self.base_grid_nodes_edges= nodes, edges
        return nodes, edges
        
    def get_deformed_grid_nodes_edges(self):
        #deformed grid graph
        nodes1, edges1 = self.create_grid_nodes_edges(self.common_grid_map_1)
        nodes2, edges2 = self.create_grid_nodes_edges(self.common_grid_map_2)
        self.deformed_grid_1_nodes_edges =  nodes1, edges1
        self.deformed_grid_2_nodes_edges = nodes2, edges2
        return nodes1, edges1, nodes2, edges2

    def create_grid_nodes_edges(self, grid):
        flat_grid = grid.reshape(-1, 2)
        rows, cols = self.map_pair.cropped_grid_1.base_grid_x.shape

        # Precompute all node positions
        nodes = {idx: {"pos": point} for idx, point in enumerate(flat_grid)}
        edges = {}

        # Precompute indices
        right_indices = [(idx, idx + 1) for idx in range(flat_grid.shape[0] - 1) if (idx + 1) % cols != 0]
        bottom_indices = [(idx, idx + cols) for idx in range(flat_grid.shape[0] - cols)]
        diagonal_br_indices = [(idx, idx + cols + 1) for idx in range(flat_grid.shape[0] - cols - 1) if (idx + 1) % cols != 0]
        diagonal_bl_indices = [(idx, idx + cols - 1) for idx in range(flat_grid.shape[0] - cols) if idx % cols != 0]

        # Vectorized edge length calculations
        def add_edges(edge_list):
            for (idx1, idx2) in edge_list:
                edge_length = np.linalg.norm(flat_grid[idx1] - flat_grid[idx2])
                edges[(idx1, idx2)] = {'length': edge_length}

        add_edges(right_indices)
        add_edges(bottom_indices)
        add_edges(diagonal_br_indices)
        add_edges(diagonal_bl_indices)

        return nodes, edges

    def create_grid_nodes_edges_old(self, grid):
        flat_grid = grid.reshape(-1, 2)
        rows, cols = self.map_pair.cropped_grid_1.base_grid_x.shape
        nodes = {idx: {"pos": point} for idx, point in enumerate(flat_grid)}
        edges = {}

        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                # Right neighbor
                if col < cols - 1:
                    right_neighbor_idx = idx + 1
                    edges[(idx, right_neighbor_idx)] = {'length': np.linalg.norm(flat_grid[idx] - flat_grid[right_neighbor_idx])}
                # Bottom neighbor
                if row < rows - 1:
                    bottom_neighbor_idx = idx + cols
                    edges[(idx, bottom_neighbor_idx)] = {'length': np.linalg.norm(flat_grid[idx] - flat_grid[bottom_neighbor_idx])}
                # Diagonal neighbors
                if row < rows - 1 and col < cols - 1:
                    diagonal_neighbor_idx_br = idx + cols + 1
                    edges[(idx, diagonal_neighbor_idx_br)] = {'length': np.linalg.norm(flat_grid[idx] - flat_grid[diagonal_neighbor_idx_br])}
                if row < rows - 1 and col > 0:
                    diagonal_neighbor_idx_bl = idx + cols - 1
                    edges[(idx, diagonal_neighbor_idx_bl)] = {'length': np.linalg.norm(flat_grid[idx] - flat_grid[diagonal_neighbor_idx_bl])}

        return nodes, edges

    def get_ratio_graph_from_dicts(self): #used?
        # Unpack edges from the tuples
        _, edges_deformed_1 = self.deformed_grid_1_nodes_edges
        _, edges_deformed_2 = self.deformed_grid_2_nodes_edges

        # Preprocess lengths
        self.preprocess_graph_lengths_from_edges(edges_deformed_1, edges_deformed_2)

        # Create ratio graphs using the preprocessed and scaled edges
        self.ratios_seeds = self.create_ratio_graph_from_dicts(edges_deformed_1, edges_deformed_2, seed_threshold=0.2)

    def convert_dicts_to_graph(self, ratios, seeds): # not used
        G = nx.Graph()
        for edge, ratio in ratios.items():
            G.add_edge(*edge, ratio=ratio)
        for node, seed in seeds.items():
            G.nodes[node]['seed'] = seed
        return G

    def create_ratio_graph_from_dicts(self, deformed_edges_1, deformed_edges_2, seed_threshold=0.1):
        # Ratio calculations
        ratios, seeds = self.calculate_ratios_and_seeds(deformed_edges_1, deformed_edges_2, seed_threshold)
        
        # Potentially return a NetworkX graph if needed for further operations
        return ratios, seeds

    def calculate_ratios_and_seeds(self, deformed_edges_1, deformed_edges_2, seed_threshold):
        ratios = {}
        seeds = {}
        
        # Calculate the ratios between edges in the two deformed graphs
        for edge, attr in deformed_edges_1.items():
            if edge in deformed_edges_2:
                length1 = attr['length']
                length2 = deformed_edges_2[edge]['length']
                ratio = length1 / length2 if length2 != 0 else float('inf')
                ratios[edge] = ratio

        # Expanded neighborhood analysis
        node_connections = {}
        for (node1, node2), ratio in ratios.items():
            if node1 not in node_connections:
                node_connections[node1] = set()
            if node2 not in node_connections:
                node_connections[node2] = set()
            node_connections[node1].add(node2)
            node_connections[node2].add(node1)

        for node, neighbors in node_connections.items():
            all_neighbor_ratios = []
            # Gather ratios for all edges between these neighbors
            for n1 in neighbors:
                for n2 in neighbors:
                    if (n1, n2) in ratios:
                        all_neighbor_ratios.append(ratios[(n1, n2)])
                    elif (n2, n1) in ratios:
                        all_neighbor_ratios.append(ratios[(n2, n1)])

            if all_neighbor_ratios:
                max_ratio = max(all_neighbor_ratios)
                min_ratio = min(all_neighbor_ratios)
                delta = max_ratio - min_ratio
                seed = 1 if delta < seed_threshold else 0
                seeds[node] = seed

        return ratios, seeds

     
    def check_for_seeds(self, G_ratios, seed_threshold=0.1):
        # Ensure all nodes have the correct data structure for processing
        for node in G_ratios.node_indexes():
            if G_ratios[node] is None or not isinstance(G_ratios[node], dict):
                G_ratios[node] = {}  # Initialize node data as an empty dictionary if not set

        for node in G_ratios.node_indexes():
            # Get the neighbors and create a subgraph
            neighbors = list(G_ratios.neighbors(node))
            subgraph_nodes = [node] + neighbors
            subgraph = G_ratios.subgraph(subgraph_nodes)

            # Collect all ratios from the subgraph's edges
            ratios = [subgraph.get_edge_data(u, v)['ratio'] for u, v in subgraph.edge_list() if 'ratio' in subgraph.get_edge_data(u, v)]

            # Calculate the variance of ratios
            if ratios:
                delta = max(ratios) - min(ratios)
                seed = 1 if delta < seed_threshold else 0
            else:
                delta = 0
                seed = 0

            # Update the node data with calculated seed information
            node_data = G_ratios[node]  # Assume it's already a dictionary
            node_data['seed'] = seed
            G_ratios[node] = node_data  # Reassign the updated dictionary back to the node

    def preprocess_graph_lengths_from_edges(self, edges_dict1, edges_dict2):
        median_1 = self.calculate_median_length_from_edges(edges_dict1)
        median_2 = self.calculate_median_length_from_edges(edges_dict2)
        combined_median = np.mean([median_1, median_2])

        scale_factor_1 = combined_median / median_1 if median_1 != 0 else 1
        scale_factor_2 = combined_median / median_2 if median_2 != 0 else 1

        self.scale_edge_lengths_from_edges(edges_dict1, scale_factor_1)
        self.scale_edge_lengths_from_edges(edges_dict2, scale_factor_2)

    def calculate_median_length_from_edges(self, edges_dict):
        lengths = [attr['length'] for attr in edges_dict.values()]
        return np.median(lengths)

    def scale_edge_lengths_from_edges(self, edges_dict, scale_factor):
        for edge, attr in edges_dict.items():
            attr['scaled_length'] = attr['length'] * scale_factor

    def get_combined_mask(self):
        """
        Creates a new mask that combines the alpha shape with seed nodes from two ratio filter results,
        setting mask points to False where nodes are seeds in at least one of the two datasets.

        Parameters:
        - alpha_shape_mask: A 2D numpy array of shape (rows, cols) with boolean values.

        Returns:
        - A new 2D numpy array of shape (rows, cols) as the combined mask.
        """
        _, edges_base = self.base_grid_nodes_edges
        _, edges_deformed_1 = self.deformed_grid_1_nodes_edges
        _, edges_deformed_2 = self.deformed_grid_2_nodes_edges
        _, seeds_1 = self.create_ratio_graph_from_dicts(edges_base, edges_deformed_1, seed_threshold=0.02)
        _, seeds_2 = self.create_ratio_graph_from_dicts(edges_base, edges_deformed_2, seed_threshold=0.02)

        alpha_shape_mask = self.map_pair.common_alpha_shape_mask
        rows, cols = self.map_pair.cropped_grid_1.base_grid_x.shape

        # Initialize the new mask with the existing alpha shape mask values
        new_mask = np.copy(alpha_shape_mask)

        # Convert the flat mask into a 2D indexable array if necessary
        if alpha_shape_mask.ndim == 1:
            new_mask = alpha_shape_mask.reshape((rows, cols))

        # Function to update mask based on seed data
        def update_mask_from_seeds(seeds):
            for node, is_seed in seeds.items():
                if is_seed:  # Check if the node is marked as a seed
                    node_index = int(node)
                    row = node_index // cols
                    col = node_index % cols
                    if 0 <= row < rows and 0 <= col < cols:
                        new_mask[row, col] = False


        # Update the mask based on seeds from both sets of ratios and seeds
        update_mask_from_seeds(seeds_1)
        update_mask_from_seeds(seeds_2)

        self.common_combined_mask = new_mask
        return new_mask

    
    def plot_graph_on_image_mask_filtered(self, map='1', mask_intensity=True, save=True, folder_path='./results/'):
        # Load and prepare the map image
        rows, cols = self.map_pair.cropped_grid_1.base_grid_x.shape
        ratios, seeds = self.ratios_seeds

        if map == '1':
            image_path = self.map_pair.original_map_1.image_path
            flat_grid = self.common_grid_map_1.reshape(-1, 2)
            name = self.map_pair.original_map_1.name +'_'+ self.map_pair.original_map_2.name
        elif map == '2':
            image_path = self.map_pair.original_map_2.image_path
            flat_grid = self.common_grid_map_2.reshape(-1, 2)
            name = self.map_pair.original_map_2.name +'_'+ self.map_pair.original_map_1.name

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        save_name =  name + '_filtered.png'
        # Create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        full_save_path = os.path.join(folder_path, save_name)

        if mask_intensity:
            mask = self.common_combined_mask
        else:
            mask = self.map_pair.common_alpha_shape_mask

        plt.figure(figsize=(30, 30))
        plt.imshow(image_rgb)

        # Plot edges based on the available ratios
        for (u, v), ratio_value in ratios.items():
            if mask[u // cols, u % cols] and mask[v // cols, v % cols]:  # Check if both nodes are within the mask
                x = [flat_grid[u][0], flat_grid[v][0]]
                y = [flat_grid[u][1], flat_grid[v][1]]
                plt.plot(x, y, color='white', lw=1, alpha=0.9)

        # Highlight seed nodes within the mask
    
        seed_nodes = [node for node, is_seed in seeds.items() if is_seed]
        filtered_seed_nodes = [node for node in seed_nodes if mask[node // cols, node % cols]]

        if filtered_seed_nodes:  # Check if there are any seed nodes to plot
            seed_positions = np.array([flat_grid[node] for node in filtered_seed_nodes])
            if seed_positions.size > 0:  # Additional check to handle edge cases
                plt.scatter(seed_positions[:, 0], seed_positions[:, 1], color='red', s=15, label='Seed Nodes', zorder=5, alpha=0.5)

        plt.legend()
        plt.axis('off')
        if save:
            plt.savefig(full_save_path, bbox_inches='tight')
        if not save:
            plt.show()


    def plot_ratios_and_seeds(self):
        """
        Plots the graph with seed nodes in black and edges colored by their ratio.

        Parameters:
        - G_ratios: The graph with edge ratios and seed information.
        - G: The original graph with correct node positions.
        """
        G_ratios = self.ratio_graph
        G = self.base_grid_graph
        # Retrieve positions from the original graph G
        pos = nx.get_node_attributes(G, 'pos')

        # Identify seed nodes
        seed_nodes = [node for node, data in G_ratios.nodes(data=True) if data.get('seed') == 1]

        # Setup for edge coloring based on ratio
        edges, ratios = zip(*nx.get_edge_attributes(G_ratios, 'ratio').items())
        edge_colors = [G_ratios[u][v]['ratio'] for u, v in G_ratios.edges()]

        # Create the plot with explicit figure and axes
        fig, ax = plt.subplots(figsize=(20, 20))
        # Highlight seed nodes in black
        nx.draw_networkx_nodes(G_ratios, pos, nodelist=seed_nodes, node_color='black', alpha=1, ax=ax , node_size=5)

        # Draw edges, colored by their ratio
        nx.draw_networkx_edges(G_ratios, pos, edge_color=edge_colors, edge_cmap=plt.get_cmap('hsv'), width=2, ax=ax)

        # Add colorbar for edge colors
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('hsv'), norm=plt.Normalize(vmin=min(ratios), vmax=max(ratios)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Edge Ratio')

        plt.axis('off')
        plt.title('Graph with Seed Nodes and Colored Edges')
        plt.show()

    def plot_common_grid_over_map(self, map='1', action='plot', figsize=(10, 8), save_path=None):
        # Construct save_path dynamically from map names if not provided
        if save_path is None:
            save_path = f"{self.original_map_1.name}_{self.original_map_2.name}_{map}_grid.png"
        
        # Select the appropriate map and corresponding coordinate grid
        if map == '1':
            image_path = self.original_map_1.image_path
            coordinate_grid = self.common_grid_map1
        elif map == '2':
            image_path = self.original_map_2.image_path
            coordinate_grid = self.common_grid_map2
        
        # Load the image and convert it to RGB
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare the figure and axis with the specified figsize
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image_rgb)
        ax.axis('off')
        
        # Plotting the grid lines
        # Plot horizontal lines
        for y in range(coordinate_grid.shape[0]):
            xs = coordinate_grid[y, :, 0]  # X coordinates for the current row
            ys = coordinate_grid[y, :, 1]  # Y coordinates for the current row
            ax.plot(xs, ys, color='blue', linewidth=1, alpha=0.6)  # Connect all points in the row
        
        # Plot vertical lines
        for x in range(coordinate_grid.shape[1]):
            xs = coordinate_grid[:, x, 0]  # X coordinates for the current column
            ys = coordinate_grid[:, x, 1]  # Y coordinates for the current column
            ax.plot(xs, ys, color='blue', linewidth=1, alpha=0.6)  # Connect all points in the column

        if action == 'save':
            # Save the plot to a file
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)  # Close the plot to free memory
        elif action == 'plot':
            # Display the plot
            plt.show()
            plt.close(fig)  # Close the plot to free memory

    def plot_regions_matrix_over_map(self, map='1', action='plot', figsize=(10, 8), save_path=None):
        # Construct save_path dynamically from map names
        if save_path is None:
            save_path = f"{self.original_map_1.name}_{self.original_map_2.name}_{map}.png"
        
        if map == '1':
            image_path = self.original_map_1.image_path
            coordinate_grid = self.common_grid_map1
        elif map == '2':
            image_path = self.original_map_2.image_path
            coordinate_grid = self.common_grid_map2

        # Load the image and convert it to RGB
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Prepare the figure and axis with the specified figsize
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image_rgb)
        ax.axis('off')

        # The region indices from self.regions_matrix
        region_indices = self.regions_matrix

        # Plotting points based on the common intensity mask and regions matrix
        for y in range(region_indices.shape[0]):
            for x in range(region_indices.shape[1]):
                region_index = region_indices[y, x]
                if self.common_intensity_mask[y, x] != 0:  # Points to be plotted (mask value is not 0)
                    # Fetching the pixel coordinates from the coordinate grid
                    pixel_x, pixel_y = coordinate_grid[y, x]

                    if region_index > 0:  # Point belongs to a region
                        # Overlap with a red dot
                        ax.plot(pixel_x, pixel_y, 'o', color=green, markersize=5, alpha=0.4)
                    else:
                        ax.plot(pixel_x, pixel_y, 'o', color=red, markersize=5, alpha=0.4)

        if action == 'save':
            # Save the plot to a file
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)  # Close the plot to free memory
        elif action == 'plot':
            # Display the plot
            plt.show()
            plt.close(fig)  # Close the plot to free memory

    def plot_folium_map(self):
        m = folium.Map(location=self.map_center_epsg_4326, zoom_start=14)
        
        # Use the region indices from self.regions_matrix
        region_indices = self.regions_matrix
        
        # And use the geographic coordinates from self.common_grid_epsg_4326
        # Assuming self.common_grid_epsg_4326 is structured with each cell containing [projected_x, projected_y]
        geo_coordinates = self.common_grid_epsg_4326

        for y in range(region_indices.shape[0]):
            for x in range(region_indices.shape[1]):
                region_index = region_indices[y, x]
                if region_index > 0:  # If the cell belongs to a region
                    # Fetch the corresponding geographic coordinates
                    geo_x, geo_y = geo_coordinates[y, x]
                    
                    # Plot each point with a small circle marker. Adjust radius and opacity as needed.
                    folium.CircleMarker(
                        location=[geo_y, geo_x],  # Folium expects [lat, lon]
                        radius=1,  # Adjust the size of the dot here
                        color="blue",  # Blue with some transparency, adjust as needed
                        fill=True,
                        fill_color="blue",  # Same color for filling, adjust as needed
                        fill_opacity=0.2  # Adjust the opacity of the dot here
                    ).add_to(m)

        return m

    def project_common_grid(self, src_epsg=28193, dst_epsg=4326): #todo substitute the src_epsg so that it will be determined by the map
        base_grid_x = self.map_pair.cropped_grid_1.base_grid_x
        base_grid_y = self.map_pair.cropped_grid_1.base_grid_y
        transformer = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)
        
        # Assuming base_grid_x and base_grid_y are 2D arrays of the same shape
        projected_x = np.zeros_like(base_grid_x)
        projected_y = np.zeros_like(base_grid_y)
        
        for y in range(base_grid_x.shape[0]):
            for x in range(base_grid_x.shape[1]):
                geo_x = base_grid_x[y, x]
                geo_y = base_grid_y[y, x]
                
                # Project coordinates
                projected_x[y, x], projected_y[y, x] = transformer.transform(geo_x, geo_y)
        
        # Save the local and projected grids in the class
        self.common_grid_local_epsg = np.stack((base_grid_x, base_grid_y), axis=-1) #it's just the base grid so we don't need to call this function
        self.common_grid_epsg_4326 = np.stack((projected_x, projected_y), axis=-1)
        self.map_center_epsg_4326 = np.mean(projected_y), np.mean(projected_x)


@dataclass 
class FieldPair:
    map_pair: MapPair
    cropped_field_1: Field = None
    cropped_field_2: Field = None
    common_intensity_mask : np.array = None
    area_intensity_mask : float = None

    common_grid_local_epsg: np.ndarray = None
    common_grid_epsg_4326: np.ndarray = None
    cosine_similarity_grid: np.ndarray = None
    filtered_cosine_similarity_grid_magnitude : np.ndarray = None
    cosine_similarity_grid_ratio : np.ndarray = None  
    filtered_cosine_similarity_grid_combined : np.ndarray = None
    regions_matrix : np.ndarray = None
    area_regions_matrix : float = None
    regions_matrix_local_epsg : np.ndarray = None
    map_center_epsg_4326 : tuple = None
    regions_matrix_epsg_4326 : np.ndarray = None
    base_grid_graph : nx.Graph = None
    deformed_grid_1_graph : nx.Graph = None
    deformed_grid_2_graph : nx.Graph = None
    ratio_graph : nx.Graph = None   
    filter_1_graph : nx.Graph = None
    filter_2_graph : nx.Graph = None

    def __post_init__(self):
        self.get_cropped_fields()
        self.get_common_intensity_mask()
        self.calculate_cosine_similarity()
        self.calculate_filtered_cosine_similarity_grid_magnitude()
        self.calculate_cosine_similarity_grid_ratio()
        self.calculate_filtered_cosine_similarity_grid_combined()
        
        #self.project_common_grid() #maybe these three could just be called when needed
        #self.project_common_grid_to_map_space()
        self.segment_regions()
        #self.get_base_grid_graph()
        #self.get_deformed_grid_graph()
        #self.get_ratio_graph(get_filters = True) # for the moment we don't calculate the filters because it takes too much time
        #self.get_combined_mask()
        
    def get_cropped_fields(self):
        cropped_grid_1 = self.map_pair.cropped_grid_1
        cropped_grid_2 = self.map_pair.cropped_grid_2
        self.cropped_field_1 = Field(cropped_grid_1)
        self.cropped_field_2 = Field(cropped_grid_2)

    def get_common_intensity_mask(self):
        #get the cropped and filtered with alpha shape displacement magnitudes for the two field
        #magnitudes_1 = np.sqrt(self.cropped_field_1.displacement_x**2 + self.cropped_field_1.displacement_y**2)
        #magnitudes_2 = np.sqrt(self.cropped_field_2.displacement_x**2 + self.cropped_field_2.displacement_y**2)
        #filter the magnitudes using the alpha shape mask
        #magnitudes_1[~self.cropped_field_1.grid.alpha_shape_mask] = np.nan
        #magnitudes_2[~self.cropped_field_2.grid.alpha_shape_mask] = np.nan
        #quartile_1 = np.nanquantile(magnitudes_1, threshold)
        #quartile_2 = np.nanquantile(magnitudes_2, threshold)
        #intensity_mask_1 = magnitudes_1 >= quartile_1
        #self.cropped_field_1.intensity_mask = intensity_mask_1
        #intensity_mask_2 = magnitudes_2 >= quartile_2
        #self.cropped_field_2.intensity_mask = intensity_mask_2
        intensity_mask_1 = self.cropped_field_1.intensity_mask
        intensity_mask_2 = self.cropped_field_2.intensity_mask
        common_intensity_mask = intensity_mask_1 & intensity_mask_2
        self.common_intensity_mask = common_intensity_mask
        #get the area as the number of true values
        self.area_intensity_mask = np.sum(common_intensity_mask)

    def calculate_cosine_similarity(self):
        displacement_x_1, displacement_y_1 = self.cropped_field_1.displacement_x, self.cropped_field_1.displacement_y
        displacement_x_2, displacement_y_2 = self.cropped_field_2.displacement_x, self.cropped_field_2.displacement_y
        
        # Determine the minimum shape to safely operate on both arrays
        min_rows = min(displacement_x_1.shape[0], displacement_x_2.shape[0])
        min_cols = min(displacement_x_1.shape[1], displacement_x_2.shape[1])

        # Restrict the arrays to their common shape
        dx1 = displacement_x_1[:min_rows, :min_cols]
        dy1 = displacement_y_1[:min_rows, :min_cols]
        dx2 = displacement_x_2[:min_rows, :min_cols]
        dy2 = displacement_y_2[:min_rows, :min_cols]

        # Calculate the norms and dot products in a vectorized way
        norms_1 = np.sqrt(dx1**2 + dy1**2)
        norms_2 = np.sqrt(dx2**2 + dy2**2)
        dot_products = dx1 * dx2 + dy1 * dy2

        # Calculate cosine similarity
        # To avoid division by zero, use np.clip to set minimum norm values to a very small positive number
        # is there a faster way with np?
        cosine_similarities = dot_products / (np.clip(norms_1 * norms_2, a_min=1e-10, a_max=None))

        self.cosine_similarity_grid = cosine_similarities

    def calculate_filtered_cosine_similarity_grid_magnitude(self):
        #copy the cosine similarity grid
        cosine_similarity_grid_filtered = self.cosine_similarity_grid.copy()
        #mask it using the intensity mask and putting nan where the mask is false
        cosine_similarity_grid_filtered[~self.common_intensity_mask] = np.nan
        self.filtered_cosine_similarity_grid_magnitude = cosine_similarity_grid_filtered

    def calculate_cosine_similarity_grid_ratio(self, threshold_ratio = 0.5, threshold_cosine = 0.85):
        cosine_similarity_grid_ratio = np.zeros(self.cosine_similarity_grid.shape)
        magnitudes_1 = np.sqrt(self.cropped_field_1.displacement_x**2 + self.cropped_field_1.displacement_y**2)
        magnitudes_2 = np.sqrt(self.cropped_field_2.displacement_x**2 + self.cropped_field_2.displacement_y**2)
        mean_magnitudes_1 = np.nanmean(magnitudes_1)
        mean_magnitudes_2 = np.nanmean(magnitudes_2)
        mean = np.mean([mean_magnitudes_1, mean_magnitudes_2])
        for i in range(self.cosine_similarity_grid.shape[0]):
            for j in range(self.cosine_similarity_grid.shape[1]):
                # Calculate magnitudes of vectors
                magnitude_1 = magnitudes_1[i, j]
                magnitude_2 = magnitudes_2[i, j]
                max_mag = np.max([magnitude_1, magnitude_2])
                min_mag = np.min([magnitude_1, magnitude_2])
                ratio = min_mag / max_mag
                if self.cosine_similarity_grid[i, j] > threshold_cosine and ratio < threshold_ratio and max_mag > mean:
                    cosine_similarity_grid_ratio[i, j] = self.cosine_similarity_grid[i, j] * ratio
                else:
                    cosine_similarity_grid_ratio[i, j] = self.cosine_similarity_grid[i, j]
                #remap the ratio from the range [0,1] to the range [-1,1]
        self.cosine_similarity_grid_ratio = cosine_similarity_grid_ratio
    
    def calculate_filtered_cosine_similarity_grid_combined(self):
        #copy the ratio grid
        combined_filtered_cosine_similarity_grid = self.cosine_similarity_grid_ratio.copy()
        #mask it using the intensity mask and putting nan where the mask is false
        combined_filtered_cosine_similarity_grid[~self.common_intensity_mask] = np.nan
        self.filtered_cosine_similarity_grid_combined = combined_filtered_cosine_similarity_grid
    
    def segment_regions(self,  similarity_matrix= None, threshold=0.8, minimum_area=20): #by default it's done with the combined similarity, in the future we may integrate more of them
        #in the future adjust the area to be expressed in meters so that it's not dependent on the grid size
        if similarity_matrix is None:
            similarity_matrix_filtered = self.filtered_cosine_similarity_grid_combined.copy()
            similarity_matrix_filtered[self.filtered_cosine_similarity_grid_combined < threshold] = np.nan
        else:
            similarity_matrix_filtered = similarity_matrix.copy()
            similarity_matrix_filtered[similarity_matrix < threshold] = np.nan
        regions_matrix = np.zeros_like(similarity_matrix_filtered)

        def is_valid(cell, matrix):
            '''Helper function to check if a cell is valid for inclusion in a region'''
            i, j = cell
            return 0 <= i < matrix.shape[0] and 0 <= j < matrix.shape[1] and not np.isnan(matrix[i, j])

        def grow_region(seed, matrix, region_id, regions_matrix):
            '''Function to grow a region from a seed point'''
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 4-connectivity
            queue = deque([seed])
            area = 0
            
            while queue:
                current = queue.popleft()
                for d in directions:
                    neighbor = (current[0] + d[0], current[1] + d[1])
                    if is_valid(neighbor, matrix) and regions_matrix[neighbor] == 0:
                        regions_matrix[neighbor] = region_id
                        queue.append(neighbor)
                        area += 1
            return area
        total_area = 0
        # Initialize variables
        region_id = 1
        for i in range(similarity_matrix_filtered.shape[0]):
            for j in range(similarity_matrix_filtered.shape[1]):
                if is_valid((i, j), similarity_matrix_filtered) and regions_matrix[i, j] == 0:
                    regions_matrix[i, j] = region_id
                    area = grow_region((i, j), similarity_matrix_filtered, region_id, regions_matrix)
                    if area < minimum_area:
                        # Reset the region to 0 if it doesn't meet the minimum area requirement
                        regions_matrix[regions_matrix == region_id] = 0
                    else:
                        region_id += 1
                        total_area += area
        
        self.regions_matrix = regions_matrix
        self.area_regions_matrix = total_area
        return regions_matrix



    def get_common_alpha_shape_mask(self):
        alpha_shape_mask_1 = self.cropped_field_1.alpha_shape_mask
        alpha_shape_mask_2 = self.cropped_field_2.alpha_shape_mask
        #crop the two masks to the common extent
        min_x, max_x, min_y, max_y = self.common_extent
        # Determine indices for cropping based on common_extent directly related to how grids are generated
        # Generate boolean arrays for rows and columns to keep 
        keep_rows_1 = (self.cropped_field_1.base_grid_y >= min_y) & (self.cropped_field_1.base_grid_y <= max_y)
        keep_columns_1 = (self.cropped_field_1.base_grid_x >= min_x) & (self.cropped_field_1.base_grid_x <= max_x)
        keep_rows_2 = (self.cropped_field_2.base_grid_y >= min_y) & (self.cropped_field_2.base_grid_y <= max_y)
        keep_columns_2 = (self.cropped_field_2.base_grid_x >= min_x) & (self.cropped_field_2.base_grid_x <= max_x)
        # Ensure we're checking along the correct axis for both rows and columns
        keep_rows_1 = np.any(keep_rows_1, axis=1)  # Any row with Y within bounds
        keep_columns_1 = np.any(keep_columns_1, axis=0)  # Any column with X within bounds
        keep_rows_2 = np.any(keep_rows_2, axis=1)  # Any row with Y within bounds
        keep_columns_2 = np.any(keep_columns_2, axis=0)  # Any column with X within bounds
        # Apply row and column masks to crop
        cropped_alpha_shape_mask_1 = alpha_shape_mask_1[keep_rows_1][:, keep_columns_1]
        cropped_alpha_shape_mask_2 = alpha_shape_mask_2[keep_rows_2][:, keep_columns_2]
        # Combine the two masks by using the logical AND operation
        common_alpha_shape_mask = cropped_alpha_shape_mask_1 & cropped_alpha_shape_mask_2
        self.common_alpha_shape_mask = common_alpha_shape_mask
    def crop_gcp_df(self, gcp_df, min_x, max_x, min_y, max_y):
        cropped_gcp_df = gcp_df[
            (gcp_df['mapX'] >= min_x) & (gcp_df['mapX'] <= max_x) &
            (gcp_df['mapY'] >= min_y) & (gcp_df['mapY'] <= max_y)
        ]
        return cropped_gcp_df
    def get_common_gcp_df(self):
        minX, maxX, minY, maxY = self.common_extent
        gcp_df_1 = self.original_field_1.gcp_df
        gcp_df_2 = self.original_field_2.gcp_df
        cropped_gcp_df1 = self.crop_gcp_df(gcp_df_1, minX, maxX, minY, maxY)
        cropped_gcp_df2 = self.crop_gcp_df(gcp_df_2, minX, maxX, minY, maxY)

        self.common_gcp_df_1 = cropped_gcp_df1
        self.common_gcp_df_2 = cropped_gcp_df2


    def get_cropped_array(self, array, keep_x_indices, keep_y_indices): #unused?
        cropped_array = array[keep_y_indices, :][:, keep_x_indices]
        return cropped_array





    def calculate_filtered_cosine_similarity_grid_magnitude(self):
        #copy the cosine similarity grid
        cosine_similarity_grid_filtered = self.cosine_similarity_grid.copy()
        #mask it using the intensity mask and putting nan where the mask is false
        cosine_similarity_grid_filtered[~self.common_intensity_mask] = np.nan
        self.filtered_cosine_similarity_grid_magnitude = cosine_similarity_grid_filtered



    def calculate_cosine_similarity_grid_ratio(self, threshold_ratio = 0.5, threshold_cosine = 0.85):
        cosine_similarity_grid_ratio = np.zeros(self.cosine_similarity_grid.shape)
        magnitudes_1 = np.sqrt(self.cropped_field_1.displacement_x**2 + self.cropped_field_1.displacement_y**2)
        magnitudes_2 = np.sqrt(self.cropped_field_2.displacement_x**2 + self.cropped_field_2.displacement_y**2)
        mean_magnitudes_1 = np.nanmean(magnitudes_1)
        mean_magnitudes_2 = np.nanmean(magnitudes_2)
        mean = np.mean([mean_magnitudes_1, mean_magnitudes_2])
        for i in range(self.cosine_similarity_grid.shape[0]):
            for j in range(self.cosine_similarity_grid.shape[1]):
                # Calculate magnitudes of vectors
                magnitude_1 = magnitudes_1[i, j]
                magnitude_2 = magnitudes_2[i, j]
                max_mag = np.max([magnitude_1, magnitude_2])
                min_mag = np.min([magnitude_1, magnitude_2])
                ratio = min_mag / max_mag
                if self.cosine_similarity_grid[i, j] > threshold_cosine and ratio < threshold_ratio and max_mag > mean:
                    cosine_similarity_grid_ratio[i, j] = self.cosine_similarity_grid[i, j] * ratio
                else:
                    cosine_similarity_grid_ratio[i, j] = self.cosine_similarity_grid[i, j]
                #remap the ratio from the range [0,1] to the range [-1,1]
        self.cosine_similarity_grid_ratio = cosine_similarity_grid_ratio
    
    def calculate_filtered_cosine_similarity_grid_combined(self):
        #copy the ratio grid
        combined_filtered_cosine_similarity_grid = self.cosine_similarity_grid_ratio.copy()
        #mask it using the intensity mask and putting nan where the mask is false
        combined_filtered_cosine_similarity_grid[~self.common_intensity_mask] = np.nan
        self.filtered_cosine_similarity_grid_combined = combined_filtered_cosine_similarity_grid
        

    
    def project_common_grid(self, src_epsg=28193, dst_epsg=4326): #todo substitute the src_epsg so that it will be determined by the map
        base_grid_x = self.cropped_field_1.base_grid_x
        base_grid_y = self.cropped_field_2.base_grid_y
        transformer = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)
        
        # Assuming base_grid_x and base_grid_y are 2D arrays of the same shape
        projected_x = np.zeros_like(base_grid_x)
        projected_y = np.zeros_like(base_grid_y)
        
        for y in range(base_grid_x.shape[0]):
            for x in range(base_grid_x.shape[1]):
                geo_x = base_grid_x[y, x]
                geo_y = base_grid_y[y, x]
                
                # Project coordinates
                projected_x[y, x], projected_y[y, x] = transformer.transform(geo_x, geo_y)
        
        # Save the local and projected grids in the class
        self.common_grid_local_epsg = np.stack((base_grid_x, base_grid_y), axis=-1)
        self.common_grid_epsg_4326 = np.stack((projected_x, projected_y), axis=-1)
        self.map_center_epsg_4326 = np.mean(projected_y), np.mean(projected_x)


    def apply_rbf_to_grid(self, gcp_df, grid):
        # Extract real-world and pixel coordinates from GCPs
        real_world_x = gcp_df['mapX'].values
        real_world_y = gcp_df['mapY'].values
        pixel_x = gcp_df['sourceX'].values
        pixel_y = -gcp_df['sourceY'].values  # invert the y axis for image coordinates maybe not!

        # Initialize RBF interpolator
        rbf = RBFInterpolator(np.column_stack((real_world_x, real_world_y)),
                              np.column_stack((pixel_x, pixel_y)),
                              kernel='thin_plate_spline')

        # Initialize a new grid for transformed coordinates
        transformed_grid = np.zeros_like(grid)
        # Apply RBF to each point in the grid
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                geo_x, geo_y = grid[y, x]
                # Apply RBF interpolation to transform to pixel space
                pixel_location = rbf(np.array([[geo_x, geo_y]]))
                # Store the transformed location
                transformed_grid[y, x] = pixel_location[0]
        
        return transformed_grid

    def project_common_grid_to_map_space(self):
        # Apply RBF to the local EPSG grid for both maps
        self.common_grid_map1 = self.apply_rbf_to_grid(self.cropped_field_1.gcp_df, self.common_grid_local_epsg)
        self.common_grid_map2 = self.apply_rbf_to_grid(self.cropped_field_2.gcp_df, self.common_grid_local_epsg)
    
    def apply_rbf_to_matrix(self, gcp_df, matrix): #maybe can be deleted todo #seems not used anywhere check and delete
        real_world_x = gcp_df['mapX'].values
        real_world_y = gcp_df['mapY'].values
        pixel_x = gcp_df['sourceX'].values
        pixel_y = -gcp_df['sourceY'].values  # invert the y axis

        rbf = RBFInterpolator(np.column_stack((real_world_x, real_world_y)), np.column_stack((pixel_x, pixel_y)), kernel='thin_plate_spline')

        transformed_matrix = np.zeros_like(matrix)

        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                cell = matrix[y, x]
                region_index = cell[0]
                if region_index > 0:  # Point belongs to a region
                    projected_x, projected_y = cell[1], cell[2]

                    # Apply RBF interpolation to project to pixel space
                    pixel_location = rbf(np.array([[projected_x, projected_y]]))

                    # Store the transformed location with the region index
                    transformed_matrix[y, x] = [region_index, pixel_location[0, 1], pixel_location[0, 0]]  # Note the order might need adjustment based on your coordinate system

        return transformed_matrix
    
    def calculate_median_length(self, G):
        lengths = [data['length'] for _, _, data in G.edges(data=True)]
        return np.median(lengths)

    def calculate_combined_median_length(self, G_deformed_1, G_deformed_2):
        median_1 = self.calculate_median_length(G_deformed_1)
        median_2 = self.calculate_median_length(G_deformed_2)
        return np.mean([median_1, median_2])

    def scale_edge_lengths(self, G, scale_factor):
        for u, v, data in G.edges(data=True):
            data['scaled_length'] = data['length'] * scale_factor

    def preprocess_graph_lengths(self, G_deformed_1, G_deformed_2):
        combined_median = self.calculate_combined_median_length(G_deformed_1, G_deformed_2)
        median_1 = self.calculate_median_length(G_deformed_1)
        median_2 = self.calculate_median_length(G_deformed_2)
        
        scale_factor_1 = combined_median / median_1 if median_1 != 0 else 1
        scale_factor_2 = combined_median / median_2 if median_2 != 0 else 1
        
        self.scale_edge_lengths(G_deformed_1, scale_factor_1)
        self.scale_edge_lengths(G_deformed_2, scale_factor_2)
    
    def get_base_grid_graph(self):
        #base grid graph
        grid = np.stack((self.cropped_field_1.base_grid_x, self.cropped_field_1.base_grid_y), axis=-1)
        G = self.create_grid_graph(grid)
        self.base_grid_graph = G
        return G
    def get_deformed_grid_graph(self):
        #deformed grid graph
        grid_1 = self.common_grid_map1
        grid_2 = self.common_grid_map2
        grid_1_graph = self.create_grid_graph(grid_1)
        grid_2_graph = self.create_grid_graph(grid_2)
        self.deformed_grid_1_graph = grid_1_graph
        self.deformed_grid_2_graph = grid_2_graph
        return grid_1_graph, grid_2_graph

    def create_grid_graph(self, grid):
        """
        Creates a graph from a flattened grid, with an option to invert the y-axis, and includes edge lengths.
        Now also includes diagonal connections in each square of the grid.
        
        Parameters:
        - flat_grid: A flattened 2D array where each entry is a point (x, y).
        - rows: The number of rows in the original grid.
        - cols: The number of columns in the original grid.
        - invert_y: Boolean indicating whether to invert the y-axis.
        
        Returns:
        - A NetworkX Graph object with edge lengths as edge attributes.
        """
        G = nx.Graph()
        flat_grid = grid.reshape(-1, 2)
        rows, cols = self.cropped_field_1.base_grid_x.shape 
        # Add nodes with positions using bulk operation
        nodes_with_positions = [(idx, {"pos": point}) for idx, point in enumerate(flat_grid)]
        G.add_nodes_from(nodes_with_positions)
        
        # Prepare edge lists, calculate distances, and add using bulk operation
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                if col < cols - 1:  # Right neighbor
                    right_neighbor_idx = idx + 1
                    edge_length = np.linalg.norm(flat_grid[idx] - flat_grid[right_neighbor_idx])
                    G.add_edge(idx, right_neighbor_idx, length=edge_length)
                if row < rows - 1:  # Bottom neighbor
                    bottom_neighbor_idx = idx + cols
                    edge_length = np.linalg.norm(flat_grid[idx] - flat_grid[bottom_neighbor_idx])
                    G.add_edge(idx, bottom_neighbor_idx, length=edge_length)
                # Diagonal neighbors
                if row < rows - 1 and col < cols - 1:  # Bottom-right neighbor
                    diagonal_neighbor_idx_br = idx + cols + 1
                    edge_length_br = np.linalg.norm(flat_grid[idx] - flat_grid[diagonal_neighbor_idx_br])
                    G.add_edge(idx, diagonal_neighbor_idx_br, length=edge_length_br)
                if row < rows - 1 and col > 0:  # Bottom-left neighbor #do we need this?
                    diagonal_neighbor_idx_bl = idx + cols - 1
                    edge_length_bl = np.linalg.norm(flat_grid[idx] - flat_grid[diagonal_neighbor_idx_bl])
                    G.add_edge(idx, diagonal_neighbor_idx_bl, length=edge_length_bl)

        return G

    def create_ratio_graph(self, G_deformed_1, G_deformed_2, seed_threshold=0.1):
        """
        Creates a graph where the edges represent the ratio of edge lengths
        between two given deformed graphs, and each node has attributes for the
        mean and variance of 'ratio' values of its connected edges.

        Parameters:
        - G_deformed_1: A NetworkX Graph object from the first deformed grid.
        - G_deformed_2: A NetworkX Graph object from the second deformed grid.

        Returns:
        - A NetworkX Graph object with edge length ratios as edge attributes and
        mean and variance of these ratios as node attributes.
        """
        G_ratios = nx.Graph()
        self.preprocess_graph_lengths(G_deformed_1, G_deformed_2)
        for (u, v, data) in G_deformed_1.edges(data=True):
            if G_deformed_2.has_edge(u, v):
                length_1 = data['scaled_length']
                length_2 = G_deformed_2[u][v]['scaled_length']
                ratio = length_1 / length_2 if length_2 != 0 else length_1/1e-6
                if ratio is not None:
                    G_ratios.add_edge(u, v, ratio=ratio)

        all_ratios = [data['ratio'] for _, _, data in G_ratios.edges(data=True)]
        min_ratio = min(all_ratios)
        significant_diff = min_ratio + seed_threshold

        for node in G_ratios.nodes():
            initial_ratios = [G_ratios.edges[u, v]['ratio'] for u, v in G_ratios.edges(node)]
            initial_delta = max(initial_ratios) - min(initial_ratios)
            if initial_delta <= significant_diff: #we prefilter the seeds to avoid useless computation
                # Create the subregion including the node and its immediate neighbors
                subregion_nodes = {node} | set(G_ratios.neighbors(node))
                # Create a subgraph for this subregion
                subgraph = G_ratios.subgraph(subregion_nodes)
                ratios = [subgraph.edges[u, v]['ratio'] for u, v in subgraph.edges()]

                if ratios:
                    mean_ratio = np.mean(ratios)
                    delta = max(ratios) - min(ratios)
                    G_ratios.nodes[node]['mean_ratio'] = mean_ratio
                    G_ratios.nodes[node]['delta'] = delta
                    # Mark as seed based on the delta condition
                    G_ratios.nodes[node]['seed'] = 1 if delta < seed_threshold else 0
                    
            else:
                G_ratios.nodes[node]['mean_ratio'] = np.mean(initial_ratios)
                G_ratios.nodes[node]['delta'] = initial_delta
                G_ratios.nodes[node]['seed'] = 0
        return G_ratios
    def get_ratio_graph(self, get_filters=False):
        base_graph_1 = self.base_grid_graph
        G_deformed_1 = self.deformed_grid_1_graph
        G_deformed_2 = self.deformed_grid_2_graph
        #save the graphs
        self.ratio_graph = self.create_ratio_graph(G_deformed_1, G_deformed_2, seed_threshold=0.2)
        if get_filters:
            self.filter_1_graph = self.create_ratio_graph(base_graph_1, G_deformed_1, seed_threshold=0.02)
            self.filter_2_graph = self.create_ratio_graph(base_graph_1, G_deformed_2, seed_threshold=0.02)

    def get_combined_mask(self):
        """
        Creates a new mask that combines the alpha shape with seed nodes from two ratio filter graphs,
        setting mask points to False where nodes are seeds in at least one of the two graphs.

        Parameters:
        - alpha_shape_mask: A 2D numpy array of shape (rows, cols) with boolean values.
        - G_ratios_filter_1: A NetworkX graph where seed nodes indicate exclusion.
        - G_ratios_filter_2: A NetworkX graph where seed nodes indicate exclusion.
        - rows, cols: Dimensions of the grid corresponding to the mask.

        Returns:
        - A new 2D numpy array of shape (rows, cols) as the combined mask.
        """
        alpha_shape_mask = self.common_alpha_shape_mask
        G_ratios_filter_1 = self.filter_1_graph
        G_ratios_filter_2 = self.filter_2_graph
        rows, cols = self.cropped_field_1.base_grid_x.shape
        # Initialize the new mask with the existing alpha shape mask values
        new_mask = np.copy(alpha_shape_mask)

        # Convert the flat mask into a 2D indexable array if necessary
        if alpha_shape_mask.ndim == 1:
            new_mask = alpha_shape_mask.reshape((rows, cols))

        # Determine seed nodes from both graphs and set their corresponding mask values to False
        def update_mask_from_seeds(G_ratios_filter):
            for node, data in G_ratios_filter.nodes(data=True):
                if data.get('seed', 0) == 1:  # Check if the node is marked as a seed
                    row = node // cols
                    col = node % cols
                    if 0 <= row < rows and 0 <= col < cols:
                        new_mask[row, col] = False

        # Update the mask based on seeds from both graphs
        update_mask_from_seeds(G_ratios_filter_1)
        update_mask_from_seeds(G_ratios_filter_2)
        self.common_combined_mask = new_mask
        return new_mask
    def plot_graph_on_image_mask_filtered(self, map = '1', mask_intensity = True, save = True, folder_path = './results' ):
        # Load and prepare the map image
        G_ratios = self.ratio_graph
        rows, cols = self.cropped_field_1.base_grid_x.shape
        if map == '1':
            image_path = self.original_map_1.image_path
            G_deformed = self.deformed_grid_1_graph
            flat_grid = self.common_grid_map1.reshape(-1, 2)
            name = self.original_map_1.name

        elif map == '2':
            image_path = self.original_map_2.image_path
            G_deformed = self.deformed_grid_2_graph
            flat_grid = self.common_grid_map2.reshape(-1, 2)
            name = self.original_map_2.name
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        save_name =   '/'+ name +'_filtered' '.png'
        # Create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        folder_path += save_name

        if mask_intensity:
            mask = self.common_combined_mask
        else:
            mask = self.common_alpha_shape_mask


        plt.figure(figsize=(30, 30))
        plt.imshow(image_rgb)

        # Convert node indices to grid position and check against the alpha_shape_mask
        filtered_nodes = {idx for idx in range(rows * cols) if mask[idx // cols, idx % cols]}
        filtered_edges = [(u, v) for u, v in G_deformed.edges() if u in filtered_nodes and v in filtered_nodes]

        for u, v in filtered_edges:
            x = [flat_grid[u][0], flat_grid[v][0]]
            y = [flat_grid[u][1], flat_grid[v][1]]
            plt.plot(x, y, color='white', lw=1, alpha=0.9)

        # Highlight seed nodes within the mask
        seed_nodes = [node for node in filtered_nodes if G_ratios.nodes[node].get('seed') == 1]
        seed_positions = np.array([flat_grid[node] for node in seed_nodes])
        plt.scatter(seed_positions[:, 0], seed_positions[:, 1], color='red', s=15, label='Seed Nodes', zorder=5, alpha=0.5)

        plt.legend()
        plt.axis('off')
        plt.savefig(folder_path, bbox_inches='tight')
        plt.show()

    def plot_ratios_and_seeds(self):
        """
        Plots the graph with seed nodes in black and edges colored by their ratio.

        Parameters:
        - G_ratios: The graph with edge ratios and seed information.
        - G: The original graph with correct node positions.
        """
        G_ratios = self.ratio_graph
        G = self.base_grid_graph
        # Retrieve positions from the original graph G
        pos = nx.get_node_attributes(G, 'pos')

        # Identify seed nodes
        seed_nodes = [node for node, data in G_ratios.nodes(data=True) if data.get('seed') == 1]

        # Setup for edge coloring based on ratio
        edges, ratios = zip(*nx.get_edge_attributes(G_ratios, 'ratio').items())
        edge_colors = [G_ratios[u][v]['ratio'] for u, v in G_ratios.edges()]

        # Create the plot with explicit figure and axes
        fig, ax = plt.subplots(figsize=(20, 20))
        # Highlight seed nodes in black
        nx.draw_networkx_nodes(G_ratios, pos, nodelist=seed_nodes, node_color='black', alpha=1, ax=ax , node_size=5)

        # Draw edges, colored by their ratio
        nx.draw_networkx_edges(G_ratios, pos, edge_color=edge_colors, edge_cmap=plt.get_cmap('hsv'), width=2, ax=ax)

        # Add colorbar for edge colors
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('hsv'), norm=plt.Normalize(vmin=min(ratios), vmax=max(ratios)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Edge Ratio')

        plt.axis('off')
        plt.title('Graph with Seed Nodes and Colored Edges')
        plt.show()

    def plot_common_grid_over_map(self, map='1', action='plot', figsize=(10, 8), save_path=None):
        # Construct save_path dynamically from map names if not provided
        if save_path is None:
            save_path = f"{self.original_map_1.name}_{self.original_map_2.name}_{map}_grid.png"
        
        # Select the appropriate map and corresponding coordinate grid
        if map == '1':
            image_path = self.original_map_1.image_path
            coordinate_grid = self.common_grid_map1
        elif map == '2':
            image_path = self.original_map_2.image_path
            coordinate_grid = self.common_grid_map2
        
        # Load the image and convert it to RGB
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare the figure and axis with the specified figsize
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image_rgb)
        ax.axis('off')
        
        # Plotting the grid lines
        # Plot horizontal lines
        for y in range(coordinate_grid.shape[0]):
            xs = coordinate_grid[y, :, 0]  # X coordinates for the current row
            ys = coordinate_grid[y, :, 1]  # Y coordinates for the current row
            ax.plot(xs, ys, color='blue', linewidth=1, alpha=0.6)  # Connect all points in the row
        
        # Plot vertical lines
        for x in range(coordinate_grid.shape[1]):
            xs = coordinate_grid[:, x, 0]  # X coordinates for the current column
            ys = coordinate_grid[:, x, 1]  # Y coordinates for the current column
            ax.plot(xs, ys, color='blue', linewidth=1, alpha=0.6)  # Connect all points in the column

        if action == 'save':
            # Save the plot to a file
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)  # Close the plot to free memory
        elif action == 'plot':
            # Display the plot
            plt.show()
            plt.close(fig)  # Close the plot to free memory

    def plot_regions_matrix_over_map(self, map='1', action='plot', figsize=(10, 8), save_path=None):
        # Construct save_path dynamically from map names
        if save_path is None:
            save_path = f"{self.original_map_1.name}_{self.original_map_2.name}_{map}.png"
        
        if map == '1':
            image_path = self.original_map_1.image_path
            coordinate_grid = self.common_grid_map1
        elif map == '2':
            image_path = self.original_map_2.image_path
            coordinate_grid = self.common_grid_map2

        # Load the image and convert it to RGB
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Prepare the figure and axis with the specified figsize
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image_rgb)
        ax.axis('off')

        # The region indices from self.regions_matrix
        region_indices = self.regions_matrix

        # Plotting points based on the common intensity mask and regions matrix
        for y in range(region_indices.shape[0]):
            for x in range(region_indices.shape[1]):
                region_index = region_indices[y, x]
                if self.common_intensity_mask[y, x] != 0:  # Points to be plotted (mask value is not 0)
                    # Fetching the pixel coordinates from the coordinate grid
                    pixel_x, pixel_y = coordinate_grid[y, x]

                    if region_index > 0:  # Point belongs to a region
                        # Overlap with a red dot
                        ax.plot(pixel_x, pixel_y, 'o', color=green, markersize=5, alpha=0.4)
                    else:
                        ax.plot(pixel_x, pixel_y, 'o', color=red, markersize=5, alpha=0.4)

        if action == 'save':
            # Save the plot to a file
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)  # Close the plot to free memory
        elif action == 'plot':
            # Display the plot
            plt.show()
            plt.close(fig)  # Close the plot to free memory


    def plot_folium_map(self):
        m = folium.Map(location=self.map_center_epsg_4326, zoom_start=14)
        
        # Use the region indices from self.regions_matrix
        region_indices = self.regions_matrix
        
        # And use the geographic coordinates from self.common_grid_epsg_4326
        # Assuming self.common_grid_epsg_4326 is structured with each cell containing [projected_x, projected_y]
        geo_coordinates = self.common_grid_epsg_4326

        for y in range(region_indices.shape[0]):
            for x in range(region_indices.shape[1]):
                region_index = region_indices[y, x]
                if region_index > 0:  # If the cell belongs to a region
                    # Fetch the corresponding geographic coordinates
                    geo_x, geo_y = geo_coordinates[y, x]
                    
                    # Plot each point with a small circle marker. Adjust radius and opacity as needed.
                    folium.CircleMarker(
                        location=[geo_y, geo_x],  # Folium expects [lat, lon]
                        radius=1,  # Adjust the size of the dot here
                        color="blue",  # Blue with some transparency, adjust as needed
                        fill=True,
                        fill_color="blue",  # Same color for filling, adjust as needed
                        fill_opacity=0.2  # Adjust the opacity of the dot here
                    ).add_to(m)

        return m


    def plot_similarity_grid(self, similarity_matrix):
        plt.imshow(similarity_matrix, interpolation='none',aspect='equal', cmap=cmap1)
        plt.clim(-1, 1)
       
        plt.gca().invert_yaxis()
        #remove x and y ticks
        plt.xticks([])
        plt.yticks([])
        #remove the colorbar

        plt.show()
        plt.close()


    def plot_regions_matrix(self):
        plt.imshow(self.regions_matrix, interpolation='nearest', aspect='equal', cmap=cmap1)
        plt.gca().invert_yaxis()
        #remove x and y ticks
        plt.xticks([])
        plt.yticks([])
        #remove the colorbar
        plt.show()
        plt.close()

        
