import matplotlib.pyplot as plt
from math import ceil, floor
import pickle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

class Tile:
    def __init__(self, x, y, z):
        '''Create a new tile at the given position. The coordiante
        represents the smallest x, y, z coordinate covered by the tile.'''
        self.x = x
        self.y = y
        self.z = z
        self.dependencies = []  # Tiles this tile depends on

    def add_dependency(self, tile):
        self.dependencies.append(tile)

    def is_ready(self, existing_tiles):
        """Check if all dependencies are already generated."""
        return all(dep in existing_tiles for dep in self.dependencies)

    def __repr__(self):
        return f"Tile({self.x}, {self.y}, {self.z})"

def gen_part1(existing_tiles, shape):
    tiles = [existing_tiles]
    for i in range(1, shape[0]):
        tiles.append(Tile(i*1.1, i*1.1))
        tiles[-1].add_dependency(tiles[-2])
    return tiles

def plot_cubes(batches, window_size, lims):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def draw_cube(position, ax, color):
        # These arrays define the vertices and faces of a unit cube (each vertex at distance 1 from the origin).
        vertices = np.array([[0, 0, 0],
                             [1, 0, 0],
                             [1, 1, 0],
                             [0, 1, 0],
                             [0, 0, 1],
                             [1, 0, 1],
                             [1, 1, 1],
                             [0, 1, 1]])

        # Scale the cube to the desired size
        vertices *= window_size
        # Translate to the desired position
        vertices += position.astype(int)

        # Define the faces by the vertices they connect
        faces = [[vertices[j] for j in [0, 1, 2, 3]],
                 [vertices[j] for j in [4, 5, 6, 7]], 
                 [vertices[j] for j in [0, 3, 7, 4]], 
                 [vertices[j] for j in [1, 2, 6, 5]], 
                 [vertices[j] for j in [0, 1, 5, 4]],
                 [vertices[j] for j in [2, 3, 7, 6]]]

        # Create a 3D Polygon object
        face_collection = Poly3DCollection(faces, facecolors=color, edgecolors='r', linewidths=1.0, linestyle='-', alpha=0.4)
        ax.add_collection3d(face_collection)

    # Color based on the batch, and draw cubes
    for i, batch in enumerate(batches):
        color = plt.cm.jet(i/len(batches))
        for tile in batch:
            position = np.array([tile.x, tile.y, tile.z]) * window_size / 2
            draw_cube(position, ax, color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Adjust the scale of the plot to fit the data

    ax.auto_scale_xyz([0,lims[0]], 
                      [0,lims[1]], 
                      [0,lims[2]])

    plt.tight_layout()
    plt.show()
    
    # plt.savefig('step6.svg', format='svg', bbox_inches='tight')

def get_next_tiles(tile_list, existing_tiles, max_batch):
    batch = []
    for tile in tile_list:
        if tile.is_ready(existing_tiles) and len(batch) < max_batch and tile not in existing_tiles:
            batch.append(tile)
    existing_tiles.extend(batch)
    return batch, existing_tiles

def checker1(shape):
    tiles = []
    for i in range(0, shape[0], 2):
        for j in range(0, shape[1], 2):
            for k in range(0, shape[2], 2):
                tiles.append(Tile(i*1.5, j*1.5, k*1.5))
    return tiles

def checker1_1(tile_list, shape):
    old_tiles = tile_list
    for i in range(0, shape[0], 2):
        for j in range(0, shape[1], 2):
            for k in range(0, shape[2]-2, 2):
                tile_list.append(Tile(i*1.5, j*1.5, k*1.5+1.5))
                for tile in old_tiles:
                    if tile.x == i*1.5 and tile.y == j*1.5 and (tile.z == k*1.5 or tile.z == (k+2)*1.5):
                        tile_list[-1].add_dependency(tile)
    return tile_list

def checker2(tile_list, shape):
    old_tiles = tile_list
    pt2 = []
    for k in range(0, shape[2]-2, 2):
        for j in range(0, shape[1]-2, 2):
            for i in range(0, shape[0]-2, 2):
                tile_list.append(Tile(i*1.5+1.5, j*1.5+1.5, k*1.5+1.5))
                pt2.append(tile_list[-1])
                for tile in old_tiles:
                    if ((tile.x == i*1.5 and tile.y == j*1.5 and tile.z == k*1.5+1.5) 
                    or (tile.x == (i+2)*1.5 and tile.y == j*1.5 and tile.z == k*1.5+1.5)
                    or (tile.x == (i+2)*1.5 and tile.y == (j+2)*1.5 and tile.z == k*1.5+1.5)
                    or (tile.x == i*1.5 and tile.y == (j+2)*1.5 and tile.z == k*1.5+1.5)):
                        tile_list[-1].add_dependency(tile)
    return tile_list, pt2

def checker2_1(tile_list, pt2, shape):
    old_tiles = tile_list
    for k in range(0, shape[2], 2):
        for j in range(0, shape[1]-2, 2):
            for i in range(0, shape[0]-2, 2):
                tile_list.append(Tile(i*1.5+1.5, j*1.5+1.5, k*1.5))
                pt2.append(tile_list[-1])
                for tile in old_tiles:
                    if tile.x == i*1.5+1.5 and tile.y == j*1.5+1.5 and (tile.z == k*1.5-1.5 or tile.z == k*1.5+1.5):
                        tile_list[-1].add_dependency(tile)
    return tile_list, pt2

def checker3(tile_list, pt2, shape):
    for k in range(0, shape[2]-2, 2):
        for j in range(0, shape[1], 2):
            for i in range(0, shape[0]-2, 2):
                tile_list.append(Tile(i*1.5+1.5, j*1.5, k*1.5+1.5))
                for tile in pt2:
                    tile_list[-1].add_dependency(tile)
    for k in range(0, shape[2]-2, 2):
        for j in range(0, shape[1]-2, 2):
            for i in range(0, shape[0], 2):
                tile_list.append(Tile(i*1.5, j*1.5+1.5 ,k*1.5+1.5))
                for tile in pt2:
                    tile_list[-1].add_dependency(tile)
    return tile_list

def checker3_1(tile_list, pt2, shape):
    for k in range(0, shape[2], 2):
        for j in range(0, shape[1], 2):
            for i in range(0, shape[0]-2, 2):
                tile_list.append(Tile(i*1.5+1.5, j*1.5, k*1.5))
                for tile in pt2:
                    tile_list[-1].add_dependency(tile)
                for tile in tile_list:
                    if tile.x == i*1.5+1.5 and tile.y == j*1.5 and (tile.z == k*1.5-1.5 or tile.z == k*1.5+1.5):
                        tile_list[-1].add_dependency(tile)
    for k in range(0, shape[2], 2):
        for j in range(0, shape[1]-2, 2):
            for i in range(0, shape[0], 2):
                tile_list.append(Tile(i*1.5, j*1.5+1.5 ,k*1.5))
                for tile in pt2:
                    tile_list[-1].add_dependency(tile)
                for tile in tile_list:
                    if tile.x == i*1.5 and tile.y == j*1.5+1.5 and (tile.z == k*1.5-1.5 or tile.z == k*1.5+1.5):
                        tile_list[-1].add_dependency(tile)
    return tile_list

def place_center(shape):
    return [Tile(floor(shape[0]*.5)+1, floor(shape[1]*.5)+1, floor(shape[2]*.5)+1)]

def gen_corners(tile_list, shape):
    center = tile_list[0]
    prev_batch = tile_list.copy()
    tile_list.append(Tile(center.x+1.5, center.y+1.5, center.z+1.5))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x-1.5, center.y+1.5, center.z+1.5))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x+1.5, center.y-1.5, center.z+1.5))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x-1.5, center.y-1.5, center.z+1.5))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x+1.5, center.y+1.5, center.z-1.5))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x-1.5, center.y+1.5, center.z-1.5))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x+1.5, center.y-1.5, center.z-1.5))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x-1.5, center.y-1.5, center.z-1.5))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    return tile_list

def gen_cross(tile_list, shape):
    center = tile_list[0]
    prev_batch = tile_list[1:]
    tile_list.append(Tile(center.x+1, center.y, center.z))
    tile_list[-1].add_dependency(tile_list[0])
    tile_list.append(Tile(center.x-1, center.y, center.z))
    tile_list[-1].add_dependency(tile_list[0])
    tile_list.append(Tile(center.x, center.y+1, center.z))
    tile_list[-1].add_dependency(tile_list[0])
    tile_list.append(Tile(center.x, center.y-1, center.z))
    tile_list[-1].add_dependency(tile_list[0])
    tile_list.append(Tile(center.x, center.y, center.z+1))
    tile_list[-1].add_dependency(tile_list[0])
    tile_list.append(Tile(center.x, center.y, center.z-1))
    tile_list[-1].add_dependency(tile_list[0])
    return tile_list

def gen_cross2(tile_list, shape, offset):
    center = tile_list[0]
    prev_batch = tile_list.copy()
    tile_list.append(Tile(center.x+offset, center.y, center.z))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x-offset, center.y, center.z))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x, center.y+offset, center.z))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x, center.y-offset, center.z))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x, center.y, center.z+offset))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x, center.y, center.z-offset))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    return tile_list

def gen_diagsx(tile_list, shape, offset):
    center = tile_list[0]
    prev_batch = tile_list
    tile_list.append(Tile(center.x+offset, center.y, center.z+1))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x-offset, center.y, center.z+1))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x+offset, center.y, center.z-1))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x-offset, center.y, center.z-1))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    return tile_list

def gen_diags(tile_list, shape, offset):
    center = tile_list[0]
    prev_batch = tile_list.copy()
    tile_list.append(Tile(center.x+offset, center.y+offset, center.z))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x-offset, center.y+offset, center.z))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x+offset, center.y-offset, center.z))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x-offset, center.y-offset, center.z))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x+offset, center.y, center.z+offset))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x-offset, center.y, center.z+offset))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x+offset, center.y, center.z-offset))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x-offset, center.y, center.z-offset))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x, center.y+offset, center.z+offset))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x, center.y-offset, center.z+offset))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x, center.y+offset, center.z-offset))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    tile_list.append(Tile(center.x, center.y-offset, center.z-offset))
    for tile in prev_batch:
        tile_list[-1].add_dependency(tile)
    return tile_list

def gen_diag_infill(tile_list, shape, offset):
    center = tile_list[0]
    prev_batch = tile_list
    combos = [(1,1,1),
              (1,-1,1),
              (-1,1,1),
              (-1,-1,1),
              (1,1,-1),
              (1,-1,-1),
              (-1,1,-1),
              (-1,-1,-1)]
    for combo in combos:
        tile_list.append(Tile(center.x+combo[0]*offset, center.y+combo[1]*offset, center.z+combo[2]*offset))
        for tile in prev_batch:
            tile_list[-1].add_dependency(tile)
    return tile_list

def gen_diag_up(tile_list, shape, offset):
    center = tile_list[0]
    prev_batch = tile_list
    combos = [(1,0,1),
            (0,1,1),
            (0,-1,1),
            (-1,0,1),
            (1,0,-1),
            (0,1,-1),
            (0,-1,-1),
            (-1,0,-1)]
    for combo in combos:
        tile_list.append(Tile(center.x+combo[0]*offset, center.y+combo[1]*offset, center.z+combo[2]*offset))
        for tile in prev_batch:
            tile_list[-1].add_dependency(tile)
        
    return tile_list

def gen_center(shape_px, window_size, max_batch):
    window_size = window_size
    max_batch = max_batch

    x_len_pixels, y_len_pixels, z_len_pixels = shape_px
    multiple = window_size/2
    x_len_round = multiple *ceil(x_len_pixels / multiple)
    y_len_round = multiple *ceil(y_len_pixels / multiple)
    z_len_round = multiple *ceil(z_len_pixels / multiple)
    shape = (round((x_len_round-16)/16), round((y_len_round-16)/16), round((z_len_round-16)/16))
    # tile_list = checker1(shape)
    tile_list = place_center(shape)
    
    tile_list = gen_cross(tile_list, shape)
    tile_list = gen_cross2(tile_list, shape, 2)
   


    tile_list = gen_diags(tile_list, shape, 1.5)
    tile_list = gen_corners(tile_list, shape)

    # tile_list = gen_diag_infill(tile_list, shape, 1.5)
    # tile_list = gen_diag_up(tile_list, shape, 3)
    # tile_list = gen_diagsx(tile_list, shape, 2)

    
    

    existing_tiles = []
    batches = []
    # batches.append([initial_tile])
    while True:
        batch, existing_tiles = get_next_tiles(tile_list, existing_tiles, max_batch)
        if not batch:
            break
        batches.append(batch)
    return batches

def gen_grid(shape_px, window_size, max_batch):
    window_size = window_size
    max_batch = max_batch

    x_len_pixels, y_len_pixels, z_len_pixels = shape_px
    multiple = window_size/2
    x_len_round = multiple *ceil(x_len_pixels / multiple)
    y_len_round = multiple *ceil(y_len_pixels / multiple)
    z_len_round = multiple *ceil(z_len_pixels / multiple)
    shape = (round((x_len_round-16)/16), round((y_len_round-16)/16), round((z_len_round-16)/16))
    tile_list = checker1(shape)
    tile_list = checker1_1(tile_list, shape)
    tile_list, pt2 = checker2(tile_list, shape)
    tile_list, pt2 = checker2_1(tile_list, pt2, shape)
    tile_list = checker3(tile_list, pt2, shape)
    tile_list = checker3_1(tile_list, pt2, shape)

    # tile_list = gen_top_row(initial_tile, shape)
    # tile_list, columns = gen_column(tile_list, shape)
    # tile_list = gen_infill_column(tile_list, shape, columns)
    existing_tiles = []
    batches = []
    # batches.append([initial_tile])
    while True:
        batch, existing_tiles = get_next_tiles(tile_list, existing_tiles, max_batch)
        if not batch:
            break
        batches.append(batch)
    return batches

def batch_to_coords(batch, window_size):
    coords = []
    for tile in batch:
        coords.append((int(tile.x*window_size/2), int(tile.y*window_size/2), int(tile.z*window_size/2)))
    return coords

# window_size = 16
# initial_tile = Tile(0, 0)
# max_batch = 16
# x_len_pixels = 125
# y_len_pixels = 80
# multiple = window_size/2
# x_len_round = multiple *ceil(x_len_pixels / multiple)
# y_len_round = multiple *ceil(y_len_pixels / multiple)
# shape = (round((x_len_round-8)/8), round((y_len_round-8)/8))


# tile_list = gen_top_row(initial_tile, shape)
# tile_list, columns = gen_column(tile_list, shape)
# tile_list = gen_infill_column(tile_list, shape, columns)
# existing_tiles = [initial_tile]
# batches = []
# while True:
#     batch, existing_tiles = get_next_tiles(tile_list, existing_tiles, max_batch)
#     if not batch:
#         break
#     batches.append(batch)
# batches = gen_2d((100, 100, 100), 32, 24)
# print(batches[1])


# gear 250,250,150
# helical 280,280,150
# turbo-blade 320, 320, 150

# generate a plan with of 32x32x32 cubes
# lims = (80, 80, 80)
# window_size = 32
# batches = gen_center(lims, window_size, 32)

# print(len(batches))

# # save the plan (useful when generating large plans)
# with open('turbo-blade_batches.pickle', 'wb') as handle:
#     pickle.dump(batches, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # to plot plan
# plot_cubes(batches, window_size=window_size, lims=(lims[0]+window_size, lims[1]+window_size, lims[2]+window_size))

# all_batches = []
# for batch in batches:
#     for coord in batch_to_coords(batch, window_size):
#         all_batches.append(coord)
# max_coord = np.array(all_batches).max(axis=0)+32
# print(max_coord)
