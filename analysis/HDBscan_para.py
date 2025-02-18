import numpy as np
from sklearn.cluster import DBSCAN
# from cuml.cluster import DBSCAN
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool 
from collections import Counter
from random import choice
import random
from glob import glob
import pickle

def replace_with_neighbors(arr):
    # Define the 6-connectivity relative positions
    neighbors = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]
    
    # Iterate through the array
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                if arr[i, j, k] == -1:
                    neighbor_values = []
                    # Check all the neighbors
                    for dx, dy, dz in neighbors:
                        ni, nj, nk = i + dx, j + dy, k + dz
                        # Make sure we don't go out of bounds
                        if 0 <= ni < arr.shape[0] and 0 <= nj < arr.shape[1] and 0 <= nk < arr.shape[2]:
                            neighbor_value = arr[ni, nj, nk]
                            if neighbor_value != -1:
                                neighbor_values.append(neighbor_value)
                    
                    if neighbor_values:
                        # Count occurrences of each neighbor value
                        value_counts = Counter(neighbor_values)
                        # Find the most common value(s)
                        max_count = max(value_counts.values())
                        most_common_values = [value for value, count in value_counts.items() if count == max_count]
                        
                        # Choose a random one if there's a tie
                        arr[i, j, k] = choice(most_common_values)
                    # If there are no valid neighbors (which should not normally happen), do nothing
                    # or you might want to handle this case separately
    return arr

def cluster_voxels_small_grain(volume):
    # 1. Get the shape of the original volume
    original_shape = volume.shape

    # 2. Create a list of voxels (x, y, z, val)
    xx, yy, zz = np.mgrid[0:original_shape[0], 0:original_shape[1], 0:original_shape[2]]
    voxels = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel(), volume.ravel()*np.array(original_shape).mean()))

    # 3. Apply HDBSCAN clustering
    # clusterer = DBSCAN(eps=2.5, min_samples=30)
    # clusterer = DBSCAN(eps=2.2, min_samples=30)
    # clusterer = DBSCAN(eps=1.5, min_samples=10)  # Use for Aniso
    clusterer = DBSCAN(eps=1.9, min_samples=20)  # Use for iso
    # clusterer = DBSCAN(eps=1.4, min_samples=2)
    # clusterer = KMeans(100)
    labels = clusterer.fit_predict(voxels)

    # 4. Reshape labels back to original volume shape
    clustered_volume = labels.reshape(original_shape)

    # return clustered_volume.astype(np.int32)
    return replace_with_neighbors(clustered_volume).astype(np.int32)

def cluster_voxels_large_grain(volume):
    # 1. Get the shape of the original volume
    original_shape = volume.shape

    # 2. Create a list of voxels (x, y, z, val)
    xx, yy, zz = np.mgrid[0:original_shape[0], 0:original_shape[1], 0:original_shape[2]]
    voxels = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel(), volume.ravel()*np.array(original_shape).mean()))

    # 3. Apply HDBSCAN clustering
    clusterer = DBSCAN(eps=1.1, min_samples=2)  # Adjust parameters as needed
    # clusterer = KMeans(100)
    labels = clusterer.fit_predict(voxels)

    # 4. Reshape labels back to original volume shape
    clustered_volume = labels.reshape(original_shape)

    # return clustered_volume.astype(np.int32)
    return replace_with_neighbors(clustered_volume).astype(np.int32)

def divide_volume(volume, sub_volume_size, overlap):
    # This function divides the volume into overlapping sub-volumes.
    # Each sub-volume has an overlap with its neighbors.
    sub_volumes = []
    coords = []
    x_coors = np.arange(0,volume.shape[0],sub_volume_size-overlap)
    x_coors = np.delete(x_coors, -1)
    x_coors = np.concatenate((x_coors, [volume.shape[0] - sub_volume_size]))

    y_coors = np.arange(0,volume.shape[1],sub_volume_size-overlap)
    y_coors = np.delete(y_coors, -1)
    y_coors = np.concatenate((y_coors, [volume.shape[1] - sub_volume_size]))

    z_coors = np.arange(0,volume.shape[2],sub_volume_size-overlap)
    z_coors = np.delete(z_coors, -1)
    z_coors = np.concatenate((z_coors, [volume.shape[2] - sub_volume_size]))
    
    for x in x_coors:
        for y in y_coors:
            for z in z_coors:
                sub_volume = volume[
                    x : x + sub_volume_size,
                    y : y + sub_volume_size,
                    z : z + sub_volume_size
                ]
                if sub_volume.shape == (sub_volume_size, sub_volume_size, sub_volume_size):
                    coords.append((x,y,z))
                    sub_volumes.append(sub_volume)
                
    return sub_volumes, coords

def remap_labels(segmented_sub_volumes):
    # This function remaps the labels in each segmented sub-volume to a global label set.
    remapped_sub_volumes = []
    label_offset = 0
    for sub_volume in segmented_sub_volumes:
        max_label = sub_volume.max()
        sub_volume[sub_volume > 0] += label_offset
        label_offset += max_label
        remapped_sub_volumes.append(sub_volume)
    return remapped_sub_volumes, label_offset

import numpy as np

def is_grain_touching_edge(grain, sub_volume_shape):
    """
    Check if a grain is touching the edge of the sub_volume.
    """
    z, y, x = np.where(grain > 0)
    return (
        np.any(x == 0) or np.any(x == sub_volume_shape[2] - 1) or
        np.any(y == 0) or np.any(y == sub_volume_shape[1] - 1) or
        np.any(z == 0) or np.any(z == sub_volume_shape[0] - 1)
    )

def is_grain_on_volume_edge(coord, grain, volume_shape, sub_volume_shape):
    """
    Check if the grain is on the edge of the sub_volume and that edge is also the edge of the whole volume.
    """
    z, y, x = np.where(grain > 0)
    z = z + coord[0]
    y = y + coord[1]
    x = x + coord[2]
    return (
        np.any(x == 0) or np.any(x == volume_shape[2] - 1) or
        np.any(y == 0) or np.any(y == volume_shape[1] - 1) or
        np.any(z == 0) or np.any(z == volume_shape[0] - 1)
    )

from collections import defaultdict

def get_sub_volumes_containing_voxels(voxel_coords, coords, sub_volume_size):
    """
    Find all sub_volumes that contain the specified voxels.
    """
    sub_volumes_indices = set()
    grain_ids = set()
    for i, coord in enumerate(coords):
        z_min, y_min, x_min = coord
        z_max, y_max, x_max = z_min + sub_volume_size, y_min + sub_volume_size, x_min + sub_volume_size

        for voxel_coord in voxel_coords:
            z, y, x = voxel_coord
            if z_min <= z < z_max and y_min <= y < y_max and x_min <= x < x_max:
                sub_volumes_indices.add(i)
                grain_ids.add(voxel_coord)
                break  # No need to check other voxels if one is found in the sub_volume

    return sub_volumes_indices

# @profile
def combine_and_assign_grains(assembled_volume, grain_maps, grain_ids, combined_grain_label):
    """
    Combine grains specified by grain_ids, assign them a combined label, and insert into assembled_volume.
    Only update voxels in assembled_volume that are currently set to zero.
    """
    combined_grain_mask = np.zeros_like(assembled_volume, dtype=bool)
    for grain_id in grain_ids:
        combined_grain_mask |= grain_maps[grain_id]
        # grain_maps[grain_id] = np.zeros_like(assembled_volume, dtype=bool)  # Clear the grain map after combining
    
    # Only update voxels in assembled_volume that are currently set to zero
    update_mask = combined_grain_mask & (assembled_volume == 0)
    assembled_volume[update_mask] = combined_grain_label

def get_next_slice(slice_origin, slice_size, volume_shape, step_size):
    """
    Compute the origin of the next slice to process, advancing in a tile pattern.
    Adjust the slice origin to stay within the bounds of the volume.
    """
    print("next_slice")
    z, y, x = slice_origin
    if x + 2*step_size < volume_shape[2]:
        x += step_size
    elif x < volume_shape[2] - slice_size[2]:
        x = volume_shape[2] - slice_size[2]
    else:
        x = 0
        if y + 2*step_size < volume_shape[1]:
            y += step_size
        elif y < volume_shape[1] - slice_size[1]:
            y = volume_shape[1] - slice_size[1]
        else:
            y = 0
            if z + 2*step_size < volume_shape[0]:
                z += step_size
            elif z < volume_shape[0] - slice_size[0]:
                z = volume_shape[0] - slice_size[0]
            else:
                return None  # Indicates that the entire volume has been processed

    return z, y, x

def remap_grain_ids(assembled_volume):
    """
    Remap grain IDs in the assembled volume to a new set of random IDs.
    """
    unique_ids = np.unique(assembled_volume)
    # Create a list of new IDs by shuffling the unique IDs
    new_ids = unique_ids.copy()
    random.shuffle(new_ids)

    # Create a mapping from old IDs to new (random) IDs
    remapping = dict(zip(unique_ids, new_ids))

    # Create a new volume to store the remapped values
    remapped_volume = np.zeros_like(assembled_volume)

    # Apply the remapping
    for old_id, new_id in remapping.items():
        remapped_volume[assembled_volume == old_id] = new_id

    return remapped_volume

def assemble_volume(segmented_sub_volumes, coords, volume_shape):
    assembled_volume = np.zeros(volume_shape, dtype=int)
    voxel_presence = np.zeros(volume_shape, dtype=int)
    grain_maps = defaultdict(lambda: np.zeros(volume_shape, dtype=bool))
    voxel_to_grains = defaultdict(list)

    # First pass: Populate voxel_presence, grain_maps, and voxel_to_grains
    for sub_volume, coord in tqdm(zip(segmented_sub_volumes, coords)):
        for z in range(sub_volume.shape[0]):
            for y in range(sub_volume.shape[1]):
                for x in range(sub_volume.shape[2]):
                    grain_label = sub_volume[z, y, x]
                    if grain_label > 0:
                        global_coord = (coord[0] + z, coord[1] + y, coord[2] + x)
                        voxel_presence[global_coord] += 1
                        grain_maps[grain_label][global_coord] = True
                        voxel_to_grains[global_coord].append(grain_label)

    # Main assembly loop
    grain_label_counter = 1
    voxel_remaining = -1
    voxel_remaining_prev = 0
    slice_size=(50, 50, 50)
    step_size=50
    zero_voxel_threshold=100
    slice_origin = (0, 0, 0)
    z, y, x = slice_origin
    current_slice = assembled_volume[z:z+slice_size[0], y:y+slice_size[1], x:x+slice_size[2]]
    prev_usage = {}
    while True:
        # Pick a random voxel with value 0

        zero_voxels = np.argwhere(current_slice == 0)
        if grain_label_counter % 100 == 0:
            # local_vars = {var: obj for var, obj in globals().items() if var not in ["prev_usage", "gc", "sys"]}
            # prev_usage = report_memory_usage(local_vars, prev_usage)
            voxel_remaining_prev = voxel_remaining
            voxel_remaining = len(zero_voxels)
        if voxel_remaining == voxel_remaining_prev or len(zero_voxels) == 0:
            slice_origin = get_next_slice(slice_origin, slice_size, volume_shape, step_size)
            if slice_origin is None:
                break
            z,y,x = slice_origin
            current_slice = assembled_volume[z:z+slice_size[0], y:y+slice_size[1], x:x+slice_size[2]]
            zero_voxels = np.argwhere(current_slice == 0)
            voxel_remaining = len(zero_voxels)
        
        random_voxel = zero_voxels[random.randint(0, len(zero_voxels) - 1)]

        # if random_voxel in voxel_to_grains:
        random_voxel[0] += z
        random_voxel[1] += y
        random_voxel[2] += x
        grain_ids = voxel_to_grains[tuple(random_voxel)]

        # Combine grains and assign them a new label
        try:
            grain_voxel_coords = np.argwhere(grain_maps[grain_ids[0]] == True)
            x_min = np.min(grain_voxel_coords[:, 0])
            y_min = np.min(grain_voxel_coords[:, 1])
            z_min = np.min(grain_voxel_coords[:, 2])
            x_max = np.max(grain_voxel_coords[:, 0])
            y_max = np.max(grain_voxel_coords[:, 1])
            z_max = np.max(grain_voxel_coords[:, 2])
            x_min_coord = tuple(grain_voxel_coords[np.argwhere(grain_voxel_coords[:,0]==x_min)[0,0]])
            x_max_coord = tuple(grain_voxel_coords[np.argwhere(grain_voxel_coords[:,0]==x_max)[0,0]])
            y_min_coord = tuple(grain_voxel_coords[np.argwhere(grain_voxel_coords[:,1]==y_min)[0,0]])
            y_max_coord = tuple(grain_voxel_coords[np.argwhere(grain_voxel_coords[:,1]==y_max)[0,0]])
            z_min_coord = tuple(grain_voxel_coords[np.argwhere(grain_voxel_coords[:,2]==z_min)[0,0]])
            z_max_coord = tuple(grain_voxel_coords[np.argwhere(grain_voxel_coords[:,2]==z_max)[0,0]])
            grain_voxel_coords = [x_min_coord, x_max_coord, y_min_coord, y_max_coord, z_min_coord, z_max_coord]
            possible_grain_ids = set(grain_ids)
            for grain_voxel_coord in grain_voxel_coords:
                possible_grain_ids = set.union(possible_grain_ids, set(voxel_to_grains[grain_voxel_coord]))
            combine_and_assign_grains(assembled_volume, grain_maps, list(possible_grain_ids), grain_label_counter)
            grain_label_counter += 1
            
        except:
            grain_label_counter += 1
        

        # if grain_label_counter % 100 == 0:
        #     voxel_remaining_prev = voxel_remaining
        #     voxel_remaining = np.sum(current_slice == 0)
        #     # print(voxel_remaining)
        #     print(len(zero_voxels))
        #     # plt.imshow(assembled_volume[0])
        #     # plt.title("Clustered")
        #     # plt.savefig("Clustered_HDBSCAN_{}_normal.png".format(grain_label_counter))
        #     if voxel_remaining_prev == voxel_remaining:
        #         print('voxel_remain break')
                
        #         break
    # plt.figure(3)
    # plt.imshow(assembled_volume[0])
    # plt.title("Clustered")
    # # plt.save("Clustered_HDBSCAN.png")
    # plt.show()
    assembled_volume[assembled_volume == 0] = -1
    assembled_volume = replace_with_neighbors(assembled_volume)
    remapped_volume = remap_grain_ids(assembled_volume)
    return remapped_volume

    # # Combine grains and update the grain_maps
    # for grain_label in tqdm(grains_to_combine):
    #     grain = grain_maps[grain_label]
    #     grain_voxel_coords = np.array(np.where(grain)).T
    #     sub_volumes_indices = get_sub_volumes_containing_voxels(grain_voxel_coords, coords, 30)
    #     combined_grain = combine_grains(grain_maps, sub_volumes_indices, segmented_sub_volumes, grain_label)
    #     grain_maps[grain_label] = combined_grain

    # # Final assembly
    # for grain_label, grain in second_grain_maps.items():
    #     if np.any(grain):
    #         # assembled_volume[grain] = grain_label
    #         assembled_volume[grain] = np.random.randint(1, 1000)

    # return assembled_volume
# import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == "__main__":
    # Load diffusion model output
    files = glob('analysis/unclustered/*7.npy')
    geo_pieces = []
    for file in files:
        geo_pieces.append(np.load(file))
    geo_pieces = np.array(geo_pieces)

    # Step 2: Cluster Each volume
    # geo_pieces = geo_pieces
    with Pool(16) as pool:

        segmented_sub_volumes_1 = []
        segmented_sub_volumes_2 = []
        # for segmented_sub_volume in tqdm(pool.imap(cluster_voxels_large_grain,geo_pieces), total=len(geo_pieces)):
        #     segmented_sub_volumes_1.append(segmented_sub_volume)
        for segmented_sub_volume in tqdm(pool.imap(cluster_voxels_small_grain,geo_pieces), total=len(geo_pieces)):
            segmented_sub_volume = remap_grain_ids(segmented_sub_volume)
            segmented_sub_volumes_2.append(segmented_sub_volume)


    # segmented_sub_volumes_1 = np.array(segmented_sub_volumes_1)
    segmented_sub_volumes_2 = np.array(segmented_sub_volumes_2)
    # for i in range(segmented_sub_volumes_2.max()):
    #     if np.where(segmented_sub_volumes_2[0] == i)[0].shape[0] < 5:
    #         segmented_sub_volumes_1[segmented_sub_volumes_2 == i] = segmented_sub_volumes_1.max() + 1
    
    plt.imshow(segmented_sub_volumes_2[0,0])
    plt.title("Clustered")
    plt.figure(2)
    plt.imshow(geo_pieces[0,0])
    plt.title("Original")
    plt.show()


    np.save('analysis/segmented_sub_volumes_iso_1res.npy', segmented_sub_volumes_2)

# geo100 = np.load('/media/nathanielhoffman/EXOS18TB/generated_ms/run_35_batch_0.npy')
# vec = np.concatenate([np.ones(32), np.zeros(16), np.ones(32), np.zeros(16), np.ones(32)])
# stripes = [vec]*vec.shape[0]
# stripes = np.array(stripes)
# stripes = stripes * stripes.T
# stripes2 = np.pad(stripes, ((24, 0), (24,0)))[:-24, :-24]
# mask = stripes+stripes2>0
# ax = plt.gca()
# ax.axis('off')
# plt.imshow(geo33[0,:100,:100], alpha=mask[:100,:100]*.99)