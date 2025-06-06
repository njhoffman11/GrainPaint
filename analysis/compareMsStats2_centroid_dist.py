
"""
    This script returns a (saved) list of microstructures features for a microstructure given in *.npy format

    Parameters
    ----------
    fileName: name of the .npy 3d array microstructure

    Return
    ------
    (0) re-enumerated microstructure with unique grain IDs for grains at different locations (by DBSCAN)
    (1) grain size distribution pdfs
    (2) minimum volume enclosing ellipsoid given a cloud of points

"""

import time
import numpy as np
# from utils import *
from sklearn.cluster import DBSCAN
# import pyvista
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
from scipy.stats import gaussian_kde, entropy
from multiprocessing import Pool
import matplotlib.colors as mcolors
import random
from scipy.spatial import cKDTree

def reEnumerate(ms):
    '''
    ASSUMPTION: NON-PERIODIC
    This function
        (1) reads a microstructure, 
        (2) performs connectivity check,
        (3) re-assign grainId to clusters that have same grainId but are faraway
    Note: 
    (1) Need to verify against ParaView with thresholding
    (2) Clustering under periodic boundary condition: https://francescoturci.net/2016/03/16/clustering-and-periodic-boundaries/
    
    Preliminary results:
    DBSCAN works (well), Birch not, HDBSCAN only available in 1.3 (N/A), MeanShift may work if 'bandwidth' is tuned correctly (won't work w/ default param), AgglomerativeClustering works well with correctly estimated n_clusters

    Parameters
    ----------
    ms
    Return
    ------
    ms (updated)
    '''
    grainIdList = np.sort(np.unique(ms))
    maxGrainId = np.max(grainIdList)
    # Segregate grains with the same grainId by clustering
    for grainId in grainIdList:
        # Collect location information
        x,y,z = np.where(ms==grainId)
        # Combine to a 3-column array
        X = np.hstack((np.atleast_2d(x).T, np.atleast_2d(y).T, np.atleast_2d(z).T))
        # # Estimate maximum number of clusters # DEPRECATED for not being used
        # x_estimated_clusters = len(np.where(np.diff(np.sort(x)) > 2)[0])
        # y_estimated_clusters = len(np.where(np.diff(np.sort(y)) > 2)[0])
        # z_estimated_clusters = len(np.where(np.diff(np.sort(z)) > 2)[0])
        # estimated_clusters = np.max([x_estimated_clusters, y_estimated_clusters, z_estimated_clusters]) + 1
        # Perform clustering algorithm
        clustering = DBSCAN(eps=2, min_samples=10).fit(X)
        # print(f"n_clusters = {len(set(clustering.labels_))}")
        # Relabel grainId for every pixels needed relabel
        for j in range(clustering.labels_.shape[0]):
            ms[x[j],y[j],z[j]] = maxGrainId+clustering.labels_[j]+1
        # Update maxGrainId
        maxGrainId = np.max(np.sort(np.unique(ms)))
        if grainId % 100 == 0 and maxGrainId> 1000:
            print(f'Segregating grains for grainId {grainId} out of {maxGrainId}...')

    # Reorder
    print('Re-enumerating microstructure ... ', end='')
    grainIdList = np.sort(np.unique(ms))
    for i in range(len(grainIdList)):
        grainId = grainIdList[i]
        # Collect location information
        x,y,z = np.where(ms==grainId)
        for j in range(x.shape[0]):
            ms[x[j],y[j],z[j]] = i
    print('Done!')
    return ms

def mvee(points, tol=0.0001):
    '''
    This function returns a minimum volume enclosing ellipsoid given a cloud of points in n-space
    For the algorithm and reference, see: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=21c3072e516c93b28ccd06f5b994998abc517a7f
    For the MATLAB implementation, see: https://www.mathworks.com/matlabcentral/fileexchange/9542-minimum-volume-enclosing-ellipsoid
    For a Python implementation: see 
        (1) https://gist.github.com/Gabriel-p/4ddd31422a88e7cdf953
        (2) http://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    '''
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, np.linalg.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = np.linalg.norm(new_u-u)
        u = new_u
    c = np.dot(u, points)
    A = np.linalg.inv(np.dot(np.dot(points.T, np.diag(u)), points) - np.multiply.outer(c, c))/d
    return A, c

def fitEllipsoid(ms):
    '''
    This function fits an 3d ellipsoid to every grain and returns a set of statistics, including
    Parameters
    ----------
    ms
    Return
    ------
    ordered major/minor axis dimensions: a > b > c
    NOTE: the off-diagonal of matrix A is approximate 0 due to equiaxed and isotropic microstructures
    '''
    msDescList = [] # microstructure descriptor list
    grainIdList = np.sort(np.unique(ms))
    maxGrainId = np.max(grainIdList)
    for grainId in grainIdList:
        x,y,z = np.where(ms==grainId)
        # Combine to a 3-column array
        X = np.hstack((np.atleast_2d(x).T, np.atleast_2d(y).T, np.atleast_2d(z).T))
        A, c = mvee(X)
        tmpDim = np.sort([A[0,0], A[1,1], A[2,2]])
        msDescList += [[tmpDim[0], tmpDim[1], tmpDim[2]]]
        print(f'Done fitting grain {grainId} out of {maxGrainId}...')
    print('Done fitting ellipsoids for all grains!')
    msDescList = np.array(msDescList)
    return msDescList

def fitEllipsoidSVD(ms):
    '''
    This function fits a 3D ellipsoid to every grain using SVD and returns a set of statistics,
    including ordered major/minor axis dimensions: a > b > c.

    Parameters
    ----------
    ms : ndarray
        3D array representing the microstructure, where each unique value represents a different grain.

    Return
    ------
    msDescList : ndarray
        Array of ellipsoid axis lengths for each grain, ordered as a > b > c.
    '''
    msDescList = []  # microstructure descriptor list
    grainIdList = np.sort(np.unique(ms))
    maxGrainId = np.max(grainIdList)
    
    for grainId in grainIdList:
        x, y, z = np.where(ms == grainId)
        # Center the data
        X = np.vstack((x, y, z)).T
        center = X.mean(axis=0)
        X_centered = X - center
        
        # Compute the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        try:
            # Apply SVD
            U, singular_values, Vt = np.linalg.svd(covariance_matrix)
            
            # The singular values are related to the axes lengths of the ellipsoid
            axes_lengths = np.sqrt(singular_values)
            
            # Ensure the axes lengths are sorted in descending order
            axes_lengths_sorted = np.sort(axes_lengths)[::-1]
            
            msDescList.append(axes_lengths_sorted)
            print(f'Done fitting grain {grainId} out of {maxGrainId}...')
        except:
            pass
    
    print('Done fitting ellipsoids for all grains!')
    msDescList = np.array(msDescList)
    return msDescList

def computeGrainSizePdf(ms):
    grainSizeList = []
    grainIdList = np.sort(np.unique(ms))
    maxGrainId = np.max(grainIdList)
    for grainId in grainIdList:
        x, _, _ = np.where(ms == grainId)
        grainSize = x.shape[0]
        if x.shape[0] == 0:
            print(f'WARNING: grain {grainId} is empty!')
        grainSizeList += [grainSize]
        if grainId % 100 == 0 and maxGrainId> 1000:
            print(f'Collecting stats on grain {grainId} out of {maxGrainId}...')
            
    print('Done computing grain size pdf!')
    return grainSizeList

def nearestNeighborDistances(ms):
    grainIdList = np.sort(np.unique(ms))
    centroids = np.zeros((len(grainIdList), 3))
    
    for i, grainId in enumerate(grainIdList):
        indices = np.argwhere(ms == grainId)
        if len(indices) == 0:
            print(f'WARNING: grain {grainId} is empty!')
            continue
        centroids[i] = np.mean(indices, axis=0)
        if grainId % 100 == 0 and len(grainIdList) > 100:
            print(f'Calculating centroid for grain {grainId} out of {grainIdList[-1]}...')
    
    print('Done calculating centroids!')
    tree = cKDTree(centroids)
    distances, _ = tree.query(centroids, k=2)  # k=2 because the nearest neighbor includes the point itself
    return distances[:, 1]  # Skip the zero distance to itself

def reSampleGrainSize(grainSizeList,size=10000):
    '''
    This function resamples a grain size list with a specific number of samples to plot KDE approx. pdf more accurately
    '''
    logGrainSizeList = np.log(np.array(grainSizeList))
    resampledLogGrainSizeList = np.nan_to_num(gaussian_kde(logGrainSizeList.T)).resample(size, seed=0)
    resampeldGrainSizeList = list(np.exp(resampledLogGrainSizeList).ravel())
    return resampeldGrainSizeList

def reSampleAspectRatio(grainSizeList,size=10000):
    '''
    This function resamples a grain size list with a specific number of samples to plot KDE approx. pdf more accurately
    '''
    grainSizeList = grainSizeList.reshape(-1,2)
    array_filtered = grainSizeList[~(grainSizeList == 0).any(axis=1)]
    logGrainSizeList = np.log(np.array(array_filtered))
    resampledLogGrainSizeList = np.nan_to_num(gaussian_kde(logGrainSizeList.T)).resample(size, seed=0)
    resampeldGrainSizeList = list(np.exp(resampledLogGrainSizeList))
    return resampeldGrainSizeList

def hex_to_rgb(hex_color):
    '''Convert hex color to RGB.'''
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    '''Convert RGB color to hex.'''
    return '#' + ''.join(f'{int(c):02x}' for c in rgb_color)

def generate_similar_colors(base_hex, n, variation=30):
    '''
    Generate n random colors close to the given hex color.

    Args:
    - base_hex: The base hex color as a string (e.g., '#6d69d8').
    - n: Number of similar colors to generate.
    - variation: The maximum variation for each RGB component (default is 30).

    Returns:
    - A list of n hex color strings close to the base color.
    '''
    base_rgb = hex_to_rgb(base_hex)
    similar_colors = []

    for _ in range(n):
        new_rgb = tuple(
            max(0, min(255, base + random.randint(-variation, variation)))
            for base in base_rgb
        )
        similar_colors.append(rgb_to_hex(new_rgb))

    return similar_colors

def lighten_color(color, amount=0.5):
    '''
    Lighten a given color by a certain amount.

    :param color: Color to lighten (in RGB format)
    :param amount: Amount to lighten the color (0 to 1)
    :return: Lightened color in RGB format
    '''
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.ColorConverter().to_rgb(c)
    return mcolors.to_hex([1 - (1 - x) * amount for x in c])

def main_volume(ms, spparks_geo):
    # Insert a new axis if the input is a single microstructure
    if len(ms.shape) == 3:
        ms = np.expand_dims(ms, axis=0)
    if len(spparks_geo.shape) == 3:
        spparks_geo = np.expand_dims(spparks_geo, axis=0)

    # Convert to list for multiprocessing.pool
    ms_list = []
    for i in range(ms.shape[0]):
        ms_list.append(ms[i])
    for i in range(spparks_geo.shape[0]):
        ms_list.append(spparks_geo[i])

    # Compute grain size distribution using multiprocessing
    with Pool() as pool:
        grainSizeList = []
        # for grainSize in pool.imap(computeGrainSizePdf,ms_list):
        for grainSize in pool.imap(nearestNeighborDistances,ms_list):
            nearestNeighborDistances
            grainSizeList.append(grainSize)
    # grainSizeList = []
    # for i in range(len(ms_list)):
    #     grainSizeList.append(fitEllipsoidSVD(ms_list[i]))
    # Calculate average grain size distribution across generated and 
    # SPPARKS microstructures 
    grainSizeListAvg_gen = [[
        x
        for xs in grainSizeList[:ms.shape[0]]
        for x in xs
    ]]
    grainSizeListAvg_gen = np.concatenate(grainSizeListAvg_gen)
    grainSizeListAvg_spparks = [[
        x
        for xs in grainSizeList[ms.shape[0]:]
        for x in xs
    ]]
    grainSizeListAvg_spparks = np.concatenate(grainSizeListAvg_spparks)
    
    # Evaluate KDEs at the same points for all groups
    
    # kde_val_gen = np.zeros((ms.shape[0], len(x)))
    # kde_val_spparks = np.zeros((spparks_geo.shape[0], len(x)))
    # for i in range(kde_val_gen.shape[0]):
    #     # kde_val_gen[i, :] = gaussian_kde(grainSizeList[:ms.shape[0]][i])(x)
    #     kde_val_gen[i, :] = sns.kdeplot(grainSizeList[:ms.shape[0]][i], bw_adjust=.25, gridsize=3000, label='Group 1').get_lines()[0].get_data()[1]
    # for i in range(kde_val_spparks.shape[0]):
    #     # kde_val_spparks[i, :] = gaussian_kde(grainSizeList[ms.shape[0]:][i])(x)
    #     kde_val_spparks[i, :] = sns.kdeplot(grainSizeList[ms.shape[0]:][i], bw_adjust=.25, gridsize=3000, label='Group 1').get_lines()[0].get_data()[1]
       
    # Resample grain size
    grainSizesAll = []
    for i in range(len(grainSizeList)):
        grainSizesAll.append(reSampleGrainSize(grainSizeList[i]))
    grainSizeListAvg_gen_res = reSampleGrainSize(grainSizeListAvg_gen)
    grainSizeListAvg_spparks_res = reSampleGrainSize(grainSizeListAvg_spparks)
    grainSizesAll.append(grainSizeListAvg_gen_res)
    grainSizesAll.append(grainSizeListAvg_spparks_res)

    # Generate labels for plot
    labels = []
    for i in range(ms.shape[0]):
        labels.append(f'gen_{i}')
    for i in range(spparks_geo.shape[0]):
        labels.append(f'spparks_{i}')
    labels.append('Generated Mean')
    labels.append('SPPARKS Mean')

    # Generate colors for plot
    colors = generate_similar_colors('#6d69d8', ms.shape[0])
    colors += generate_similar_colors('#f44336', spparks_geo.shape[0])
    colors += generate_similar_colors('#6d69d8', 1)
    colors += generate_similar_colors('#f44336', 1)
    
    # Lighten colors for the non-averaged data
    for i in range(len(colors)-2):
        colors[i] = lighten_color(colors[i], amount=0.3)

    all_data = np.concatenate(grainSizesAll)
    x_min, x_max = np.min(all_data), np.max(all_data)
    x = np.linspace(x_min, x_max, 3000)

    # Prepare for collecting KDE values
    kde_val_gen = []
    kde_val_spparks = []

    plt.figure(figsize=(10, 6))

    # Plot and collect KDE data on the common x-axis range
    for label, color, grainSize in zip(labels, colors, grainSizesAll):
        # Compute KDE on the common x-axis range for each dataset
        kde = gaussian_kde(grainSize, bw_method=0.04)
        kde_values = kde(x)
        
        if label == 'Generated Mean':
            plt.plot(x, kde_values, label=label, color=color, linewidth=3, linestyle='dashed')
            kde_generated_mean = kde_values
        elif label == 'SPPARKS Mean':
            plt.plot(x, kde_values, label=label, color=color, linewidth=3, linestyle='dotted')
            kde_spparks_mean = kde_values
        elif label.split('_')[0] == 'gen':
            kde_val_gen.append(kde_values)
        elif label.split('_')[0] == 'spparks':
            kde_val_spparks.append(kde_values)

    if kde_generated_mean is not None and kde_spparks_mean is not None:
        kde_generated_mean /= np.trapz(kde_generated_mean, x)
        kde_spparks_mean /= np.trapz(kde_spparks_mean, x)
        epsilon = 1e-10  # A small constant
        kde_generated_mean += epsilon
        kde_spparks_mean += epsilon
        # Calculate KL divergence
        kl_div = entropy(kde_generated_mean, kde_spparks_mean)
        print(f'KL Divergence from Generated Mean to SPPARKS Mean: {kl_div}')

    # Calculate mean and standard deviation for the shaded areas
    kde_val_gen = np.array(kde_val_gen)
    kde_val_spparks = np.array(kde_val_spparks)

    kde_std_gen = np.std(kde_val_gen, axis=0)
    kde_std_spparks = np.std(kde_val_spparks, axis=0)

    plt.fill_between(x, kde_val_gen.mean(axis=0) - kde_std_gen, kde_val_gen.mean(axis=0) + kde_std_gen, 
                     color='#6d69d8', alpha=0.2, label='±1σ (Generated)')
    plt.fill_between(x, kde_val_spparks.mean(axis=0) - kde_std_spparks, kde_val_spparks.mean(axis=0) + kde_std_spparks, 
                     color='#f44336', alpha=0.2, label='±1σ (SPPARKS)')

    plt.legend(loc='best', fontsize='18')
    plt.xlabel('Nearest Centroid Distance [voxel]', fontsize=24)
    plt.ylabel('Density',fontsize=24)
    plt.xlim([0, 12])
    plt.ylim(bottom=0)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Set up plot parameters
    mpl.rcParams['xtick.labelsize'] = 24
    mpl.rcParams['ytick.labelsize'] = 24

    tStart = time.time()
    # For re-enumerating the spparks microstructures
    # aniso_geo_files = glob.glob('data_numpy/seed-2*.npy')[:30]
    # aniso_geos = np.zeros((len(aniso_geo_files), 100, 100, 100))
    # for i, aniso_geo_file in enumerate(aniso_geo_files):
    #     aniso_geos[i] = np.load(aniso_geo_file)

    # with Pool() as pool:
    #     geo_list = []
    #     for corrected_geo in pool.imap(reEnumerate,aniso_geos):
    #         geo_list.append(corrected_geo)
    # spparks_geo = np.array(geo_list)
    # np.save('corrected_potts_aniso_30.npy', spparks_geo)

    # load the microstructures
    # spparks_geo = np.load('analysis/corrected_potts_aniso_30.npy')[:,:80,:80,:80]
    spparks_geo = np.load('analysis/corrected_potts_10.npy') 
    # spparks_geo = reEnumerate(np.load('potts_3d.50.npy'))
    generated_geo = np.load('analysis/segmented_sub_volumes_new2.npy')[:100,:100,:100]
    main_volume(generated_geo, spparks_geo)

    tStop = time.time()
    print('Total time: %.2f' % (tStop - tStart))





