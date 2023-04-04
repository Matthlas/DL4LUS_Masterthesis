import numpy as np
from PIL import Image
from skimage.transform import resize
import os

def scale_down(array, size):
    """Scale down an array using bilinear interpolation.

    Parameters
    ----------
    array : numpy.ndarray
        The array to scale down.
    size : tuple
        The size of the scaled array.

    Returns
    -------
    numpy.ndarray
        The scaled array.
    """
    
    # Compute the scaling factor
    height, width = array.shape
    scaling_factor = min(size[0] / height, size[1] / width)
    
    # Compute the size of the scaled image
    scaled_height = int(height * scaling_factor)
    scaled_width = int(width * scaling_factor)
    
    # Create a scaled image using bilinear interpolation
    scaled_array = np.empty((scaled_height, scaled_width))
    for i in range(scaled_height):
        for j in range(scaled_width):
            # Compute the coordinates of the corresponding pixel in the original image
            x = i / scaling_factor
            y = j / scaling_factor
            
            # Get the four nearest pixels in the original image
            x0 = int(np.floor(x))
            x1 = x0 + 1
            y0 = int(np.floor(y))
            y1 = y0 + 1
            
            # Check if the coordinates are within the bounds of the original image
            if x0 < 0 or x0 >= height or y0 < 0 or y0 >= width or x1 < 0 or x1 >= height or y1 < 0 or y1 >= width:
                continue
            
            # Compute the weights for each pixel
            wx1 = x1 - x
            wx0 = x - x0
            wy1 = y1 - y
            wy0 = y - y0
            
            # Compute the value of the pixel in the scaled image using bilinear interpolation
            scaled_array[i, j] = (array[x0, y0] * wx1 * wy1 + array[x1, y0] * wx0 * wy1 + array[x0, y1] * wx1 * wy0 + array[x1, y1] * wx0 * wy0)
    
    return scaled_array

def get_class(array, seg_class):
    """Extracts a specific cluster from the segmentation mask.

    Parameters
    ----------
    array : numpy.ndarray
        The segmentation mask.
    seg_class : int
        The class of the segmentation mask.

    Returns
    -------
    numpy.ndarray
        The cluster.
    """

    # Check if seg_class is a valid value (1,2,3,4)
    if seg_class < 1 or seg_class > 4:
        raise ValueError("Cluster must be a value between 1 and 4")
    # Create a copy of the array to avoid modifying the original array
    array = array.copy()
    # Set all the values that are not in the cluster to 0
    array[array != seg_class] = 0
    # Set all the values that are in the cluster to 1
    array[array == seg_class] = 1

    return array

def get_classes(array):
    """Extracts all the clusters from the segmentation mask.

    Parameters
    ----------
    array : numpy.ndarray
        The segmentation mask.

    Returns
    -------
    dict
        A dictionary with the clusters.
    """

    # Create a dict to store the clusters
    classes = {}
    
    # Iterate over all the clusters
    for i in range(1, 5):
        # Extract the cluster
        cluster = get_class(array, i)
        # Store the cluster in the dictionary
        classes[i] = cluster
    
    return classes

def normalize_pc(points):
    """Normalize a point cloud. 
    The point cloud is translated to the origin and scaled such that the furthest point is at a distance of 1 from the origin.

    Parameters
    ----------
    points : numpy.ndarray
        The point cloud.

    Returns
    -------
    numpy.ndarray
        The normalized point cloud.
    """


    # Check if the points consist only of one point. This is the case if the respective class is not present in the video. In this case, the point is simply returned.
    if len(points) == 1:
        return points
    
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance

    return points



#################### CUBICAL COMPLEX ####################

def load_gif(path):
    frame_list = []
    with Image.open(os.path.join(path)) as im:
        im.seek(1)  # skip to the second frame

        try:
            while 1:
                im.seek(im.tell() + 1)
                frame = im.copy()
                # Convert to grayscale
                frame = frame.convert('L')
                frame_list.append(np.array(frame))
        except EOFError:
            pass  # end of sequence
    
    return np.array(frame_list)



def filter_frame(frame, mask, resize_factor=10):
    # Offest mask by 500 pixels to the left to match the unprocessed LUS image
    mask = np.roll(mask, -500, axis=1)

    # Only take the left half of the gif and the mask
    frame = frame[:, :1000]
    mask = mask[:, :1000]

    classes = [1, 2, 3, 4]
    class__filtered_dict = {}
    for c in classes:
        # Set all values in the gif to 0 where the mask is not equal to the class
        class_filtered_frame = frame.copy()
        class_filtered_frame[mask != c] = 0
        # Resize by a factor of resize_factor
        class_filtered_frame = resize(class_filtered_frame,
                                    (class_filtered_frame.shape[0] // resize_factor, class_filtered_frame.shape[1] // resize_factor),
                                    anti_aliasing=True)
        class__filtered_dict[c] = class_filtered_frame

    return class__filtered_dict

def crop_volume(volume):
    # This creates a minimal cropping of a 3D volume around the non-zero values
    # This will result in differently sized arrays for each class and video
    # This works because we calculate the persistence diagrams and then transform them into a scale invariant feature vector

    # Check if the volume contains only zeros. If so, return a minimal 1x1x1 array with a value of 0 so the persistence diagram can be "calculated"
    if np.count_nonzero(volume) == 0:
        return np.zeros((1, 1, 1))

    # Find the indices of the first and last non-zero planes along each axis
    x_start, x_end = np.nonzero(volume.sum(axis=(1,2)))[0][[0, -1]] + 1
    y_start, y_end = np.nonzero(volume.sum(axis=(0,2)))[0][[0, -1]] + 1
    z_start, z_end = np.nonzero(volume.sum(axis=(0,1)))[0][[0, -1]] + 1

    # Use the indices to slice the array and remove the zero planes
    volume_cropped = volume[x_start:x_end, y_start:y_end, z_start:z_end]
    return volume_cropped

def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    https://stackoverflow.com/questions/55917328/numpy-trim-zeros-in-2d-or-3d
    Extended to check for zeros arrays
    """
    if np.count_nonzero(arr) == 0:
        return np.zeros((1, 1, 1))
        
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices]