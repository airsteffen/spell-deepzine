import numpy as np
import scipy


def add_parameter(class_object, kwargs, parameter, default=None):
    
    """ I find the typical way of adding parameters to classes a little opaque,
        so I added this method. 
    
    Parameters
    ----------
    class_object : Object
        Object to assign attribute to.
    kwargs : dict
        Parameters passed to object.
    parameter : str
        Name of attribute
    default : None, optional
        Default value of attribute.
    """

    if parameter in kwargs:
        setattr(class_object, parameter, kwargs.get(parameter))
    else:
        setattr(class_object, parameter, default)


def merge(images, size, channels=3):
    
    """I grabbed this code from someone else, but now I don't remember where :(.
    
    Parameters
    ----------
    images : array
        [batch_size x row x column x RGB] input array
    size : tuple
        row x column input tuple, specifying dimensions of the mosaic
    channels : int, optional
        Number of channels in image.
    
    Returns
    -------
    array
        Description
    
    """

    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], channels))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img


def save_grid_images(images, size, path):
    
    """ 
    """

    if images.shape[-1] == 3:
        return scipy.misc.imsave(path, merge(images, size))
    elif images.shape[-1] == 1:
        scipy.misc.imsave(path, np.squeeze(merge(images[..., 0][..., np.newaxis], size, channels=1)))


def inverse_transform(image):
    
    """ Reverse intensity normalization for image rendering purposes.
    """
    
    return ((image + 1.) * 127.5).astype(np.uint8)


def save_images(images, size, image_path):

    """Saves a batch of images in [batch_size, row, column, RGB] format into
    a composite mosaic image.
    
    Parameters
    ----------
    images : array
        [batch_size x row x column x RGB] input array
    size : tuple
        row x column input tuple, specifying dimensions of the mosaic
    image_path : str
        Output image filepath to save to.
    """

    data = inverse_transform(images)
    save_grid_images(data, size, image_path)


def save_image(data, image_path):

    """ Just a wrapper around scipy /shrug
    """
    
    return scipy.misc.imsave(image_path, data)


if __name__ == '__main__':

    pass