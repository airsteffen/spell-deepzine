""" Utilities for moving from internetarchive API to hdf5 file.
"""

import glob
import os
import tables
import internetarchive
import numpy as np
import math

from subprocess import call
from scipy.misc import imresize
from PIL import Image


# def internet_archive_download(destination_directory, collection='MBLWHOI', pdf_num=None):

#     """ Uses the internetarchive Python package to stream pdf pages from a given collection
#         into a provided destination_directory.
#     """

#     print('Beginning internet archive download...')

#     for i in internetarchive.search_items('collection:' + collection):

#         if pdf_num is not None:
#             if i == pdf_num:
#                 break

#         archive_id = i['identifier']
#         try:
#             if not os.path.exists(os.path.join(destination_directory, archive_id)):
#                 x = internetarchive.download(archive_id, verbose=True, glob_pattern='*.pdf', destdir=destination_directory)
#             elif os.listdir(os.path.join(destination_directory, archive_id)) == []:
#                 x = internetarchive.download(archive_id, verbose=True, glob_pattern='*.pdf', destdir=destination_directory)
#         except KeyboardInterrupt:
#             print('Cancelling download.')
#             break
#         except:
#             print('ERROR downloading', archive_id)
#     return


def convert_pdf_to_image(conversion_directory, output_directory, conversion_program='pdftoppm', pdftoppm_path='pdftoppm', ghostscript_path='gswin64c.exe"'):

    """ Converts a directory full of pdf files into png files using either the 
        external package pdftoppm or the external package ghostscript.
    """

    print('Beginning pdf image conversion.')

    documents = glob.glob(os.path.join(conversion_directory, '*/'))

    for document in documents:
        
        try:

            pdfs = glob.glob(os.path.join(document, '*.pdf'))
            document_basename = os.path.join(output_directory, os.path.basename(os.path.dirname(document)))

            first_page = glob.glob(document_basename + '*1.png')
            if first_page != []:
                print('Skipping', document_basename)
                continue

            for pdf in pdfs:

                print(pdf)

                if pdf.endswith('_bw.pdf'):
                    continue

                if conversion_program == 'pdftoppm':
                    command = pdftoppm_path + " " + pdf + " " + document_basename + " -png"
                elif conversion_program == 'ghostscript':
                    command = ghostscript_path + " -dBATCH -dNOPAUSE -sDEVICE=png16m -r144 -sOutputFile=" + document_basename + "-%d.png" + ' ' + pdf
                else:
                    print('Conversion program', conversion_program, 'not recognized, exiting.')
                    raise NotImplementedError

                call(command, shell=True)

        except KeyboardInterrupt:
            print('Cancelling pdf to image conversion.')
            break

    return


def create_hdf5_file(output_filepath, num_cases, output_sizes, preloaded=False):

    """ Creates a multi-tiered HDF5 file at each resolution provided in 'output_sizes'.
        Also stores string filepaths associated with the data.

        Big credit to https://github.com/ellisdg/3DUnetCNN for bringing HDF5 into
        my life.
    """

    hdf5_file = tables.open_file(output_filepath, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')

    hdf5_file.create_earray(hdf5_file.root, 'imagenames', tables.StringAtom(256), shape=(0, 1), filters=filters, expectedrows=num_cases)

    for output_size in output_sizes:
        hdf5_file.create_earray(hdf5_file.root, 'data_' + str(output_size[0]), tables.Float32Atom(), shape=(0,) + output_size, filters=filters, expectedrows=num_cases)

    return hdf5_file


def store_to_hdf5(data_directory, hdf5_filepath, output_size=64, verbose=True, preloaded=True):

    """ Stores a directory of images into an HDF5 file. Also resizes these images at every power
        of two between 4 and the output_size, provided that preloaded is set to True.
    """

    print('Beginning compression to HDF5')

    input_images = glob.glob(os.path.join(data_directory, '*.png'))

    if preloaded:
        output_sizes = [(4 * 2 ** x, 4 * 2 ** x, 3) for x in range(int(math.log(output_size, 2) - 1))]
    else:
        output_sizes = [(output_size, output_size, 3)]

    hdf5_file = create_hdf5_file(hdf5_filepath, num_cases=len(input_images), output_sizes=output_sizes)

    for image in input_images:
        try:

            if verbose:
                print('Storing...', image)

            img = Image.open(image)
            data = np.asarray(img, dtype=float)

            for output_size in output_sizes:
                if data.shape != output_size:
                    resized_data = imresize(data, (output_size[0], output_size[1]))
                else:
                    resized_data = data
                getattr(hdf5_file.root, 'data_' + str(output_size[0])).append(resized_data[np.newaxis] / 127.5 - 1)

            hdf5_file.root.imagenames.append(np.array(os.path.basename(image))[np.newaxis][np.newaxis])
        except KeyboardInterrupt:
            raise
        except:
            print('ERROR WRITING TO HDF5', image)

    hdf5_file.close()

    return hdf5_filepath


class PageData(object):

    """ An object for reading and writing from an HDF5 file created via the other
        methods in this script. It iterates through all the data in the HDF5, and
        shuffles the order once a full iteration is complete. Images at a given
        resolution can be requested.
    """

    def __init__(self, output_size=64, hdf5=None):

        self.hdf5 = hdf5
        self.image_num = getattr(self.hdf5.root, 'imagenames').shape[0]
        self.indexes = np.arange(self.image_num)
        np.random.shuffle(self.indexes)

        self.zoom_mapping = {idx + 1: 4 * 2 ** x for idx, x in enumerate(range(int(math.log(output_size, 2) - 1)))}

    def get_next_batch(self, batch_num=0, batch_size=64, zoom_level=1):

        total_batches = self.image_num // batch_size - 1

        if batch_num % total_batches == 0:
            np.random.shuffle(self.indexes)

        indexes = self.indexes[(batch_num % total_batches) * batch_size: (batch_num % total_batches + 1) * batch_size]

        data = np.array([getattr(self.hdf5.root, 'data_' + str(self.zoom_mapping[zoom_level]))[idx] for idx in indexes])
        return data

    def close(self):

        self.hdf5.close()


if __name__ == '__main__':

    pass