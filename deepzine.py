
import os
import tables
import numpy as np
import math

# from download_internet_archive import internet_archive_download, convert_pdf_to_image, store_to_hdf5, PageData
from download_internet_archive import PageData

from utils import add_parameter
from model import PGGAN


class DeepZine(object):

    def __init__(self, **kwargs):

        # Execution Parameters
        add_parameter(self, kwargs, 'load_data', False)
        add_parameter(self, kwargs, 'train', False)
        add_parameter(self, kwargs, 'inference', False)
        add_parameter(self, kwargs, 'interpolation', False)

        # Data Loading Parameters
        add_parameter(self, kwargs, 'data_hdf5', None)
        add_parameter(self, kwargs, 'pdf_directory', None)
        add_parameter(self, kwargs, 'image_directory', None)
        add_parameter(self, kwargs, 'overwrite', False)
        add_parameter(self, kwargs, 'pdf_conversion_program', 'pdftoppm')

        add_parameter(self, kwargs, 'download_pdf', True)
        add_parameter(self, kwargs, 'internetarchive_collection', 'MBLWHOI')
        add_parameter(self, kwargs, 'convert_pdf', True)
        
        add_parameter(self, kwargs, 'pdf_num', None)
        add_parameter(self, kwargs, 'data_output_size', 1024)
        add_parameter(self, kwargs, 'preload_resized_data', True)

        # Training GAN Parameters
        add_parameter(self, kwargs, 'samples_dir', './samples')
        add_parameter(self, kwargs, 'log_dir', './log')
        add_parameter(self, kwargs, 'starting_depth', None)
        add_parameter(self, kwargs, 'progressive_depth', None)
        add_parameter(self, kwargs, 'gan_starting_size', 4)
        add_parameter(self, kwargs, 'gan_output_size', 128)

        # Inference Parameters
        add_parameter(self, kwargs, 'inference_output_format', 'png')
        add_parameter(self, kwargs, 'inference_batch_size', 16)
        add_parameter(self, kwargs, 'inference_model_directory', None)
        add_parameter(self, kwargs, 'inference_output_directory', None)
        add_parameter(self, kwargs, 'inference_model_path', None)
        add_parameter(self, kwargs, 'inference_output_num', 100)
        add_parameter(self, kwargs, 'inference_input_latent', None)

        # Latent Space Interpolation Parameters
        add_parameter(self, kwargs, 'interpolation_method', 'slerp')
        add_parameter(self, kwargs, 'interpolation_latents', None)
        add_parameter(self, kwargs, 'interpolation_frames', 100)
        add_parameter(self, kwargs, 'interpolation_vector_num', 10)

        # Derived Parameters
        if self.progressive_depth is None:
            self.progressive_depth = int(math.log(self.gan_output_size, 2) - 1)
        if self.starting_depth is None:
            self.starting_depth = int(math.log(self.gan_starting_size, 2) - 1)

        if self.gan_output_size is None:
            self.gan_output_size = 2 * 2 ** self.progressive_depth
        if self.gan_starting_size is None:
            self.gan_starting_size = 2 * 2 ** self.starting_depth

        self.training_storage = None

        self.kwargs = kwargs

        return

    def execute(self):

        if self.train or self.load_data:

            # Data preparation.
            self.training_storage = self.download_data()

            if self.train:
            
                try:
                    self.train_gan()
                except:
                    self.close_storage()
                    raise

                self.close_storage()

        if self.inference:

            self.inference_gan()

        if self.interpolation:

            self.interpolate_gan()

        self.close_storage()

        return

    def close_storage(self):

        if self.training_storage is not None:
            self.training_storage.close()
            self.training_storage = None

    def download_data(self):

        # Check if an HDF5 exists, otherwise initiate the process of creating one.
        if self.data_hdf5 is None:
            raise ValueError('Please provide an HDF5 file to stream data from.')
        else:
            if os.path.exists(self.data_hdf5) and not self.overwrite:
                output_hdf5 = self.data_hdf5
            else:
                output_hdf5 = None

        # if output_hdf5 is None:

        #     # Create a working data_directory if necessary.
        #     if not os.path.exists(self.pdf_directory) and not self.download_pdf:
        #         raise ValueError('Data directory not found.')
        #     elif not os.path.exists(self.pdf_directory):
        #         os.mkdir(self.pdf_directory)

        #     # Download data
        #     # ---------------------------------------------
        #     # if self.download_pdf:
        #     #     internet_archive_download(self.pdf_directory, self.internetarchive_collection, self.pdf_num)

        #     # Convert PDFs into images.
        #     # ---------------------------------------------
        #     # if self.convert_pdf:
        #     #     if not os.path.exists(self.image_directory):
        #     #         os.mkdir(self.image_directory)
        #     #     convert_pdf_to_image(self.pdf_directory, self.image_directory, conversion_program=self.pdf_conversion_program)

        #     # Preprocess images and write to HDF5.
        #     # output_hdf5 = store_to_hdf5(self.image_directory, self.data_hdf5, self.data_output_size)

        # output_hdf5 = tables.open_file(output_hdf5, "r")

        # Convert to data-loading object. The logic is all messed up here for pre-loading images.
        return PageData(hdf5=output_hdf5, output_size=self.gan_output_size)

    def train_gan(self):

        # Create necessary directories
        for work_dir in [self.samples_dir, self.log_dir]:
            if not os.path.exists(work_dir):
                os.mkdir(work_dir)

        # Some explanation on training stages: The progressive gan trains each resolution
        # in two stages. One interpolates from the previous resolution, while one trains 
        # solely on the current resolution. The loop below looks odd because the lowest 
        # resolution only has one stage.

        training_stages = range(int(np.ceil((self.starting_depth) / 2)) - 2, (self.progressive_depth * 2) - 2)

        for training_stage in training_stages:

            if (training_stage % 2 == 1):
                transition = False
                transition_string = ''
            else:
                transition = True
                transition_string = '_Transition'

            current_depth = np.ceil((training_stage + 1) / 2)
            previous_depth = np.ceil((training_stage) / 2)

            current_size = int(4 * 2 ** current_depth)
            previous_size = int(4 * 2 ** previous_depth)

            output_model_path = os.path.join(self.log_dir, str(current_size), 'model.ckpt')
            if not os.path.exists(os.path.dirname(output_model_path)):
                os.mkdir(os.path.dirname(output_model_path))

            input_model_path = os.path.join(self.log_dir, str(previous_size), 'model.ckpt')

            sample_path = os.path.join(self.samples_dir, 'samples_' + str(current_size) + transition_string)
            if not os.path.exists(sample_path):
                os.mkdir(sample_path)

            print(input_model_path, output_model_path, sample_path)

            pggan = PGGAN(training_data=self.training_storage,
                            input_model_path=input_model_path, 
                            output_model_path=output_model_path,
                            model_sample_dir=sample_path, 
                            model_logging_dir=self.log_dir,
                            model_output_size=current_size,
                            transition=transition,
                            **self.kwargs)

            pggan.build_model()
            pggan.train()

    def load_model(self):

        if self.inference_model_path is None:

            if self.inference_model_directory is not None:
                self.inference_model_path = os.path.join(self.inference_model_directory, str(self.gan_output_size))
            else:
                print('No model given for inference!')
                raise

        self.inference_model_path = os.path.join(self.inference_model_path, 'model.ckpt')

    def inference_gan(self, input_latent=None):

        if not os.path.exists(self.inference_output_directory):
            os.mkdir(self.inference_output_directory)

        self.load_model()

        pggan = PGGAN(input_model_path=self.inference_model_path,
                        model_output_size=self.gan_output_size,
                        inference_mode=True,
                        **self.kwargs)

        pggan.build_model()

        pggan.model_inference(self.inference_output_directory, output_num=self.inference_output_num, output_format=self.inference_output_format, input_latent=None)

    def interpolate_gan(self):

        if not os.path.exists(self.inference_output_directory):
            os.mkdir(self.inference_output_directory)

        self.load_model()

        pggan = PGGAN(input_model_path=self.inference_model_path,
                        model_output_size=self.gan_output_size,
                        inference_mode=True,
                        **self.kwargs)

        pggan.build_model()

        pggan.model_interpolation(self.inference_output_directory, interpolation_frames=self.interpolation_frames, interpolation_method=self.interpolation_method, input_latent=self.interpolation_latents, input_latent_length=self.interpolation_vector_num)


if __name__ == '__main__':

    pass