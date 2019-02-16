import sys
import os
import yaml


def process_config(config_dict):

    with open(config_dict, 'r') as stream:
        try:
            data_dict = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if data_dict['gpu_num'] is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(data_dict['gpu_num'])

    from deepzine import DeepZine

    gan = DeepZine(**data_dict)

    gan.execute()

    return


if __name__ == '__main__':

    process_config(sys.argv[1])

    # import tensorflow as tf
    # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

    # latest_ckp = tf.train.latest_checkpoint('./pretrained_models/1024')
    # print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')

    # from tensorflow.python import pywrap_tensorflow
    # import os

    # checkpoint_path = os.path.join('./pretrained_models/1024', "model.ckpt")
    # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    # var_to_shape_map = reader.get_variable_to_shape_map()

    # for key in sorted(var_to_shape_map):
    #     print("tensor_name: ", key)
    #     print(reader.get_tensor(key).shape) # Remove this is you want to print only variable names

    pass