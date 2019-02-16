# From https://github.com/zhangqianhui/progressive_growing_of_gans_tensorflow

import tensorflow as tf
import numpy as np
import math

from utils import add_parameter, save_images, save_image
from ops import lrelu, conv2d, fully_connect, upscale, downscale, pixel_norm, avgpool2d, minibatch_state_concat


class PGGAN(object):

    # build model
    def __init__(self, **kwargs):

        # Training Parameters
        add_parameter(self, kwargs, 'batch_size', 16)
        add_parameter(self, kwargs, 'iterations_per_stage', 10000)
        add_parameter(self, kwargs, 'learning_rate', 0.0001)
        add_parameter(self, kwargs, 'model_output_size', 64)
        add_parameter(self, kwargs, 'transition', False)

        # Data Parameters
        add_parameter(self, kwargs, 'training_data', None)
        add_parameter(self, kwargs, 'input_model_path', None)
        add_parameter(self, kwargs, 'output_model_path', None)

        # Logging Parameters
        add_parameter(self, kwargs, 'verbose', True)
        add_parameter(self, kwargs, 'model_save_interval_steps', 1000)
        add_parameter(self, kwargs, 'model_sample_interval_steps', 400)
        add_parameter(self, kwargs, 'model_loss_output_interval_steps', 40)
        add_parameter(self, kwargs, 'model_sample_dir', './samples')
        add_parameter(self, kwargs, 'model_logging_dir', './log')

        # Model Parameters
        add_parameter(self, kwargs, 'latent_size', 128)
        add_parameter(self, kwargs, 'max_filter', 128)
        add_parameter(self, kwargs, 'channels', 3)
        add_parameter(self, kwargs, 'discriminator_updates', 1)
        add_parameter(self, kwargs, 'generator_updates', 1)

        # Model Size Throttling Parameters
        add_parameter(self, kwargs, 'reduced_batch_size', 4)
        add_parameter(self, kwargs, 'batch_size_reduction_transition_depth', 20)
        add_parameter(self, kwargs, 'batch_size_reduction_transition_size', 128)
        add_parameter(self, kwargs, 'filter_num_reduction_transition_depth', None)
        add_parameter(self, kwargs, 'filter_num_reduction_transition_size', 128)

        # Misc Parameters
        add_parameter(self, kwargs, 'random_seed', None)
        add_parameter(self, kwargs, 'inference_mode', False)
        add_parameter(self, kwargs, 'pretrained_model', False)

        # Allow users to specify output sizes directly, but convert to 'depth' because it's easier later.
        self.model_progressive_depth = int(math.log(self.model_output_size, 2) - 1)
        if self.batch_size_reduction_transition_depth is None:
            self.batch_size_reduction_transition_depth = int(math.log(self.batch_size_reduction_transition_size, 2) - 1)
        if self.filter_num_reduction_transition_depth is None:
            self.filter_num_reduction_transition_depth = int(math.log(self.filter_num_reduction_transition_size, 2) - 1)

        # Shrink model at larger sizes to maintain performance / memory constraints.
        if self.batch_size_reduction_transition_depth is not None and not self.inference_mode:
            if self.model_progressive_depth >= self.batch_size_reduction_transition_depth:
                self.batch_size = self.reduced_batch_size

        # Derived Parameters
        self.log_vars = []
        self.model_output_size = pow(2, self.model_progressive_depth + 1)
        self.zoom_level = self.model_progressive_depth

        # Placeholders
        self.training_step = tf.placeholder(tf.float32, shape=None)
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.model_output_size, self.model_output_size, self.channels])
        self.latent = tf.placeholder(tf.float32, [self.batch_size, self.latent_size])
        self.alpha_transition = tf.Variable(initial_value=0.0, trainable=False, name='alpha_transition')

    def get_filter_num(self, depth):

        # There's something wrong here with indexing, sorry.
        if depth < self.filter_num_reduction_transition_depth - 1:
            return self.max_filter
        else:
            if self.pretrained_model and (2 ** (depth - self.filter_num_reduction_transition_depth + 2)) > 2:
                return 16
            else:
                return self.max_filter // (2 ** (depth - self.filter_num_reduction_transition_depth + 2))

    def generate(self, latent_var, model_progressive_depth=1, transition=False, alpha_transition=0.0, reuse=False):

        with tf.variable_scope('generator') as scope:

            if reuse:
                scope.reuse_variables()

            convs = []

            convs += [tf.reshape(latent_var, [self.batch_size, 1, 1, self.latent_size])]
            convs[-1] = pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), k_h=4, k_w=4, d_w=1, d_h=1, padding='Other', name='gen_n_1_conv')))

            convs += [tf.reshape(convs[-1], [self.batch_size, 4, 4, self.get_filter_num(1)])]
            convs[-1] = pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), d_w=1, d_h=1, name='gen_n_2_conv')))

            for i in range(model_progressive_depth - 1):

                if i == model_progressive_depth - 2 and transition:
                    # To RGB, low resolution
                    transition_conv = conv2d(convs[-1], output_dim=self.channels, k_w=1, k_h=1, d_w=1, d_h=1, name='gen_y_rgb_conv_{}'.format(convs[-1].shape[1]))
                    transition_conv = upscale(transition_conv, 2)

                convs += [upscale(convs[-1], 2)]
                convs[-1] = pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(i + 1), d_w=1, d_h=1, name='gen_n_conv_1_{}'.format(convs[-1].shape[1]))))

                convs += [pixel_norm(lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(i + 1), d_w=1, d_h=1, name='gen_n_conv_2_{}'.format(convs[-1].shape[1]))))]

            # To RGB, high resolution
            convs += [conv2d(convs[-1], output_dim=self.channels, k_w=1, k_h=1, d_w=1, d_h=1, name='gen_y_rgb_conv_{}'.format(convs[-1].shape[1]))]

            if transition:
                convs[-1] = (1 - alpha_transition) * transition_conv + alpha_transition * convs[-1]

            return convs[-1]

    def discriminate(self, input_image, reuse=False, model_progressive_depth=1, transition=False, alpha_transition=0.01, input_classes=None):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            if transition:
                # from RGB, low resolution
                transition_conv = avgpool2d(input_image)
                transition_conv = lrelu(conv2d(transition_conv, output_dim=self.get_filter_num(model_progressive_depth - 2), k_w=1, k_h=1, d_h=1, d_w=1, name='dis_y_rgb_conv_{}'.format(transition_conv.shape[1])))

            convs = []

            # from RGB, high resolution
            convs += [lrelu(conv2d(input_image, output_dim=self.get_filter_num(model_progressive_depth - 1), k_w=1, k_h=1, d_w=1, d_h=1, name='dis_y_rgb_conv_{}'.format(input_image.shape[1])))]

            for i in range(model_progressive_depth - 1):

                convs += [lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(model_progressive_depth - 1 - i), d_h=1, d_w=1, name='dis_n_conv_1_{}'.format(convs[-1].shape[1])))]

                convs += [lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(model_progressive_depth - 2 - i), d_h=1, d_w=1, name='dis_n_conv_2_{}'.format(convs[-1].shape[1])))]
                convs[-1] = avgpool2d(convs[-1], 2)

                if i == 0 and transition:
                    convs[-1] = alpha_transition * convs[-1] + (1 - alpha_transition) * transition_conv

            convs += [minibatch_state_concat(convs[-1])]
            convs[-1] = lrelu(conv2d(convs[-1], output_dim=self.get_filter_num(1), k_w=3, k_h=3, d_h=1, d_w=1, name='dis_n_conv_1_{}'.format(convs[-1].shape[1])))
            
            output = tf.reshape(convs[-1], [self.batch_size, -1])
            discriminate_output = fully_connect(output, output_size=1, scope='dis_n_fully')

            return tf.nn.sigmoid(discriminate_output), discriminate_output

    def build_model(self):

        # Output functions
        self.fake_images = self.generate(self.latent, model_progressive_depth=self.model_progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition)
        _, self.D_pro_logits = self.discriminate(self.images, reuse=False, model_progressive_depth=self.model_progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition)
        _, self.G_pro_logits = self.discriminate(self.fake_images, reuse=True, model_progressive_depth=self.model_progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition)

        # Loss functions
        self.D_loss = tf.reduce_mean(self.G_pro_logits) - tf.reduce_mean(self.D_pro_logits)
        self.G_loss = -tf.reduce_mean(self.G_pro_logits)

        # Wasserstein Loss...
        self.differences = self.fake_images - self.images
        self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = self.images + (self.alpha * self.differences)
        _, discri_logits = self.discriminate(interpolates, reuse=True, model_progressive_depth=self.model_progressive_depth, transition=self.transition, alpha_transition=self.alpha_transition)
        gradients = tf.gradients(discri_logits, [interpolates])[0]

        # ...with gradient penalty
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.D_origin_loss = self.D_loss
        self.D_loss += 10 * self.gradient_penalty
        self.D_loss += 0.001 * tf.reduce_mean(tf.square(self.D_pro_logits - 0.0))

        # Create resolution fade-in (transition) parameters.
        self.alpha_transition_assign = self.alpha_transition.assign(self.training_step / self.iterations_per_stage)

        """ A slightly magical bit of code inherited from the previous repository. Creates
            multiple variable loaders in Tensorflow. When going up a resolution, variables
            from the previous resolution are preserved, while new variables are left untouched.
            Tensorflow throws a fit if you try to load variables that aren't present in your
            provided model file, so this system circumvents that without loading the full
            model every run.
        """
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        # Save the variables, which remain unchanged
        self.d_vars_n = [var for var in self.d_vars if 'dis_n' in var.name]
        self.g_vars_n = [var for var in self.g_vars if 'gen_n' in var.name]

        # Remove the new variables for the new model
        self.d_vars_n_read = [var for var in self.d_vars_n if '{}'.format(self.model_output_size) not in var.name]
        self.g_vars_n_read = [var for var in self.g_vars_n if '{}'.format(self.model_output_size) not in var.name]

        # Save the rgb variables, which remain unchanged
        self.d_vars_n_2 = [var for var in self.d_vars if 'dis_y_rgb_conv' in var.name]
        self.g_vars_n_2 = [var for var in self.g_vars if 'gen_y_rgb_conv' in var.name]
        self.d_vars_n_2_rgb = [var for var in self.d_vars_n_2 if '{}'.format(self.model_output_size) not in var.name]
        self.g_vars_n_2_rgb = [var for var in self.g_vars_n_2 if '{}'.format(self.model_output_size) not in var.name]

        self.saver = tf.train.Saver(self.d_vars + self.g_vars)
        self.r_saver = tf.train.Saver(self.d_vars_n_read + self.g_vars_n_read)
        if len(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb):
            self.rgb_saver = tf.train.Saver(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb)

        # Create Optimizers
        self.opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        self.opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.G_loss, var_list=self.g_vars)

        # Data Loading Tools
        self.low_images = upscale(downscale(self.images, 2), 2)
        self.real_images = self.alpha_transition * self.images + (1 - self.alpha_transition) * self.low_images

        # Tensorboard Logging Variables
        tf.summary.scalar("gp_loss", self.gradient_penalty)
        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

        # if self.verbose:
        #     for layer in tf.trainable_variables():
        #         print(layer)
        
        return

    def train(self, verbose=True):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)

            # Some Tensorboard stuff could go here..
            # summary_op = tf.summary.merge_all()
            # summary_writer = tf.summary.FileWriter(self.model_logging_dir, sess.graph)

            if self.random_seed is not None:
                np.random.seed(self.random_seed)
                tf.set_random_seed(self.random_seed)

            if self.model_progressive_depth != 1 and self.model_progressive_depth != 7:

                if self.transition:
                    self.r_saver.restore(sess, self.input_model_path)
                    self.rgb_saver.restore(sess, self.input_model_path)
                else:
                    self.saver.restore(sess, self.input_model_path)

            step = 0
            batch_num = 0
            for step in range(self.iterations_per_stage):

                # Update Discriminator
                for i in range(self.discriminator_updates):

                    sample_latent = np.random.normal(size=[self.batch_size, self.latent_size])

                    real_data = self.training_data.get_next_batch(batch_num=batch_num, zoom_level=self.zoom_level, batch_size=self.batch_size)

                    # If in the 'resolution transition' stage, upsample and then downsample images.
                    if self.transition and self.model_progressive_depth != 0:
                        input_data = sess.run(self.real_images, feed_dict={self.images: real_data})
                    else:
                        input_data = real_data

                    sess.run(self.opti_D, feed_dict={self.images: real_data, self.latent: sample_latent})
                    batch_num += 1

                for i in range(self.generator_updates):

                    # Update Generator
                    sess.run(self.opti_G, feed_dict={self.latent: sample_latent})

                if self.transition and self.model_progressive_depth != 0:

                    # Change the interpolation ratio as training steps increase.
                    sess.run(self.alpha_transition_assign, feed_dict={self.training_step: step})

                if step % self.model_loss_output_interval_steps == 0:

                    D_loss, G_loss, D_origin_loss, interpolation_percentage = sess.run([self.D_loss, self.G_loss, self.D_origin_loss, self.alpha_transition], feed_dict={self.images: real_data, self.latent: sample_latent})

                    if self.verbose:

                        if self.transition:
                            print("PGGAN Depth %d, Interpolation Stage, Step %d: Dis_WP Loss=%.7f Gen Loss=%.7f, Interpolation=%.7f" % (self.model_progressive_depth, step, D_loss, G_loss, interpolation_percentage))
                        else:
                            print("PGGAN Depth %d, Step %d: Dis_WP Loss=%.7f Gen Loss=%.7f" % (self.model_progressive_depth, step, D_loss, G_loss))

                if step % self.model_save_interval_steps == 0:

                    save_images(real_data[0:self.batch_size], [2, self.batch_size // 2], '{}/{:02d}_real.png'.format(self.model_sample_dir, step))

                    if self.transition and self.model_progressive_depth != 0:

                        save_images(input_data[0:self.batch_size], [2, self.batch_size // 2], '{}/{:02d}_real_interpolate.png'.format(self.model_sample_dir, step))
                   
                    fake_image = sess.run(self.fake_images, feed_dict={self.latent: sample_latent})
                    fake_image = np.clip(fake_image, -1, 1)
                    save_images(fake_image[0:self.batch_size], [2, self.batch_size // 2], '{}/{:02d}_train.png'.format(self.model_sample_dir, step))

                if step % self.model_save_interval_steps == 0:
                    self.saver.save(sess, self.output_model_path)

            save_path = self.saver.save(sess, self.output_model_path)
            if self.verbose:
                print("Model saved in file: %s" % save_path)

        tf.reset_default_graph()

    def model_inference(self, output_directory, output_num, output_format='png', input_latent=None):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            self.saver.restore(sess, self.input_model_path)

            for batch_idx in range(0, output_num, self.batch_size):

                if self.verbose:
                    print('Working on images', batch_idx, 'to', batch_idx + self.batch_size)

                if input_latent is None:
                    sample_latent = np.random.normal(size=[self.batch_size, self.latent_size])
                else:
                    sample_latent = input_latent[batch_idx:batch_idx + self.batch_size]

                if sample_latent.shape[0] < self.batch_size:
                    sample_latent = np.concatenate([sample_latent, np.zeros(self.batch_size - sample_latent.shape[0], self.latent_size)], axis=0)

                inference_images = sess.run(self.fake_images, feed_dict={self.latent: sample_latent})
                inference_images = np.clip(inference_images, -1, 1)

                for channel_idx in range(self.batch_size):

                    if batch_idx + channel_idx > output_num:
                        break

                    save_image(inference_images[channel_idx, ...], '{}/inference_{:05d}.{}'.format(output_directory, batch_idx + channel_idx, output_format))

        tf.reset_default_graph()

    def model_interpolation(self, output_directory, interpolation_frames=100, interpolation_method='slerp', input_latent=None, input_latent_length=2, output_format='png'):
        
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        interpolation_dict = {'slerp': slerp,
                                'linear': linear_interpolation,
                                'tozero': linear_to_zero_interpolation}

        with tf.Session(config=config) as sess:

            sess.run(init)
            self.saver.restore(sess, self.input_model_path)

            if input_latent is None:
                input_latent = [np.random.normal(size=[self.latent_size]) for x in range(input_latent_length)]

            latents = []
            for idx in range(1, len(input_latent)):

                latents += interpolation_dict[interpolation_method](input_latent[idx - 1], input_latent[idx])

                pass

            latents = np.array(latents)
            output_num = latents.shape[0]

            for batch_idx in range(0, output_num, self.batch_size):

                if self.verbose:
                    print('Working on images', batch_idx, 'to', batch_idx + self.batch_size)

                sample_latent = latents[batch_idx:batch_idx + self.batch_size]

                if sample_latent.shape[0] < self.batch_size:
                    sample_latent = np.concatenate([sample_latent, np.zeros((self.batch_size - sample_latent.shape[0], self.latent_size))], axis=0)

                inference_images = sess.run(self.fake_images, feed_dict={self.latent: sample_latent})
                inference_images = np.clip(inference_images, -1, 1)

                for channel_idx in range(self.batch_size):

                    if batch_idx + channel_idx > output_num:
                        break

                    save_image(inference_images[channel_idx, ...], '{}/interpolation_{:05d}.{}'.format(output_directory, batch_idx + channel_idx, output_format))


def slerp(input_latent1, input_latent2, interpolation_frames=100):
    
    """Spherical linear interpolation ("slerp", amazingly enough).
    
    Parameters
    ----------
    input_latent1, input_latent2 : NumPy arrays
        Two arrays which will be interpolated between.
    interpolation_frames : int, optional
        Number of frame returned during interpolation.
    
    Returns
    -------
    list
        List of vectors of size interpolation_frames
    """

    output_latents = []

    for idx in range(interpolation_frames):
    
        val = float(idx) / interpolation_frames

        if np.allclose(input_latent1, input_latent2):
            output_latents += [input_latent2]
            continue

        omega = np.arccos(np.dot(input_latent1 / np.linalg.norm(input_latent1), input_latent2 / np.linalg.norm(input_latent2)))
        so = np.sin(omega)

        output_latents += [np.sin((1.0 - val) * omega) / so * input_latent1 + np.sin(val * omega) / so * input_latent2]

    return output_latents


def linear_interpolation(input_latent1, input_latent2, interpolation_frames=100):

    """Linear interpolation
    
    Parameters
    ----------
    input_latent1, input_latent2 : NumPy arrays
        Two arrays which will be interpolated between.
    interpolation_frames : int, optional
        Number of frame returned during interpolation.
    
    Returns
    -------
    list
        List of vectors of size interpolation_frames
    """

    output_latents = []

    for idx in range(interpolation_frames):
    
        val = float(idx) / interpolation_frames

        if np.allclose(input_latent1, input_latent2):
            output_latents += [input_latent2]
            continue

        output_latents += [input_latent1 + (input_latent2 - input_latent1) * val]

    return output_latents


def linear_to_zero_interpolation(input_latent1, input_latent2, interpolation_frames=100):

    """Linear interpolation, but via the origin. Values will first interpolate
        to the origin, and then to the destination value. Given the the center
        of a GAN's latent space often seems to be correlated with "emptiness"
        in my experience, this has the effect of an image disappearing and 
        reappearing as something else.
    
    Parameters
    ----------
    input_latent1, input_latent2 : NumPy arrays
        Two arrays which will be interpolated between.
    interpolation_frames : int, optional
        Number of frame returned during interpolation.
    
    Returns
    -------
    list
        List of vectors of size interpolation_frames
    """

    output_latents = []

    # Way too over-elaborate, I know. Indexing is hard.
    part1 = range(len(range(interpolation_frames)[0:interpolation_frames // 2]))
    part2 = range(len(range(interpolation_frames)[interpolation_frames // 2:]))
    origin = np.zeros_like(input_latent1)

    for idx in part1:
        val = float(idx) / len(part1)
        if np.allclose(input_latent1, origin):
            output_latents += [origin]
            continue
        output_latents += [input_latent1 + (origin - input_latent1) * val]

    for idx in part2:
        val = float(idx) / len(part2)
        if np.allclose(origin, input_latent2):
            output_latents += [input_latent2]
            continue
        output_latents += [origin + (input_latent2 - origin) * val]

    return output_latents


if __name__ == '__main__':

    pass