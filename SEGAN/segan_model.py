import tensorflow as tf
import ops


tf.set_random_seed(777)  # reproducibility


class SEGAN:

    def __init__(self, s, batch_size=64, input_height=28, input_width=28, input_channel=1, n_classes=10,
                 sample_num=8 * 8, sample_size=8, output_height=28, output_width=28,
                 n_input=784, fc_unit=1024, df_dim=64, gf_dim=64,
                 z_dim=128, g_lr=2e-4, d_lr=2e-4, epsilon=1e-12):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 64
        :param input_height: input image height, default 28
        :param input_width: input image width, default 28
        :param input_channel: input image channel, default 1 (gray-scale)

        # Output Settings
        :param sample_num: the number of output images, default 64
        :param sample_size: sample image size, default 8
        :param output_height: output images height, default 28
        :param output_width: output images width, default 28

        # For DNN model
        :param n_input: input image size, default 784(28x28)
        :param fc_unit: fully connected units, default 1024
        :param df_dim: the number of disc filters, default 64
        :param gf_dim: the number of gen filters, default 64

        # Training Option
        :param z_dim: z dimension (kinda noise), default 128
        :param g_lr: generator learning rate, default 2e-4
        :param d_lr: discriminator learning rate, default 2e-4
        :param epsilon: epsilon, default 1e-12
        """

        self.s = s
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.image_shape = [self.batch_size, self.input_height, self.input_width, self.input_channel]
        self.n_classes = n_classes

        self.sample_num = sample_num
        self.sample_size = sample_size
        self.output_height = output_height
        self.output_width = output_width

        self.n_input = n_input
        self.fc_unit = fc_unit
        self.df_dim = df_dim
        self.gf_dim = gf_dim

        self.z_dim = z_dim
        self.beta1 = .5
        self.beta2 = .999
        self.d_lr, self.g_lr = d_lr, g_lr
        self.eps = epsilon

        # pre-defined
        self.d_loss = 0.
        self.d_1_loss = 0.
        self.d_2_loss = 0.
        self.g_loss = 0.
        self.g_1_loss = 0.
        self.g_2_loss = 0.

        self.g = None
        self.g_sample = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        # Placeholder
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size,
                                                   self.input_height, self.input_width, self.input_channel],
                                name="x-sound")
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim],
                                name='z-noise')

        # ops
        self.ops = ops.VBN()

        self.num_blocks = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024

        self.build_segan()  # build SEGAN model

    def discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            def residual_block(x, name='residual_block'):
                x = ops.conv2d(x)
                x = self.ops(x)
                x = tf.nn.leaky_relu(x)
                return x

            if len(x) == 2:
                x = tf.expand_dims(x, axis=-1)
            else:
                raise ValueError("[-] disc: waveform must be 2, 3-D")

            for idx, f in enumerate(self.num_blocks):
                x = residual_block(x)


            return x

    def generator(self, z, reuse=False, training=True):
        with tf.variable_scope("generator", reuse=reuse):

            return z

    def build_segan(self):
        def sce_loss(x, y):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y))

        # Generator
        self.g = self.generator(self.z)
        self.g_sample = self.generator(self.z, reuse=True, training=False)

        # Discriminator
        d_real = self.discriminator(self.x, reuse=False)
        d_fake = self.discriminator(self.g, reuse=True)

        # Losses
        d_real_loss = sce_loss(d_real, tf.ones_like(d_real))
        d_fake_loss = sce_loss(d_fake, tf.zeros_like(d_fake))
        self.d_1_loss = d_real_loss + d_fake_loss
        self.d_2_loss = d_real_loss + d_fake_loss
        self.d_loss = self.d_1_loss + self.d_2_loss

        self.g_loss = sce_loss(d_fake, tf.ones_like(d_fake))

        # Summary
        tf.summary.scalar("loss/d_real_loss", d_real_loss)
        tf.summary.scalar("loss/d_fake_loss", d_fake_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_params)
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr,
                                           beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_params)

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
