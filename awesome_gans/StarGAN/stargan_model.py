import tensorflow as tf

import awesome_gans.modules as t

tf.set_random_seed(777)  # reproducibility


def residual_block(x, f, name="0"):
    with tf.variable_scope("residual_block-" + name):
        scope_name = "residual_block-" + name

        x = t.conv2d(x, f=f, k=3, s=1)
        x = t.instance_norm(x, affine=True, name=scope_name + '_0')
        x = tf.nn.relu(x)

        x = t.conv2d(x, f=f, k=3, s=1)
        x = t.instance_norm(x, affine=True, name=scope_name + '_1')

        return x


class StarGAN:
    def __init__(
        self,
        s,
        batch_size=32,
        height=64,
        width=64,
        channel=3,
        attr_labels=(),
        sample_num=1 * 1,
        sample_size=1,
        df_dim=64,
        gf_dim=64,
        g_lr=1e-4,
        d_lr=1e-4,
    ):

        """
        # General Settings
        :param s: TF Session
        :param batch_size: training batch size, default 32
        :param height: input image height, default 4
        :param width: input image width, default 64
        :param channel: input image channel, default 3 (RGB)
        - in case of Celeb-A, image size is 64x64x3(HWC).
        :param attr_labels: attributes of Celeb-A image, default empty list
        - in case of Celeb-A, the number of attributes is 40

        # Output Settings
        :param sample_num: the number of output images, default 1
        :param sample_size: sample image size, default 64

        # Hyper Parameters
        :param df_dim: discriminator filter, default 64
        :param gf_dim: generator filter, default 64

        # Training Option
        :param g_lr: generator learning rate, default 1e-4
        :param d_lr: discriminator learning rate, default 1e-4
        """

        self.s = s
        self.batch_size = batch_size

        self.height = height
        self.width = width
        self.channel = channel

        self.attr_labels = attr_labels
        self.n_classes = len(self.attr_labels)  # Select 10 of them
        self.image_shape = [None, self.height, self.width, self.channel]

        self.sample_num = sample_num
        self.sample_size = sample_size

        # Model Hyper-Parameters
        self.df_dim = df_dim
        self.gf_dim = gf_dim

        self.d_lr = d_lr
        self.g_lr = g_lr

        self.lambda_cls = 1.0  #
        self.lambda_rec = 10.0  #
        self.lambda_gp = 0.25  # gradient penalty

        # Training Setting
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.d_real = 0
        self.d_fake = 0
        self.g_loss = 0.0
        self.d_loss = 0.0

        # Placeholders
        self.x_A = tf.placeholder(
            tf.float32, shape=[None, self.height, self.width, self.channel + self.n_classes], name='x-image-A'
        )  # input image
        self.x_B = tf.placeholder(
            tf.float32, shape=[None, self.height, self.width, self.channel + self.n_classes], name='x-image-B'
        )  # target image
        self.fake_x_B = tf.placeholder(tf.float32, shape=self.image_shape, name='x-image-fake-B')
        self.y_B = tf.placeholder(tf.float32, shape=[None, self.n_classes], name='y-label-B')

        self.lr_decay = tf.placeholder(tf.float32, shape=None, name='lr-decay')
        self.epsilon = tf.placeholder(tf.float32, shape=[None, 1, 1, 1], name='epsilon')

        # pre-defined
        self.fake_A = None
        self.fake_B = None

        self.d_op = None
        self.g_op = None

        self.merged = None
        self.writer = None
        self.saver = None

        self.build_stargan()  # build StarGAN model

    def discriminator(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("discriminator", reuse=reuse):

            def conv_lrelu(x, f, k, s):
                x = t.conv2d(x, f=f, k=k, s=s)
                x = tf.nn.leaky_relu(x)
                return x

            for i in range(6):
                x = conv_lrelu(x, f=self.df_dim * (2 ** (i + 1)), k=4, s=2)

            x = t.conv2d(x, f=1 + self.n_classes, k=1, s=1)

            x = tf.layers.flatten(x)  # (-1, 1 + n_classes_1)

            out_real = x[:, 0]  # (-1, 1)
            out_aux = x[:, 1:]  # (-1, n_classes_1)

            return out_real, out_aux

    def generator(self, x, reuse=None):
        """
        :param x: images
        :param reuse: re-usable
        :return: logits
        """
        with tf.variable_scope("generator", reuse=reuse):

            def conv_in_relu(x, f, k, s, de=False, name=""):
                if not de:
                    x = t.conv2d(x, f=f, k=k, s=s)
                else:
                    x = t.deconv2d(x, f=f, k=k, s=s)

                x = t.instance_norm(x, name=name)
                x = tf.nn.relu(x)
                return x

            x = conv_in_relu(x, f=self.gf_dim * 1, k=7, s=1, name="1")

            # down-sampling
            x = conv_in_relu(x, f=self.gf_dim * 2, k=4, s=2, name="2")
            x = conv_in_relu(x, f=self.gf_dim * 4, k=4, s=2, name="3")

            # bottleneck
            for i in range(6):
                x = residual_block(x, f=self.gf_dim * 4, name=str(i))

            # up-sampling
            x = conv_in_relu(x, self.gf_dim * 2, k=4, s=2, de=True, name="4")
            x = conv_in_relu(x, self.gf_dim * 1, k=4, s=2, de=True, name="5")

            x = t.deconv2d(x, f=3, k=7, s=1)
            x = tf.nn.tanh(x)

            return x

    def build_stargan(self):
        def gp_loss(real, fake, eps=self.epsilon):
            # alpha = tf.random_uniform(shape=real.get_shape(), minval=0., maxval=1., name='alpha')
            # diff = fake - real  # fake data - real data
            # interpolates = real + alpha * diff
            interpolates = eps * real + (1.0 - eps) * fake
            d_interp = self.discriminator(interpolates, reuse=True)
            gradients = tf.gradients(d_interp, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
            return gradient_penalty

        x_img_a = self.x_A[:, :, :, : self.channel]
        x_attr_a = self.x_A[:, :, :, self.channel :]
        x_img_b = self.x_B[:, :, :, : self.channel]
        # x_attr_b = self.x_B[:, :, :, self.channel:]

        # Generator
        self.fake_B = self.generator(self.x_A)
        gen_in = tf.concat([self.fake_B, x_attr_a], axis=3)
        self.fake_A = self.generator(gen_in, reuse=True)

        # Discriminator
        d_src_real_b, d_aux_real_b = self.discriminator(x_img_b)
        g_src_fake_b, g_aux_fake_b = self.discriminator(self.fake_B, reuse=True)  # used at updating G net
        d_src_fake_b, d_aux_fake_b = self.discriminator(self.fake_x_B, reuse=True)  # used at updating D net

        # using WGAN-GP losses
        gp = gp_loss(x_img_b, self.fake_x_B)
        d_src_loss = tf.reduce_mean(d_src_fake_b) - tf.reduce_mean(d_src_real_b) + gp
        d_aux_loss = t.sce_loss(d_aux_real_b, self.y_B)

        self.d_loss = d_src_loss + self.lambda_cls * d_aux_loss
        g_src_loss = -tf.reduce_mean(g_src_fake_b)
        g_aux_fake_loss = t.sce_loss(g_aux_fake_b, self.y_B)
        g_rec_loss = t.l1_loss(x_img_a, self.fake_A)
        self.g_loss = g_src_loss + self.lambda_cls * g_aux_fake_loss + self.lambda_rec * g_rec_loss

        # Summary
        tf.summary.scalar("loss/d_src_loss", d_src_loss)
        tf.summary.scalar("loss/d_aux_loss", d_aux_loss)
        tf.summary.scalar("loss/d_loss", self.d_loss)
        tf.summary.scalar("loss/g_src_loss", g_src_loss)
        tf.summary.scalar("loss/g_aux_fake_loss", g_aux_fake_loss)
        tf.summary.scalar("loss/g_rec_loss", g_rec_loss)
        tf.summary.scalar("loss/g_loss", self.g_loss)

        # Optimizer
        t_vars = tf.trainable_variables()
        d_params = [v for v in t_vars if v.name.startswith('d')]
        g_params = [v for v in t_vars if v.name.startswith('g')]

        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr * self.lr_decay, beta1=self.beta1).minimize(
            self.d_loss, var_list=d_params
        )
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr * self.lr_decay, beta1=self.beta1).minimize(
            self.g_loss, var_list=g_params
        )

        # Merge summary
        self.merged = tf.summary.merge_all()

        # Model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
