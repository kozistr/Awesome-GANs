import tensorflow as tf


tf.set_random_seed(777)


class GAN:

    def __init__(self, s, batch_size=64,
                 input_height=28, input_width=28, channel=1, z_dim=128, sample_num=64, sample_size=8,
                 output_height=28, output_width=28, n_input=784, n_classes=10,
                 n_hidden_layer_1=64, n_hidden_layer_2=128, g_lr=1e-3, d_lr=1e-3, epsilon=1e-8):
        self.s = s
        self.batch_size = batch_size

        self.sample_num = sample_num
        self.sample_size = sample_size

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.channel = channel
        self.z_dim = z_dim

        self.n_input = n_input
        self.n_hl_1 = n_hidden_layer_1
        self.n_hl_2 = n_hidden_layer_2
        self.n_classes = n_classes

        self.eps = epsilon

        self.d_lr, self.g_lr = d_lr, g_lr

        self.build_gan()

    def discriminator(self, x, reuse=None):  # simple neural networks
        with tf.variable_scope("discriminator", reuse=reuse):
            net = tf.nn.bias_add(tf.matmul(x, self.W['d_h1']), self.b['d_b1'])
            net = tf.nn.relu(net)
            net = tf.nn.dropout(net, 0.8)

            net = tf.nn.bias_add(tf.matmul(net, self.W['d_h2']), self.b['d_b2'])
            net = tf.nn.relu(net)
            net = tf.nn.dropout(net, 0.8 - 0.05)

            logits = tf.nn.bias_add(tf.matmul(net, self.W['d_h_out']), self.b['d_b_out'])
            prob = tf.nn.sigmoid(logits)

        return prob

    def generator(self, z, reuse=None):
        with tf.variable_scope("generator", reuse=reuse):
            de_net = tf.nn.bias_add(tf.matmul(z, self.W['g_h1']), self.b['g_b1'])
            de_net = tf.nn.relu(de_net)

            de_net = tf.nn.bias_add(tf.matmul(de_net, self.W['g_h2']), self.b['g_b2'])
            de_net = tf.nn.relu(de_net)

            logits = tf.nn.bias_add(tf.matmul(de_net, self.W['g_h_out']), self.b['g_b_out'])
            prob = tf.nn.tanh(logits)

        return prob

    def build_gan(self):
        # x, z placeholder
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_input], name="x-image")
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z-noise')

        # weights
        self.W = {
            # weights for discriminator
            'd_h1': tf.get_variable('d_h1', shape=[self.n_input, self.n_hl_1],
                                    initializer=tf.contrib.layers.variance_scaling_initializer()),
            'd_h2': tf.get_variable('d_h2', shape=[self.n_hl_1, self.n_hl_2],
                                    initializer=tf.contrib.layers.variance_scaling_initializer()),
            'd_h_out': tf.get_variable('d_h_out', shape=[self.n_hl_2, 1],
                                       initializer=tf.contrib.layers.variance_scaling_initializer()),
            # weights for generator
            'g_h1': tf.get_variable('g_h1', shape=[self.z_dim, self.n_hl_2],
                                    initializer=tf.contrib.layers.variance_scaling_initializer()),
            'g_h2': tf.get_variable('g_h2', shape=[self.n_hl_2, self.n_hl_1],
                                    initializer=tf.contrib.layers.variance_scaling_initializer()),
            'g_h_out': tf.get_variable('g_h_out', shape=[self.n_hl_1, self.n_input],
                                       initializer=tf.contrib.layers.variance_scaling_initializer()),
        }

        # bias
        self.b = {
            # biases for discriminator
            'd_b1': tf.Variable(tf.zeros([self.n_hl_1])),
            'd_b2': tf.Variable(tf.zeros([self.n_hl_2])),
            'd_b_out': tf.Variable(tf.zeros([1])),
            # biases for generator
            'g_b1': tf.Variable(tf.zeros([self.n_hl_2])),
            'g_b2': tf.Variable(tf.zeros([self.n_hl_1])),
            'g_b_out': tf.Variable(tf.zeros([self.n_input])),
        }

        # generator
        self.G = self.generator(self.z)

        # discriminator
        self.D_real = self.discriminator(self.x)
        self.D_fake = self.discriminator(self.G, reuse=True)

        # maximize log(D(G(z)))
        self.g_loss = -tf.reduce_mean(tf.log(self.D_fake + self.eps))

        # maximize log(D(x)) + log(1 - D(G(z)))
        self.d_real_loss = -tf.reduce_mean(tf.log(self.D_real + self.eps))
        self.d_fake_loss = -tf.reduce_mean(tf.log((1. - self.D_fake) + self.eps))
        self.d_loss = self.d_real_loss + self.d_fake_loss

        # summary
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = tf.reshape(self.G, shape=[-1, self.output_height, self.output_height, self.channel])
        self.G_sum = tf.summary.image("G", self.G)  # generated image from G model
        self.D_real_sum = tf.summary.histogram("D_real", self.D_real)
        self.D_fake_sum = tf.summary.histogram("D_fake", self.D_fake)

        self.d_real_loss_sum = tf.summary.scalar("d_real_loss", self.d_real_loss)
        self.d_fake_loss_sum = tf.summary.scalar("d_fake_loss", self.d_fake_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # model saver
        self.saver = tf.train.Saver()

        # optimizer
        self.d_op = tf.train.AdamOptimizer(learning_rate=self.d_lr).\
            minimize(self.d_loss, var_list=[self.W['d_h1'], self.W['d_h2'], self.W['d_h_out'],
                                            self.b['d_b1'], self.b['d_b2'], self.b['d_b_out']])
        self.g_op = tf.train.AdamOptimizer(learning_rate=self.g_lr).\
            minimize(self.g_loss, var_list=[self.W['g_h1'], self.W['g_h2'], self.W['g_h_out'],
                                            self.b['g_b1'], self.b['g_b2'], self.b['g_b_out']])

        # merge summary
        self.g_sum = tf.summary.merge([self.z_sum, self.D_fake_sum, self.G_sum, self.d_fake_loss_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.D_real_sum, self.d_real_loss_sum, self.d_loss_sum])
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./model/', self.s.graph)
