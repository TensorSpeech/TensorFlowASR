import tensorflow_asr.losses.rnnt_losses as rnnt_losses
import numpy as np
import tensorflow as tf


class WarpRNNTTest(tf.test.TestCase):
    def _run_rnnt(self, acts, labels, input_lengths, label_lengths,
                    expected_costs, expected_grads, use_gpu=False):
        self.assertEquals(acts.shape, expected_grads.shape)
        acts_t = tf.constant(acts)
        labels_t = tf.constant(labels)
        input_lengths_t = tf.constant(input_lengths)
        label_lengths_t = tf.constant(label_lengths)

        with tf.GradientTape() as tape:
            # by default, GradientTape doesn’t track constants
            tape.watch(acts_t)
            tape.watch(labels_t)
            tape.watch(input_lengths_t)
            tape.watch(label_lengths_t)
            logits = acts_t if use_gpu else tf.nn.log_softmax(acts_t)
            costs = rnnt_losses.rnnt_loss_tf(logits=logits,
                                             labels=labels_t, 
                                             label_length=label_lengths_t,
                                             logit_length=input_lengths_t, 
                                             name=None)

        grads = tape.gradient(costs, [acts_t])[0]
        self.assertAllClose(costs, expected_costs, atol=1e-6)
        self.assertAllClose(grads, expected_grads, atol=1e-6)


    def _test_multiple_batches(self, use_gpu):
        B = 2; T = 4; U = 3; V = 3

        acts = np.array([0.065357, 0.787530, 0.081592, 0.529716, 0.750675, 0.754135, 
                        0.609764, 0.868140, 0.622532, 0.668522, 0.858039, 0.164539, 
                        0.989780, 0.944298, 0.603168, 0.946783, 0.666203, 0.286882, 
                        0.094184, 0.366674, 0.736168, 0.166680, 0.714154, 0.399400, 
                        0.535982, 0.291821, 0.612642, 0.324241, 0.800764, 0.524106, 
                        0.779195, 0.183314, 0.113745, 0.240222, 0.339470, 0.134160, 
                        0.505562, 0.051597, 0.640290, 0.430733, 0.829473, 0.177467, 
                        0.320700, 0.042883, 0.302803, 0.675178, 0.569537, 0.558474, 
                        0.083132, 0.060165, 0.107958, 0.748615, 0.943918, 0.486356, 
                        0.418199, 0.652408, 0.024243, 0.134582, 0.366342, 0.295830, 
                        0.923670, 0.689929, 0.741898, 0.250005, 0.603430, 0.987289, 
                        0.592606, 0.884672, 0.543450, 0.660770, 0.377128, 0.358021], dtype=np.float32).reshape(B, T, U, V);

        expected_costs = np.array([4.28065, 3.93844], dtype=np.float32)
        expected_grads = np.array([-0.186844, -0.062555, 0.249399, -0.203377, 0.202399, 0.000977,
                                    -0.141016, 0.079123, 0.061893, -0.011552, -0.081280, 0.092832,
                                    -0.154257, 0.229433, -0.075176, -0.246593, 0.146405, 0.100188,
                                    -0.012918, -0.061593, 0.074512, -0.055986, 0.219831, -0.163845,
                                    -0.497627, 0.209240, 0.288387, 0.013605, -0.030220, 0.016615,
                                    0.113925, 0.062781, -0.176706, -0.667078, 0.367659, 0.299419,
                                    -0.356344, -0.055347, 0.411691, -0.096922, 0.029459, 0.067463,
                                    -0.063518, 0.027654, 0.035863, -0.154499, -0.073942, 0.228441,
                                    -0.166790, -0.000088, 0.166878, -0.172370, 0.105565, 0.066804,
                                    0.023875, -0.118256, 0.094381, -0.104707, -0.108934, 0.213642,
                                    -0.369844, 0.180118, 0.189726, 0.025714, -0.079462, 0.053748,
                                    0.122328, -0.238789, 0.116460, -0.598687, 0.302203, 0.296484], dtype=np.float32).reshape(B, T, U, V);

        labels = np.array([[1, 2], [1, 1]], dtype=np.int32)
        input_lengths = np.array([4, 4], dtype=np.int32)
        label_lengths = np.array([2, 2], dtype=np.int32)

        self._run_rnnt(acts,
                       labels,
                       input_lengths,
                       label_lengths,
                       expected_costs,
                       expected_grads)
    
    def test_multiple_batches_gpu(self):
        rnnt_losses.use_warprnnt = False
        self._test_multiple_batches(use_gpu=True)

    def test_multiple_batches_cpu(self):
        rnnt_losses.use_warprnnt = False
        self._test_multiple_batches(use_gpu=False)


if __name__ == "__main__":
    tf.test.main()
