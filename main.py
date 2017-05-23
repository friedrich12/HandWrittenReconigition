# import MNIST data
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Set junk
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

# create a model

# set model weights
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
    # construct a linear model
    model = tf.nn.softmax(tf.matmul(x,W) + b)

# Add summary ops to collect data
w_h = tf.summary.histogram(name="weights", values=W)
b_h = tf.summary.histogram(name="biases", values=b)

with tf.name_scope("cost_function") as scope:
    # Minimize error using croos entropy
    # croos entropy
    cost_function = -tf.reduce_sum(y*tf.log(model))
    tf.summary.scalar(name="cost_function", tensor=cost_function)

with tf.name_scope("train") as scope:
    # Gradient Decest
    optimizer = \
        tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate).\
            minimize(cost_function)

# Initializing the variables
init = tf.initialize_all_variables()

# Merg all summaries into a single operator
merged_summary_op = tf.global_variables_initializer()

# launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Set the logs writer to the folder /tmp/tensorflow_logs
    summary_writer = tf.summary.FileWriter('logs',
                                           graph_def=sess.graph_def)

    # training cycle
    for iteration in range(training_iteration):
        avg_cost = 0. # checks if the model is improving during training
        total_batch = int(mnist.train.num_examples/batch_size)

        # Loop over all the batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch

            # write logs for each iteration of loop
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)
        # Display logs per each iteration setp
        if iteration % display_step == 0:
            print("Iteration:", '%04d' % (iteration + 1), "cost=","{:.9f}".format(avg_cost))

# Let's Test the model
predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
print("Accuracy:", accuracy.eval(session=sess, {x: mnist.test.images, y: mnist.test.labels}))