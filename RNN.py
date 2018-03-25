import tensorflow as tf
import numpy as np

num_epochs = 1000  # The number of times we iterate through the training data
learning_rate = 0.01  # The size of the steps we take in the gradient descent
batch_size = 200  # How many characters we feed through before back-propagating all of them
num_cells = 80

def read_files():
    njala = open(file='Islendingasogur/HranaSagaHrings.txt',encoding='UTF-8')
    # Todo: read more sagas in and concatenate them
    return njala.read()


def embed(sagas):
    '''
    The data is embedded into numerical data (vectors to represent characters) that we can use to train our RNN.
    :param sagas: Training data
    :return:
    '''
    unique_characters = list(set(sagas))  # All unique characters in our sagas.  We use this to generate the invidividual input vectors representing single characters.

    input_data= np.zeros((len(sagas), len(unique_characters)))
    output_data = np.zeros((len(sagas), len(unique_characters)))

    for i, char in enumerate(sagas):
        input_data[i, unique_characters.index(char)] = 1
        if i != 0:
            output_data[i-1, unique_characters. index(char)] = 1

    return input_data, output_data, unique_characters


def make_rnn():
    '''
    Instantiates a recurrant neural network.
    :return:
    '''
    weights = tf.Variable(tf.random_normal([num_cells, len(unique_characters)]), name='weights')
    biases = tf.Variable(tf.random_normal([len(unique_characters)]), name='biases')

    lstm = tf.contrib.rnn.BasicLSTMCell(num_cells)
    output = tf.nn.static_rnn(cell=lstm, inputs=[tf.convert_to_tensor(input_p)], dtype=tf.float32)[0]

    pred = tf.matmul(output[-1], weights) + biases

    return pred


def train(optimizer, input_p, output_p):
    '''
    Trains the model, 1 pass over the input data.
    :param optimizer: an optimizer
    :param input_p: input node
    :param output_p: output node
    '''
    # For each batch of data
    for i in range(len(input_data) // batch_size):
        in_batch = input_data[i * batch_size: i * batch_size + batch_size]  # The i-th batch of input data.
        out_batch = output_data[i * batch_size:i * batch_size + batch_size]  # The i-th batch of output data
        o, loss, acc = sess.run([optimizer, cost, acc_op_t], feed_dict={input_p: in_batch, output_p: out_batch})

        if i%50 == 0:
            print("loss: ", loss, "Accuracy", acc)



def generate(seed):
    '''
    :param seed:
    :return:
    '''
    tac = ''
    one_run = sess.run([pred], feed_dict={input_p: seed})

    for i in range(200):
        seed = seed[1:]  # Remove the first character from the seed
        seed = np.concatenate((seed, [one_run[0][0]]))  # Append the just-generated character to the seed

        one_run = sess.run([pred], feed_dict={input_p: seed})

        predicted = np.asarray(one_run[0]).astype('float64')[0]
        index, probabilities = activate_clean(predicted)
        predicted_chars = unique_characters[index]
        tac += predicted_chars  # Just take the character with the highest probability for now.

    #writer2.add_summary(summary, i)
    #print(i, acc)
    print("Epoch nr ", j, ": ", tac)


def sigmoid(x):
    '''
    The sigmoid function.
    :param x:
    :return: sigmoid(x)
    '''
    return np.exp(x) / (1+np.exp(x))


def activate_clean(output):
    '''
    Output is a probability distribution.
    We want to select a character based on this distribution, but always selecting the character with the highest
    probability is not correct, as the highest probability is not necessarily 100%, so we use
    np.random.multinomial to select a character randomly, based on the probabilities.
    :param output:
    :return:
    '''
    # First we must normalize the probability distribution.
    # We start by getting all positive numbers by running the values through an activation function.

    #Activate
    output = sigmoid(output)
    #Normalize
    output = output / np.sum(output)

    clean_output = np.random.multinomial(1, output, 1)
    index_of_probability = np.argmax(clean_output)

    return index_of_probability, output


if __name__ == '__main__':
    input_data, output_data, unique_characters = embed(read_files())

    # These become our input and output nodes.
    input_p = tf.placeholder(tf.float32, [None, len(unique_characters)], name='input_p')
    output_p = tf.placeholder(tf.float32, [None, len(unique_characters)], name='output_p')

    pred = make_rnn()

    # Parameters for back propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = output_p))  # Our cost function
    # We will use gradient descent to find a local minimum of the cost function.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    #Accuracy shit
    correct_pred_t = tf.equal(tf.argmax(output_p, 1), tf.argmax(input_p, 1))  # Count correct predictions
    acc_op_t = tf.reduce_mean(tf.cast(correct_pred_t, "float"))  # Cast boolean to float to average

    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(output_p, 1), tf.argmax(input_p, 1)) # Count correct predictions
        acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
        # Add scalar summary for accuracy tensor
        tf.summary.scalar("accuracy", acc_op)

    # Our tensorflow session!
    sess = tf.Session()

    # Initialise
    init = tf.global_variables_initializer()
    sess.run(init)

    # Tensorboard setup
    writer1 = tf.summary.FileWriter('./logs')
    writer1.add_graph(sess.graph)

    #writer2 = tf.summary.FileWriter("./logs/nn_logs", sess.graph)  # for 0.8
    #merged = tf.summary.merge_all()

    for j in range(num_epochs):

        # Train our model
        train(optimizer, input_p, output_p)
        # Test it by making it generate some characters
        generate(input_data[3 * batch_size: 3 * batch_size + batch_size])  # For now the seed will just be the third batch.
        #TODO: allow user-input seeds