import tensorflow as tf
import numpy as np

num_epochs = 10000  # The number of times we iterate through the training data
learning_rate = 0.001  # The size of the steps we take in the gradient descent
batch_size = 200  # How many characters we feed through before back-propagating all of them
sequence_len = 10 #The length of the sequence of characters used to predict the next one
num_cells = 80 #The number of hidden units per LSTM cell
num_lstms = 1 #Currently not in use.  TODO: implement
forget = 1.0 #The forget bias of LSTM forget gates.

def read_files():
    njala = open(file='Islendingasogur/BrennuNjalsSaga.txt',encoding='UTF-8')
    # Todo: read more sagas in and concatenate them
    return njala.read()


def embed(sagas):
    '''
    The data is embedded into numerical data (vectors to represent characters) that we can use to train our RNN.
    :param sagas: Training data
    :return:
    '''
    unique_characters = list(set(sagas))  # All unique characters in our sagas.  We use this to generate the invidividual input vectors representing single characters.

    input_data= np.zeros((len(sagas), sequence_len, len(unique_characters)))
    output_data = np.zeros((len(sagas), len(unique_characters)))

    ootpoot = ''
    for i, char in enumerate(sagas):
        sequence_slice = sagas[i:i+sequence_len+1]
        tic = ''
        for j, char_input in enumerate(sequence_slice):
            if(j < sequence_len):
                input_data[i,j,unique_characters.index(char_input)] = 1
            else:
                output_data[i, unique_characters.index(char_input)] = 1
                tic += char_input
        #print(tic)

    return input_data, output_data, unique_characters


def make_rnn(in_placeholder):
    '''
    Instantiates a recurrant neural network.
    :return:
    '''
    weights = tf.Variable(tf.random_normal([num_cells, len(unique_characters)]), name='weights')
    biases = tf.Variable(tf.random_normal([len(unique_characters)]), name='biases')


    #Transpose the data, reshape and split to get sequence len number of tensors representing
    #print("at start: ", in_placeholder)
    in_placeholder = tf.transpose(in_placeholder, [1, 0, 2])
    #print("after transpose: ", in_placeholder)
    in_placeholder = tf.reshape(in_placeholder, [-1, len(unique_characters)])
    #print("after reshape: ", in_placeholder)
    in_placeholder = tf.split(in_placeholder, sequence_len, 0)
    #print("after split: ", in_placeholder)

    lstm = tf.contrib.rnn.BasicLSTMCell(num_units = num_cells, forget_bias = forget)
    output = tf.nn.static_rnn(lstm, in_placeholder, dtype=tf.float32)[0]

    pred = tf.matmul(output[-1], weights) + biases

    return pred


def train(optimizer, input_p, output_p, accuracy):
    '''
    Trains the model, 1 pass over the input data.
    :param optimizer: an optimizer
    :param input_p: input node
    :param output_p: output node
    '''
    # For each batch of data
    i = 0
    for j in range(len(input_data) // batch_size):
        in_batch = input_data[i * batch_size: i * batch_size + batch_size]  # The i-th batch of input data.
        out_batch = output_data[i * batch_size:i * batch_size + batch_size]  # The i-th batch of output data

        outtext = ''
        intext = ''
        for k in range(batch_size):
            intext += unique_characters[np.argmax(in_batch[k])]
            outtext += unique_characters[np.argmax(out_batch[k])]
        #print("in: ", intext)
        #print("out: ", outtext)

        o,acc, loss = sess.run([optimizer, accuracy, cost], feed_dict={input_p: in_batch, output_p: out_batch})

    return acc, loss


def generate(seed):
    '''
    :param seed:
    :return:
    '''
    tac = ''
    one_run = sess.run([pred], feed_dict={input_p: seed})

    #Print the contents of the seed
    seed_str = ''
    for i in range(10):
        seed_str += unique_characters[np.argmax(seed[0][i])]
    print(seed_str)

    for i in range(200):
        predicted = np.asarray(one_run[0]).astype('float64')[0]
        index, probabilities = activate_clean(predicted)
        seed = seed[:,1:,:] # Remove the first character from the seed

        seed = np.append(seed, np.reshape(probabilities, [1, 1, len(unique_characters)]), 1)  # Append the just-generated character to the seed
        one_run = sess.run([pred], feed_dict={input_p: seed})

        predicted_chars = unique_characters[index]
        tac += predicted_chars

    #writer2.add_summary(summary, i)
    #print(i, acc)
    print("Epoch nr ", j, ": ", seed_str , tac)

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
    #output = sigmoid(output / 0.5)
    output = np.exp(output / 0.5) #Just plugging the value in for now
    #Normalize
    output = output / sum(output)
    index_of_probability = np.random.choice(range(len(unique_characters)), p = output)
    #index_of_probability = np.argmax(output)
    clean_output = np.zeros(len(output))
    clean_output[index_of_probability] = 1

    return index_of_probability, clean_output


if __name__ == '__main__':
    input_data, output_data, unique_characters = embed(read_files())

    # These become our input and output nodes.
    input_p = tf.placeholder(tf.float32, [None, sequence_len, len(unique_characters)], name='input_p')
    output_p = tf.placeholder(tf.float32, [None, len(unique_characters)], name='output_p')

    pred = make_rnn(input_p)

    correct_pred =  tf.equal(tf.argmax(pred,1), tf.argmax(output_p, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype = tf.float32))

    # Parameters for back propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = output_p))  # Our cost function
    # We will use gradient descent to find a local minimum of the cost function.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Our tensorflow session!
    with tf.Session() as sess:

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
            acc, loss = train(optimizer, input_p, output_p, accuracy)
            # Test it by making it generate some characters

            generate(input_data[4:5:])  # For now the seed will just be one of our training batches.
            print("Accuracy: ", acc, ", Loss: ", loss)
            #TODO: allow user-input seeds