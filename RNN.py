import tensorflow as tf
import numpy as np

num_epochs = 100  # The number of times we iterate through the training data
learning_rate = 0.001  # The size of the steps we take in the gradient descent
batch_size = 200  # How many characters we feed through before back-propagating all of them
sequence_len = 20 #The length of the sequence of characters used to predict the next one
num_cells = 80 #The number of hidden units per LSTM cell
num_lstms = 1 #Currently not in use.  TODO: implement
forget = 1.0 #The forget bias of LSTM forget gates.
model_name = "default_RNN" #The name of the network as it gets saved

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

#Helper functions for conversion to and from one-hot vectors
def char_to_onehot(char):
    onehot = np.zeros(len(unique_characters))
    onehot[unique_characters.index(char)] = 1
    return onehot

def make_rnn(in_placeholder):
    '''
    Instantiates a recurrant neural network.
    :return:
    '''
    weights = tf.Variable(tf.random_normal([num_cells, len(unique_characters)]), name='weights')
    biases = tf.Variable(tf.random_normal([len(unique_characters)]), name='biases')


    #Transpose the data, reshape and split to get sequence len number of tensors representing
    in_placeholder = tf.transpose(in_placeholder, [1, 0, 2])
    in_placeholder = tf.reshape(in_placeholder, [-1, len(unique_characters)])
    in_placeholder = tf.split(in_placeholder, sequence_len, 0)

    lstm = tf.contrib.rnn.BasicLSTMCell(num_units = num_cells, forget_bias = forget)
    output = tf.nn.static_rnn(lstm, in_placeholder, dtype=tf.float32)[0]

    pred = tf.matmul(output[-1], weights)
    pred = tf.add(pred, biases)

    return pred


def train(optimizer, input_p, output_p, accuracy):
    '''
    Trains the model, 1 pass over the input data.
    :param optimizer: an optimizer
    :param input_p: input node
    :param output_p: output node
    '''
    # For each batch of data
    for j in range(len(input_data) // batch_size):
        in_batch = input_data[j * batch_size: j * batch_size + batch_size]  # The i-th batch of input data.
        out_batch = output_data[j * batch_size:j * batch_size + batch_size]  # The i-th batch of output data

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
    generated_string = ''
    one_run = sess.run([pred], feed_dict={input_p: seed})

    #Print the contents of the seed
    seed_str = ''
    for i in range(sequence_len):
        seed_str += unique_characters[np.argmax(seed[0][i])]

    for i in range(500):
        predicted = np.asarray(one_run[0]).astype('float64')[0]
        index, probabilities = activate_clean(predicted)
        seed = seed[:,1:,:] # Remove the first character from the seed

        seed = np.append(seed, np.reshape(probabilities, [1, 1, len(unique_characters)]), 1)  # Append the just-generated character to the seed
        one_run = sess.run([pred], feed_dict={input_p: seed})

        predicted_chars = unique_characters[index]
        generated_string += predicted_chars

    #writer2.add_summary(summary, i)
    print("Epoch nr ", j, ": ", seed_str , generated_string)

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
    output = np.exp(output / 0.5) #Just plugging the value of the temperature in for now
    #Normalize
    output = output / sum(output)
    #Choose a character
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

    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    #setting variables for other programs to reference when generating strings on a saved model
    tf.get_variable('unique_characters', dtype=tf.string, initializer=tf.convert_to_tensor(unique_characters))
    tf.get_variable('sequence_length', dtype=tf.int32, initializer=sequence_len)
    tf.add_to_collection('pred', pred)
    tf.add_to_collection('input_p', input_p)

    # Our tensorflow session!
    with tf.Session() as sess:

        # Initialise
        init = tf.global_variables_initializer()
        sess.run(init)

        # Tensorboard setup
        writer1 = tf.summary.FileWriter('./logs')
        writer1.add_graph(sess.graph)

        #instantiate a saver for the model
        saver = tf.train.Saver()

        #for i in range(len(seed)):
            #user_onehots[0][i][unique_characters.index(seed[i])] = 1

        for j in range(num_epochs):

            # Train our model
            acc, loss = train(optimizer, input_p, output_p, accuracy)
            # Test it by making it generate some characters
            generate(input_data[110:111:])
            #save our model so we can use it later
            saver.save(sess, ('./model/' + model_name))
            print("Accuracy: ", acc, ", Loss: ", loss)
