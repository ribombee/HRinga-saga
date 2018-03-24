import tensorflow as tf
import numpy as np

num_epochs = 200 #The number of times we iterate through the training data
learning_rate = 0.01 #The size of the steps we take in the gradient descent
batch_size = 50 #How many characters we feed through before back-propagating all of them
num_cells = 80

def read_files():
    njala = open(file='Islendingasogur/BrennuNjalsSaga.txt',encoding='UTF-8')
    #Todo: read more sagas in and concatenate them
    return njala.read()

#The data is embedded into numerical data (vectors to represent characters) that we can use to train our RNN.
def embed(sagas):

    unique_characters = list(set(sagas)) #All unique characters in our sagas.  We use this to generate the invidividual input vectors representing single characters.

    input_data= np.zeros((len(sagas), len(unique_characters)))
    output_data = np.zeros((len(sagas), len(unique_characters)))

    for i,char in enumerate(sagas):
        input_data[i, unique_characters.index(char)] = 1
        if(i != 0):
            output_data[i-1, unique_characters. index(char)] = 1
    output_data[len(sagas)-1, unique_characters.index('\n')] = 1

    return input_data, output_data, unique_characters

def make_rnn():
    weights = tf.Variable(tf.random_normal([num_cells, len(unique_characters)]))
    biases = tf.Variable(tf.random_normal([len(unique_characters)]))

    lstm = tf.contrib.rnn.BasicLSTMCell(num_cells)
    output = tf.nn.static_rnn(cell=lstm, inputs=[tf.convert_to_tensor(input_p)], dtype=tf.float32)[0]

    pred = tf.matmul(output[-1], weights) + biases

    return pred

input_data, output_data, unique_characters = embed(read_files())


#These become our input and output nodes.
input_p =  tf.placeholder(tf.float32, [None, len(unique_characters)])
output_p = tf.placeholder(tf.float32, [None, len(unique_characters)])

pred = make_rnn()

#Numpy inputs
in_batch = input_data[:batch_size]
out_batch = output_data[:batch_size]

#Now they're tensors
tf_in_batch = tf.convert_to_tensor(in_batch, tf.float32)
tf_out_batch = tf.convert_to_tensor(out_batch, tf.float32)

#Our sweet tensorflow session!
sess = tf.Session()

#Initialise
init = tf.global_variables_initializer()
sess.run(init)

# Parameters for back propagation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=output_p)) #Our cost function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) #We will use gradient descent to find a local minimum of the cost function.
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(output_p, 1)) #
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Initial session with random stuff in it. (i bet)


#Try sweet run
for j in range(num_epochs):
    for i in range(len(input_data) // batch_size):
        in_batch = input_data[i * batch_size: i * batch_size + batch_size]
        out_batch =  output_data[i * batch_size:i * batch_size + batch_size]

        sess.run(optimizer, feed_dict={input_p: in_batch, output_p:out_batch})

    one_run = sess.run(pred, feed_dict = {input_p: in_batch})
    tac = ''
    for i in range(batch_size):
        #print(unique_characters[input_data[i].tolist().index(input_data[i].max())])
        tac +=  unique_characters[one_run[i].tolist().index(one_run[i].max())]
    print("Batch nr ", j, ": ", tac)