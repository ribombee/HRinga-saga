import tensorflow as tf
import numpy as np

#length = input('How long should the story be?: ')
#seed = input('Please start the story with blabla: ')

def activate_clean(output, unique_characters):
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

def get_unique_characters():
    temp = tf.global_variables('unique_characters')
    temp = temp[0].eval()
    unique_characters = []
    for i, char in enumerate(temp):
        unique_characters += temp[i].decode('utf-8')
    return unique_characters

def string_to_matrix(string, unique_characters):
    matrix = np.zeros((len(string), len(unique_characters)))
    for i, char in enumerate(string):
        matrix[i, unique_characters.index(char)] = 1
    return tf.reshape(matrix, [1, len(string), len(unique_characters)]).eval()

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./model/test_save.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./model/'))
    unique_chars = get_unique_characters()
    seed = string_to_matrix('asdasdasdasdasdasdas', unique_chars)

    input_p = tf.get_collection('input_p')[0]
    pred = tf.get_collection('pred')[0]

    one_run = sess.run([pred], feed_dict={input_p: seed})

    output = ''
    for i in range(1000):
        predicted = np.asarray(one_run[0]).astype('float64')[0]
        index, probabilities = activate_clean(predicted, unique_chars)
        seed = seed[:,1:,:] # Remove the first character from the seed

        seed = np.append(seed, np.reshape(probabilities, [1, 1, len(unique_chars)]), 1)  # Append the just-generated character to the seed
        one_run = sess.run([pred], feed_dict={input_p: seed})

        predicted_chars = unique_chars[index]
        output += predicted_chars
    print(output)

