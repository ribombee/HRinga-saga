import tensorflow as tf
import numpy as np

def activate_clean(output, unique_characters):
    '''
    Function borrowed from RNN.py
    '''
    output = np.exp(output / 0.5)
    output = output / sum(output)
    index_of_probability = np.random.choice(range(len(unique_characters)), p = output)
    clean_output = np.zeros(len(output))
    clean_output[index_of_probability] = 1

    return index_of_probability, clean_output

def get_unique_characters():
    '''
    Queries the stored information about unique characters from when 
    model was created.
    '''
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

if __name__ == '__main__':        
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('./model/test_save.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        unique_chars = get_unique_characters()
        sequence_len = tf.global_variables('sequence_length')[0].eval()
        
        length = input('How long should the story be <press enter to skip>?: ')
        seed_string = 'Enter a seed, max ' + sequence_len.astype(str) + ' characters <press enter to skip>: '
        user_seed = input(seed_string)

        #if nothing was entered for length or seed, default values are selected
        if(len(length) == 0):
            length = 200
        if(len(user_seed) == 0):
            user_seed = 'Mörður hét maður er kallaður var gígja.'

        #TODO: maybe have some padding. It's probably bad for generation but better than errors.

        #shorten the seed to sequence length
        user_seed = user_seed[:sequence_len]

        user_onehots = np.zeros([1, len(user_seed), len(unique_chars)])
        for i in range(len(user_seed)):
            user_onehots[0][i][unique_chars.index(user_seed[i])] = 1

        seed = user_onehots

        input_p = tf.get_collection('input_p')[0]
        pred = tf.get_collection('pred')[0]

        one_run = sess.run([pred], feed_dict={input_p: seed})

        output = user_seed
        for i in range(1000):
            predicted = np.asarray(one_run[0]).astype('float64')[0]
            index, probabilities = activate_clean(predicted, unique_chars)
            seed = seed[:,1:,:] # Remove the first character from the seed

            seed = np.append(seed, np.reshape(probabilities, [1, 1, len(unique_chars)]), 1)  # Append the just-generated character to the seed
            one_run = sess.run([pred], feed_dict={input_p: seed})

            predicted_chars = unique_chars[index]
            output += predicted_chars
        print(output)

