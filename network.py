import tensorflow as tf
import pickle
import numpy as np
train_x, train_y, test_x, test_y, n_classes = pickle.load(open('note_features.pickle', 'rb'))

n_nodes_hl1 = 500
n_nodes_hl2 = 500

#n_classes = 2
hm_data = 2000000

batch_size = 100
hm_epochs = 10

n = 2
x = tf.placeholder('float', [None, n])
y = tf.placeholder('float')

current_epoch = tf.Variable(1)

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([n, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes]))}

def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 1
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+=batch_size                  
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:test_x,y:test_y}))
        save = saver.save(sess, "./model.ckpt")
        print("Model saved to file:", save)

#train_neural_network(x)

def use_neural_network(input_data):
    prediction = neural_network_model(x)
    # with open('lexicon.pickle','rb') as f:
    #     lexicon = pickle.load(f)
        
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,"model.ckpt")
        features = input_data[:]
        while 1:
            print(features)
            result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
            print(result)
            features = [result[0], 1]

use_neural_network([48, 0])
