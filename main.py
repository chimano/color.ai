import csv
import numpy as np
import tensorflow as tf
from NeuralNet import fc_model_fn
from random import shuffle
def main(unused_argv):
    # Load training and eval data

    mnist_classifier = tf.estimator.Estimator(
        model_fn=fc_model_fn, model_dir="/tmp/mnist_convnet_model")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    data = csv_list('red.csv', 0) + csv_list('yellow.csv', 1)
    shuffle(data)
    train_values = [data[i][:3] for i in  range(200)]
    train_labels = [data[i][3] for i in  range(200)]
    eval_values = [data[i][:3] for i in  range(250, 300)]
    eval_labels = [data[i][3] for i in range(250, 300)]
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(train_values)},
        y=np.array(train_labels),
        batch_size=50,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=200,
        hooks=[logging_hook])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(eval_values)},
        y=np.array(eval_labels),
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

def csv_list(filename, l):
    data_list = [[]]
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            r = np.float32(row[0])
            g = np.float32(row[1])
            b = np.float32(row[2])
            label = l

            data_list.append([r,g,b,label])
    return data_list[1:]

main("nah")