import csv
import numpy as np
import tensorflow as tf
from NeuralNet import fc_model_fn
from random import shuffle
import sys
tf.logging.set_verbosity(tf.logging.INFO)

def main(action):
    # Load training and eval data
    print(action)
    classifier = tf.estimator.Estimator(
        model_fn=fc_model_fn, model_dir="/tmp/fcnet_model")
    if action == 'train':
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
        data = csv_list('red.csv', 0) + csv_list('yellow.csv', 1)
        shuffle(data)
        train_values = [data[i][:3] for i in range(len(data) - 100)]
        train_labels = [data[i][3] for i in  range(len(data) - 100)]
        eval_values = [data[i][:3] for i in  range(len(data) - 100,len(data))]
        eval_labels = [data[i][3] for i in range(len(data) - 100, len(data))]

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(train_values)},
            y=np.array(train_labels),
            batch_size=40,
            num_epochs=None,
            shuffle=False)
        classifier.train(
            input_fn=train_input_fn,
            steps=700,
            hooks=[logging_hook])
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(eval_values)},
            y=np.array(eval_labels),
            num_epochs=1,
            shuffle=False)
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)
    elif action == 'predict':

        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': np.array([[np.float32(196),np.float32(150),np.float32(152)]])},
            y=np.array([0]),
            shuffle=False,
            batch_size=1)

        output = classifier.predict(input_fn=pred_input_fn)
        print(list(output))

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

main(sys.argv[1])