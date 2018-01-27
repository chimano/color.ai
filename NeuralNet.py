import tensorflow as tf


def fc_model_fn(features, labels, mode):
    input_layer = tf.layers.Input(shape=(3,), batch_size=-1, dtype=tf.int32)

    fc_1 = tf.layers.dense(inputs=input_layer, units=3, activation=tf.nn.relu)
    fc_2 = tf.layers.dense(inputs=fc_1, units=8, activation=tf.nn.relu)
    fc_3 = tf.layers.dense(inputs=fc_2, units=18, activation=tf.nn.relu)
    
    dropout = tf.layers.dropout(inputs=fc_3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=11)
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    
    if mode = tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=11)

    loss = tf.losses.softmax_cross_entropy( onehot_labels=onehot_labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
        }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)