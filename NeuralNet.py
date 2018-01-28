import tensorflow as tf


def fc_model_fn(features, labels, mode):

    fc_1 = tf.layers.dense(inputs=features['x'], units=3, activation=tf.nn.relu)
    dropout_1 = tf.layers.dropout(inputs=fc_1, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    fc_2 = tf.layers.dense(inputs=dropout_1, units=80, activation=tf.nn.relu)
    dropout_2 = tf.layers.dropout(inputs=fc_2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    fc_3 = tf.layers.dense(inputs=dropout_2, units=60, activation=tf.nn.relu)
    
    logits = tf.layers.dense(inputs=fc_3, units=11)
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(loss)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
        train_op = optimizer.minimize(
            loss=cross_entropy_mean,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy_mean, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
