import time
import tensorflow as tf
import numpy as np
import warnings

def nn_using_tf():


  tf.random.set_seed(42)
  np.random.seed(42)

  N = 500
  X = np.random.uniform(-1,1,size=(N,1)).astype(np.float32)
  true_w = 3.0
  true_b = 2.0
  noise = np.random.normal(0.0, 0.1, size=(N,1)).astype(np.float32)
  Y = (true_w*X + true_b + noise)
  X_tf = tf.constant(X)
  Y_tf = tf.constant(Y)

  input_dim = 1
  hidden_layer = 8
  output_dim  = 1

  W1 = tf.Variable(tf.random.normal([input_dim,hidden_layer], stddev = 0.5), name="w1")
  b1 = tf.Variable(tf.zeros([hidden_layer]), name='b1')

  w2 = tf.Variable(tf.random.normal([hidden_layer,output_dim], stddev = 0.5), name="w2")
  b2 = tf.Variable(tf.zeros([output_dim]), name="b2")

  def forward_pass(x):
    z1 = tf.matmul(x, W1) + b1
    a1 = tf.nn.relu(z1)
    out = tf.matmul(a1,w2) + b2
    return out

  def mse_loss(y_pred, y_true):

     return tf.reduce_mean(tf.square(y_pred - y_true))

  learning_rate = 0.1
  epochs = 200
  batch_size = 32
  steps_per_epoch = int(np.ceil(N/batch_size))

  start_time = time.time()
  for epoch in range(1, epochs+1):
    idx = np.random.permutation(N)
    X_shuffled = X_tf.numpy()[idx]
    Y_shuffled = Y_tf.numpy()[idx]

    epoch_loss = 0.0
    for step in range(steps_per_epoch):
     start = step * batch_size
     end = min(start + batch_size, N)

     x_batch = tf.constant(X_shuffled[start:end])
     y_batch = tf.constant(Y_shuffled[start:end])
     with tf.GradientTape() as tape:

      y_pred = forward_pass(x_batch)
      loss = mse_loss(y_pred, y_batch)

     grad = tape.gradient(loss, [W1, b1, w2, b2])

     W1.assign_sub(learning_rate*grad[0])
     b1.assign_sub(learning_rate*grad[1])
     w2.assign_sub(learning_rate*grad[2])
     b2.assign_sub(learning_rate*grad[3])

     epoch_loss += loss.numpy() * (end - start)

  epoch_loss = epoch_loss / N

  if epoch % 20 == 0 or epoch == 1:
   print(f"Epoch {epoch:3d}/{epochs} - Loss: {epoch_loss:.6f}")

  end_time = time.time()
  print(f"Training finished in {end_time - start_time:.3f} seconds")

  y_pred_final = forward_pass(X_tf)
  final_loss = mse_loss(y_pred_final, Y_tf).numpy()
  print("\nFinal MSE on training set:", final_loss)


  # Print learned linear approximation for sanity (since target is linear)
  # Note: The small hidden net should be able to represent a linear mapping approx.
  # We can run a simple least-square on predictions to see approximate slope/intercept.
  coef = np.polyfit(X.flatten(), y_pred_final.numpy().flatten(), 1)
  print(f"Learned linear fit (approx): y = {coef[0]:.3f}x + {coef[1]:.3f}")
  print(f"True function was y = {true_w}x + {true_b}")

  # -----------------------
  # 8) Inference example: predict on a new point
  # -----------------------
  x_test = tf.constant([[0.5]], dtype=tf.float32)
  y_test_pred = forward_pass(x_test).numpy().squeeze()
  y_test_true = true_w * 0.5 + true_b
  print(f"\nFor x = 0.5, predicted y = {y_test_pred:.3f}, true y = {y_test_true:.3f}")

def basics():

 sample_tensor = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
 print(tf.rank(sample_tensor))

 def add(a,b):
  print(tf.add(a,b))

 def multiply(a,b):
  print(tf.multiply(a,b))

 def matmultiply(a,b):
  print(tf.matmul(a,b))

 a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
 b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
 add(a,b)
 multiply(a,b)
 matmultiply(a,b)


if __name__=='__main__':
    # print(softmax([2.0,1.0,0.1]))
    # basics()
    nn_using_tf()
