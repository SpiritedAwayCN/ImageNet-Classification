input_shape = 224, 224, 3
num_class = 1000
total_epoches = 50

batch_size = 128
train_num = 1281167
val_num = 50000

iterations_per_epoch = train_num // batch_size + 1
test_iterations = val_num // batch_size + 1

weight_decay = 1e-4
label_smoothing = 0.1

# 这些是别人算好的
mean = [103.939, 116.779, 123.68]
std = [58.393, 57.12, 57.375]
eigval = [55.46, 4.794, 1.148]
eigvec = [[-0.5836, -0.6948, 0.4203],
          [-0.5808, -0.0045, -0.8140],
          [-0.5675, 0.7192, 0.4009]]