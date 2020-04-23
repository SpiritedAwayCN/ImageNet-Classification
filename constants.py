input_shape = 224, 224, 3
num_class = 100#0
total_epoches = 50

batch_size = 128
train_num = 50000
val_num = 5000

iterations_per_epoch = train_num // batch_size + 1
test_iterations = val_num // batch_size + 1

weight_decay = 1e-4
label_smoothing = 0.1

# for ILSVEC 2012
mean_o = [103.939, 116.779, 123.68]
std_o = [58.393, 57.12, 57.375]
eigval_o = [55.46, 4.794, 1.148]
eigvec_o = [[-0.5836, -0.6948, 0.4203],
          [-0.5808, -0.0045, -0.8140],
          [-0.5675, 0.7192, 0.4009]]
		  
# for mini imageNet
mean_m = [102.38590308, 114.03596477, 120.20542635]
std_m = [73.43149194666307, 70.09254559555757, 72.16840074564995]
eigval_m = [69.1475759, 6.97101293, 1.44854629]
eigvec_m = [[-0.58137452, 0.69014481, 0.43093364],
       [-0.58432837, 0.01440479, -0.81138946],
       [-0.56618374, -0.72352791, 0.39489662]]

# mini imageNet
mean = mean_m
std = std_m
eigval = eigval_m
eigvec = eigvec_m