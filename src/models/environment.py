import os

# Set couple flags to configure Tensorflow and Keras
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Without the last flag importing said modules will always print
# a warning printout, so this is the way to suppress that

# Why in a separate file? Excelent questiom
# Since python linters want import statements to be the first lines,
# setting these flags will always cause linters to have a stroke as
# they have to be set before importing the modules...

# So this is a dumb workararound for that :)
