import tensorflow as tf
import argparse

'''
This script is used to add signatures/signature definitions to tensorflow graphs,if it doesn't already exist.
These are essential to converting to TFLite as they define the input and outputs of the graph
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    args = parser.parse_args()

    loaded = tf.saved_model.load(args.model_dir)
    if len(list(loaded.signatures.keys())) == 0:
        call = loaded.__call__.get_concrete_function(tf.TensorSpec(None, tf.float32))
        tf.saved_model.save(loaded, args.model_dir, signatures=call)
        loaded = tf.saved_model.load(args.model_dir)
        print('signature added:', list(loaded.signatures.keys()))
    else:
        print('signature exists:', list(loaded.signatures.keys()))
