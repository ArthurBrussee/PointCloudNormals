import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework.graph_util import convert_variables_to_constants

# Based on code from Chengwei, DLology
# https://www.dlology.com/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/
model = load_model('./models/model_050.hdf5')


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        # output_names = output_names or []
        # output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        transforms = [
            'remove_nodes(op=Identity)',
            'merge_duplicate_nodes',
            'strip_unused_nodes',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
        ]

        optimized_graph_def = TransformGraph(
            frozen_graph,
            ['input_input'],
            output_names,
            transforms)

        if clear_devices:
            for node in optimized_graph_def.node:
                node.device = ""

        return optimized_graph_def


frozen_graph = freeze_session(K.get_session(), output_names=['output/BiasAdd'])

# Save to ./model/tf_model.pb
tf.train.write_graph(frozen_graph, "saved_models", "./tf_model.pb", as_text=False)

print("Saved model!")
