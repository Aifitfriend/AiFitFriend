import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import graph_def_editor as ge
from tensorflow.keras.models import model_from_config
from tensorflow.compat.v1.keras import backend as K

print(tf.__version__)

savedModel_path = "/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/saved_model_2/saved_model"
frozenGraph_path = "/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/frozen_models/frozen_graph.pb"


def conv_saved_model_to_frozen(saved_model_path):
    model = tf.saved_model.load(saved_model_path)

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.signatures['serving_default'].inputs[0].shape, model.signatures['serving_default'].inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    # Check if we can access layers in the converted model
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/frozen_models",
                      name="frozen_graph.pb",
                      as_text=False)


def conv_frozen_to_h5(frozen_graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.io.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.graph_util.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        const_var_name_pairs = {}
        probable_variables = [op for op in detection_graph.get_operations() if op.type == "Const"]
        available_names = [op.name for op in detection_graph.get_operations()]
        for op in probable_variables:
            name = op.name
            if name + '/read' not in available_names:
                continue
            tensor = detection_graph.get_tensor_by_name('{}:0'.format(name))
            with tf.compat.v1.Session() as s:
                tensor_as_numpy_array = s.run(tensor)
            var_shape = tensor.get_shape()
            # Give each variable a name that doesn't already exist in the graph
            var_name = '{}_turned_var'.format(name)
            var = tf.Variable(name=var_name, dtype=op.outputs[0].dtype, initial_value=tensor_as_numpy_array,
                              trainable=True, shape=var_shape)
            const_var_name_pairs[name] = var_name

    ge_graph = ge.Graph(detection_graph)
    for const_name, var_name in const_var_name_pairs.items():
        const_op = ge_graph._node_name_to_node[const_name + '/read']
        var_reader_op = ge_graph._node_name_to_node[var_name + '/Read/ReadVariableOp']
        ge.swap_outputs(ge.sgv(const_op), ge.sgv(var_reader_op))

    with detection_graph.as_default():
        meta_saver = tf.compat.v1.train.Saver()
        meta = meta_saver.export_meta_graph()
    ge_graph = ge.Graph(detection_graph, collections=ge.graph._extract_collection_defs(meta))

    test_graph = tf.Graph()
    with test_graph.as_default():
        tf.import_graph_def(ge_graph.to_graph_def(), name="")
        for var_name in ge_graph.variable_names:
            var = ge_graph.get_variable_by_name(var_name)
            ret = variable_pb2.VariableDef()
            ret.variable_name = var._variable_name
            ret.initial_value_name = var._initial_value_name
            ret.initializer_name = var._initializer_name
            ret.snapshot_name = var._snapshot_name
            ret.trainable = var._trainable
            ret.is_resource = True
            tf_var = tf.Variable(variable_def=ret, dtype=tf.float32)
            test_graph.add_to_collections(var.collection_names, tf_var)

conv_saved_model_to_frozen(savedModel_path)
