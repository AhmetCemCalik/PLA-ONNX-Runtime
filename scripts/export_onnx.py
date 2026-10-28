# Converts the trained PLA weights into perceptron.onnx file

import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper

weights = np.loadtxt( "build/weights.txt", dtype=np.float64 )

bias = np.array([weights[ 0 ]], dtype=np.float64 )
W = weights[1:].reshape( -1, 1 )

input_dim = len( weights ) - 1

# Define model input and output
X = helper.make_tensor_value_info( "X", TensorProto.DOUBLE, [ None, input_dim ])
Y = helper.make_tensor_value_info( "Y", TensorProto.INT64, [ None, 1 ])

# Store learned parameters and constants
initializers = [
    numpy_helper.from_array( W, name="W" ),
    numpy_helper.from_array( bias, name="B"),
    numpy_helper.from_array( np.array([0.0], dtype=np.float64), name="Zero"),
    numpy_helper.from_array( np.array([ 1 ], dtype=np.int64), name="Positive"),
    numpy_helper.from_array( np.array([-1 ], dtype=np.int64), name="Negative"),
]

# Build computational graph
nodes = [
    helper.make_node( "MatMul", [ "X", "W" ], [ "WX" ]),
    helper.make_node( "Add", [ "WX", "B" ], [ "Score" ]),
    helper.make_node( "GreaterOrEqual", [ "Score", "Zero" ], [ "IsPositive"]),
    helper.make_node( "Where", [ "IsPositive", "Positive", "Negative"], [ "Y" ]),
]

graph = helper.make_graph(
    nodes,
    "Perceptron",
    [ X ],
    [ Y ],
    initializer=initializers,
)

model = helper.make_model(
    graph,
    producer_name="CPP_PLA",
    opset_imports=[ helper.make_opsetid( "", 13 )],
)

model.ir_version = 9

onnx.checker.check_model( model )
onnx.save( model, "build/perceptron.onnx" )

print( "Created perceptron.onnx" )