import torch
import tensorrt as trt
from vgg16_397923af_trt import populate_network


def build_engine(weights):
    # flag implies the input batch is explicit. The input shape is {P * C * H * W}.
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flag) as network:
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        builder.max_workspace_size = 4 * 1 << 30
        config = builder.create_builder_config()
        return builder.build_engine(network, config)


vgg16_path = './vgg16-397923af.pth'
vgg16_weights = torch.load(vgg16_path, map_location='cpu')
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# Do inference with TensorRT
with build_engine(vgg16_weights) as engine:
    # Build an engine, allocate buffers and create a stream.
    host_memory = engine.serialize()
    output_engine = 'vgg16-397923af_fp32.engine'
    print("===> Save %s\n" % output_engine)
    with open(output_engine, "wb") as f:
        f.write(engine.serialize())
