import tensorrt as trt


class ModelData(object):
    INPUT_NAME  = "in_frame"
    # P, C, H, W
    INPUT_SHAPE = (1, 3, 224, 224)
    OUTPUT_NAME = "out_frame"
    DTYPE       = trt.float32


def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor      = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    # VGG16 features
    # VGG16_block_1
    vgg16_f0_w        = weights['features.0.weight'].numpy()
    vgg16_f0_b        = weights['features.0.bias'].numpy()
    vgg16_f0          = network.add_convolution(input=input_tensor, num_output_maps=64, kernel_shape=(3, 3), kernel=vgg16_f0_w, bias=vgg16_f0_b)
    vgg16_f0.padding  = (1, 1)
    vgg16_f0.name     = 'vgg16_conv_1_1'
    vgg16_f1          = network.add_activation(input=vgg16_f0.get_output(0), type=trt.ActivationType.RELU)
    vgg16_f1.name     = 'vgg16_relu_1_1'
    vgg16_f2_w        = weights['features.2.weight'].numpy()
    vgg16_f2_b        = weights['features.2.bias'].numpy()
    vgg16_f2          = network.add_convolution(input=vgg16_f1.get_output(0), num_output_maps=64, kernel_shape=(3, 3), kernel=vgg16_f2_w, bias=vgg16_f2_b)
    vgg16_f2.padding  = (1, 1)
    vgg16_f2.name     = 'vgg16_conv_1_2'
    vgg16_f3          = network.add_activation(input=vgg16_f2.get_output(0), type=trt.ActivationType.RELU)
    vgg16_f3.name     = 'vgg16_relu_1_2'
    vgg16_f4          = network.add_pooling(input=vgg16_f3.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    vgg16_f4.stride   = (2, 2)
    vgg16_f4.name     = 'vgg16_max_pool_1'

    # VGG16_block_2
    vgg16_f5_w        = weights['features.5.weight'].numpy()
    vgg16_f5_b        = weights['features.5.bias'].numpy()
    vgg16_f5          = network.add_convolution(input=vgg16_f4.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=vgg16_f5_w, bias=vgg16_f5_b)
    vgg16_f5.padding  = (1, 1)
    vgg16_f5.name     = "vgg16_conv_2_1"
    vgg16_f6          = network.add_activation(input=vgg16_f5.get_output(0), type=trt.ActivationType.RELU)
    vgg16_f6.name     = 'vgg16_relu_2_1'
    vgg16_f7_w        = weights['features.7.weight'].numpy()
    vgg16_f7_b        = weights['features.7.bias'].numpy()
    vgg16_f7          = network.add_convolution(input=vgg16_f6.get_output(0), num_output_maps=128, kernel_shape=(3, 3), kernel=vgg16_f7_w, bias=vgg16_f7_b)
    vgg16_f7.padding  = (1, 1)
    vgg16_f7.name     = "vgg16_conv_2_2"
    vgg16_f8          = network.add_activation(input=vgg16_f7.get_output(0), type=trt.ActivationType.RELU)
    vgg16_f8.name     = 'vgg16_relu_2_2'
    vgg16_f9          = network.add_pooling(input=vgg16_f8.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    vgg16_f9.stride   = (2, 2)
    vgg16_f9.name     = 'vgg16_max_pool_2'

    # VGG16_block_3
    vgg16_f10_w       = weights['features.10.weight'].numpy()
    vgg16_f10_b       = weights['features.10.bias'].numpy()
    vgg16_f10         = network.add_convolution(input=vgg16_f9.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=vgg16_f10_w, bias=vgg16_f10_b)
    vgg16_f10.padding = (1, 1)
    vgg16_f10.name    = "vgg16_conv_3_1"
    vgg16_f11         = network.add_activation(input=vgg16_f10.get_output(0), type=trt.ActivationType.RELU)
    vgg16_f11.name    = 'vgg16_relu_3_1'
    vgg16_f12_w       = weights['features.12.weight'].numpy()
    vgg16_f12_b       = weights['features.12.bias'].numpy()
    vgg16_f12         = network.add_convolution(input=vgg16_f11.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=vgg16_f12_w, bias=vgg16_f12_b)
    vgg16_f12.padding = (1, 1)
    vgg16_f12.name    = "vgg16_conv_3_2"
    vgg16_f13         = network.add_activation(input=vgg16_f12.get_output(0), type=trt.ActivationType.RELU)
    vgg16_f13.name    = 'vgg16_relu_3_2'
    vgg16_f14_w       = weights['features.14.weight'].numpy()
    vgg16_f14_b       = weights['features.14.bias'].numpy()
    vgg16_f14         = network.add_convolution(input=vgg16_f13.get_output(0), num_output_maps=256, kernel_shape=(3, 3), kernel=vgg16_f14_w, bias=vgg16_f14_b)
    vgg16_f14.padding = (1, 1)
    vgg16_f14.name    = "vgg16_conv_3_3"
    vgg16_f15         = network.add_activation(input=vgg16_f14.get_output(0), type=trt.ActivationType.RELU)
    vgg16_f15.name    = 'vgg16_relu_3_3'
    vgg16_f16         = network.add_pooling(input=vgg16_f15.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    vgg16_f16.stride  = (2, 2)
    vgg16_f16.name    = 'vgg16_max_pool_3'

    # VGG16_block_4
    vgg16_f17_w       = weights['features.17.weight'].numpy()
    vgg16_f17_b       = weights['features.17.bias'].numpy()
    vgg16_f17         = network.add_convolution(input=vgg16_f16.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=vgg16_f17_w, bias=vgg16_f17_b)
    vgg16_f17.padding = (1, 1)
    vgg16_f17.name    = "vgg16_conv_4_1"
    vgg16_f18         = network.add_activation(input=vgg16_f17.get_output(0), type=trt.ActivationType.RELU)
    vgg16_f18.name    = 'vgg16_relu_4_1'
    vgg16_f19_w       = weights['features.19.weight'].numpy()
    vgg16_f19_b       = weights['features.19.bias'].numpy()
    vgg16_f19         = network.add_convolution(input=vgg16_f18.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=vgg16_f19_w, bias=vgg16_f19_b)
    vgg16_f19.padding = (1, 1)
    vgg16_f19.name    = "vgg16_conv_4_2"
    vgg16_f20         = network.add_activation(input=vgg16_f19.get_output(0), type=trt.ActivationType.RELU)
    vgg16_f20.name    = 'vgg16_relu_4_2'
    vgg16_f21_w       = weights['features.21.weight'].numpy()
    vgg16_f21_b       = weights['features.21.bias'].numpy()
    vgg16_f21         = network.add_convolution(input=vgg16_f20.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=vgg16_f21_w, bias=vgg16_f21_b)
    vgg16_f21.padding = (1, 1)
    vgg16_f21.name    = "vgg16_conv_4_3"
    vgg16_f22         = network.add_activation(input=vgg16_f21.get_output(0), type=trt.ActivationType.RELU)
    vgg16_f22.name    = 'vgg16_relu_4_3'
    vgg16_f23         = network.add_pooling(input=vgg16_f22.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    vgg16_f23.stride  = (2, 2)
    vgg16_f23.name    = 'vgg16_max_pool_4'

    # VGG16_block_5
    vgg16_f24_w       = weights['features.24.weight'].numpy()
    vgg16_f24_b       = weights['features.24.bias'].numpy()
    vgg16_f24         = network.add_convolution(input=vgg16_f23.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=vgg16_f24_w, bias=vgg16_f24_b)
    vgg16_f24.padding = (1, 1)
    vgg16_f24.name    = "vgg16_conv_5_1"
    vgg16_f25         = network.add_activation(input=vgg16_f24.get_output(0), type=trt.ActivationType.RELU)
    vgg16_f25.name    = "vgg16_relu_5_1"
    vgg16_f26_w       = weights['features.26.weight'].numpy()
    vgg16_f26_b       = weights['features.26.bias'].numpy()
    vgg16_f26         = network.add_convolution(input=vgg16_f25.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=vgg16_f26_w, bias=vgg16_f26_b)
    vgg16_f26.padding = (1, 1)
    vgg16_f26.name    = "vgg16_conv_5_2"
    vgg16_f27         = network.add_activation(input=vgg16_f26.get_output(0), type=trt.ActivationType.RELU)
    vgg16_f27.name    = "vgg16_relu_5_2"
    vgg16_f28_w       = weights['features.28.weight'].numpy()
    vgg16_f28_b       = weights['features.28.bias'].numpy()
    vgg16_f28         = network.add_convolution(input=vgg16_f27.get_output(0), num_output_maps=512, kernel_shape=(3, 3), kernel=vgg16_f28_w, bias=vgg16_f28_b)
    vgg16_f28.padding = (1, 1)
    vgg16_f28.name    = "vgg16_conv_5_3"
    vgg16_f29         = network.add_activation(input=vgg16_f28.get_output(0), type=trt.ActivationType.RELU)
    vgg16_f29.name    = "vgg16_relu_5_3"
    vgg16_f30         = network.add_pooling(input=vgg16_f29.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    vgg16_f30.stride  = (2, 2)
    vgg16_f30.name    = 'vgg16_max_pool_5'

    # VGG16 nn.AdaptiveAvgPool2d((7, 7))
    vgg16_a0          = network.add_pooling(input=vgg16_f30.get_output(0), type=trt.PoolingType.AVERAGE, window_size=(1, 1))
    vgg16_a0.name     = 'vgg16_avg_pool_0'

    # VGG16 torch.flatten(x, 1)
    # there is no need for torch.flatten(x, 1). because, tensorrt.IFullyConnectedLayer would first reshape the input
    # tensor from shape {P, C, H, W} into {P, C*H*W}.

    # VGG16 classifier
    # VGG16_fc_1
    vgg16_c0_w        = weights['classifier.0.weight'].numpy()
    vgg16_c0_b        = weights['classifier.0.bias'].numpy()
    vgg16_c0          = network.add_fully_connected(input=vgg16_a0.get_output(0), num_outputs=4096, kernel=vgg16_c0_w, bias=vgg16_c0_b)
    vgg16_c0.name     = "vgg16_fc_1"
    vgg16_c1          = network.add_activation(input=vgg16_c0.get_output(0), type=trt.ActivationType.RELU)
    vgg16_c1.name     = "vgg16_relu_fc_1"
    # there is no need for Dropout during inference

    # VGG16_fc_2
    vgg16_c3_w        = weights['classifier.3.weight'].numpy()
    vgg16_c3_b        = weights['classifier.3.bias'].numpy()
    vgg16_c3          = network.add_fully_connected(input=vgg16_c1.get_output(0), num_outputs=4096, kernel=vgg16_c3_w, bias=vgg16_c3_b)
    vgg16_c3.name     = "vgg16_fc_2"
    vgg16_c4          = network.add_activation(input=vgg16_c3.get_output(0), type=trt.ActivationType.RELU)
    vgg16_c4.name     = "vgg16_relu_fc_2"
    # there is no need for Dropout during inference

    # VGG16_fc_3
    vgg16_c6_w        = weights['classifier.6.weight'].numpy()
    vgg16_c6_b        = weights['classifier.6.bias'].numpy()
    vgg16_c6          = network.add_fully_connected(input=vgg16_c4.get_output(0), num_outputs=1000, kernel=vgg16_c6_w, bias=vgg16_c6_b)
    vgg16_c6.name     = "vgg16_fc_3"
    # Output
    vgg16_c6.get_output(0).name = ModelData.OUTPUT_NAME
    network.mark_output(tensor=vgg16_c6.get_output(0))
