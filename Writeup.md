## RoboND Deep Learning Project: Follow Me

### Overview
* Network Architecture
  * Summary
  * Encoder
  * Decoder
* Hyperparameter Training
* Network Tuning

### Network Architecture:
  * Summary:
    Utilized a fully connected network to evaluate every pixel of an image allowing for per-pixel class labels to enable semantic segmentation.
    The network is broke up into three parts. 1: Encoder Layers, 2: convolution layers, 3: decoder layers.
   The FCN uses:
    * 3 encoder blocks
    * 1x1 Convolution Layer
    * 3 Decoder Blocks

   ```

def fcn_model(inputs, num_classes):
    filter = 32
    # TODO Add Encoder Blocks.
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder1 = encoder_block(inputs, filter, 2)
    encoder2 = encoder_block(encoder1, filter*2, 2)
    encoder3 = encoder_block(encoder2, filter*2*2, 2)


    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_layer = conv2d_batchnorm(encoder3, filter*2*2, 1, 1)
    print("1-by-1 Conv", conv_layer.shape)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder1 = decoder_block(conv_layer, encoder2,filter*2*2)
    decoder2 = decoder_block(decoder1, encoder1,filter*2)
    decoder3 = decoder_block(decoder2, inputs,filter)



    output_layer = layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decoder3)
    return output_layer

   ```

   * Encoder Block:

      Each encoder block contains 1 seperable convolution layer.

```

def encoder_block(input_layer, filters, strides):

    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```

   * Decoder Block:

      Each decoder block contains:

     * Bilinear Sampling Layer
     * Layer Concatination
     * Seperable convolution layer

     The decoders concatenation step concatenates the current input with the output of of the corresponding layer ahead of it.
     Concatenation steps help retain spatial information between layers.

     I created a filter variable within the fcn_model. However I did not tune it. Modifying other hyperparameter values proved sufficient
     to achieve the project requirement.
```
def decoder_block(small_ip_layer, large_ip_layer, filters):

    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsampled_layer = bilinear_upsample(small_ip_layer)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    concatened_layers = layers.concatenate([upsampled_layer, large_ip_layer])
    # TODO Add some number of separable convolution layers
    separable_layer = separable_conv2d_batchnorm(concatened_layers, filters, 1)
    output_layer = separable_conv2d_batchnorm(separable_layer, filters, 1)
    return output_layer
```

### Hyperparameter Training

The parameters below were used in [model_4](https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/data/weights/model_weights_4) training.

The steps per epoch was set so that each epoch was one run through all of the training images

The original batchsize was set to 32, but after identifying an acceptable learning rate I adjusted the batchsize to see how it influenced
the performance of the model. A slight improvement was seen by doubling the batch size from 32 to 64.

```
learning_rate = .0001
batch_size = 64
num_epochs = 40
steps_per_epoch = 63
validation_steps = validation_steps = 15
workers = 2
```
Tuning:
* Varied learning rate: .01, .001, .0001
* Varied Batch size, 32, 64, 128. Kept learning rate at .0001
