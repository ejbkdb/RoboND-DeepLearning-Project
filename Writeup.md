## RoboND Deep Learning Project: Follow Me

[//]: # (Image References)
[lr1]: https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/pics/lr1.png
[lr2]: https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/pics/lr2.png
[lr3]: https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/pics/lr3.png
[lrall]: https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/pics/learning_rate.jpg
### Overview
* Network Architecture
  * Summary
  * Encoder
  * Decoder
* Hyperparameter Training


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

      Each encoder block contains 1 seperable convolution layer that has batch normalization and utilizes Relu activation.

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

     I created a filter variable within the fcn_model. However I did not tune it much. Modifying other hyperparameter values proved sufficient
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

The parameters below were used in [model_10](https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/data/weights/model_weights_10) training.

The steps per epoch was set so that each epoch was one run through all of the training images

The original batchsize was set to 32, but after identifying an acceptable learning rate I adjusted the batchsize to see how it influenced
the performance of the model. 

```
learning_rate = .01
batch_size = 32
num_epochs = 50
steps_per_epoch = 128
validation_steps = validation_steps = 30
workers = 2

[Final Score: 47%](https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/pics/finalscore.JPG)


```
Tuning:
* Varied learning rate: .01, .001, .0001
* Varied Batch size, 32, 64, 128
* Varied Filters 16, 32

![Learing Rate All][lrall]
Learning Rate Results

Results from Tuning can be found [here](https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/data/weights/learning_rate.xlsx)

![Learing Rate 0.01][lr1]
Learning Rate 0.01

![Learing Rate 0.001][lr2]
Learning Rate 0.001

![Learing Rate 0.0001][lr3]
Learning Rate 0.0001

When evaluating the scores of the learning rates it was found that model performance degraded as the learning rate was reduced. Its possible the models were allowed enough time to train. Based on the scores though, I didn't anticipate a considerable improvement from one learning rate to the next by allowing the model to continue to run. I also varied batch and filter size. This had a minimal impact on model performance. The result varied by ~1-2% depending on batch size and filter. Again, there was not a step change in performance by tuning these hyper parameters.

Would this model perform well tracking other features? Yes, we would just need to train it with new mask data.

[link to html code](https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/code/model_training.html)
