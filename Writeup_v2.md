## RoboND Deep Learning Project: Follow Me

[//]: # (Image References)
[lr1]: https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/pics/lr1.png
[lr2]: https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/pics/lr2.png
[lr3]: https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/pics/lr3.png
[lrall]: https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/pics/learning_rate.jpg
[fcn]: https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/pics/FCN.png
### Overview
* Network Architecture
  * Summary
  * Encoder
  * 1x1 Convolutional Layer
  * Decoder
* Hyperparameter Training
* Future Enhancements

### Network Architecture:
  * Summary:
    Utilized a fully connected network to evaluate every pixel of an image allowing for per-pixel class labels to enable semantic segmentation.
    The network is broke up into three parts. 1: Encoder Layers, 2: convolution layers, 3: decoder layers.
   The FCN uses:
    * 3 encoder blocks
    * 1x1 Convolution Layer
    * 3 Decoder Blocks
    
  #### Skip Connections:
  
  Skip Connections are what give the FCN the ability to retain information from previous layers. With Skip Connections we can use the output from an encoder layer as the input for a decoder layer. This allows the model to learn from multiple resolutions which helps it retain information that may have other wise been lost
  
  #### FCN Model
  
  I ended up using a filter of 32, with 3 encoder, 3 decoder, and 1 convolutionals layer. I tested other filters after i had tuned my hyperparameters but little improvement was found. The number of encoder/decoder layers were set at 3 to achieve a 20x20 output size. This seemed like an appropriately sized layer vs. a 2 layer network which would have results in a 40x40 layer. 
 
 ![network][fcn]

   ```python
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

      Each encoder block contains 1 seperable convolution layer that has batch normalization and utilizes Relu activation. Each layer in the encoder decreases the images resolution by 1/2 (160>80>40>20) and increases the number of filter maps (depth). This reduction in size causes the FCN to find generic features in the data. This generalization of the data features is preferred as it reduces the changes of overfitting.

```python

def encoder_block(input_layer, filters, strides):

    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```
   * Convolutional Layer
   
     A 1x1 convolutional layers econnects the encoder block to the decoder block. The layer is used to compute a pixel by pixel classification. The convolutional layer output results in a operation that preserves spatial information. This spatial information is required for semantic segmentation. 

     If the convolutional layer was replaced with a fully connected layer. The output dimensions would be flattened to 2 dimensions. This flatting would destroy the spatial information and sematic segmentation would not be feasible. 
   
```python
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

   * Decoder Block:

      Each decoder block contains:

     * Bilinear Sampling Layer
     * Layer Concatination
     * Seperable convolution layer

     The decoder works in the reverse as the encoder. From the 1x1 convolution the decoder recieves the output classifications and upscales the image layer by layer until it matches the input resolution. Skip connections are also used at all points in the decoder which in this case concatenates the input to the encoder layer of the same size with the decoded layer.
     
     The decoder uses a bilinear upsampling technique. Bilinear upsampling computes the weighted average of the local 4 diagonal pixels to estimate a new pixel value. The downside to the upsampling technique is that details from the original image are lost during the upscaling. The skip connections employed in the FCN are used to reduce this effect.

     I created a filter variable within the fcn_model. However I did not tune it much. Modifying other hyperparameter values proved sufficient to achieve the project requirement.

```python
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

```python
learning_rate = .01
batch_size = 32
num_epochs = 50
steps_per_epoch = 128
validation_steps = validation_steps = 30
workers = 2
```

Final Score: 47% 
[link](https://github.com/ejbkdb/RoboND-DeepLearning-Project/blob/master/pics/finalscore.JPG)



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

### Future Enhancements
  * If I could train on multiple gpu instances I would have taken a more DOE approach to tuning the Hyperparameters. But my initial conditions were really good so I just played around with the other parameter some to see how they impacted the results. A more scientific and methodical appropach should be employed in the future
  * I used the 'stock' training data in the lab. I would attempt to generate my own datasets in a future revision
  * I would consider modifying the layer count to see how it effects the model. It would be a good learning experience.
  
