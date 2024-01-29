import tensorflow as tf
import numpy as np
import keras
import keras.utils as utils
import cv2


class GradCAM():

    VGG_CONV_LAYER_NAME = 'block5_conv3'
    BASE_CONV_LAYER_NAME = 'conv2d_8'
    
    def __init__(self, modelpath, uploaded_file, model_id):
        self.modelpath = modelpath
        self.uploaded_file = uploaded_file
        self.model = self.instantiate_model()
        self.image = self.process_image()
        self.heatmap, self.cam = self.grad_cam(layer_name=self.get_conv_layer_name(model_id))

    def instantiate_model(self, compile=False):
        return tf.keras.models.load_model(self.modelpath, compile=compile)

    def process_image(self, size=(32, 32)):
        img = utils.load_img(self.uploaded_file, target_size=size)
        img_arr = utils.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        return img_arr/255.0
    
    # refactored from https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    def grad_cam(self, layer_name, eps=1e-8):

        with tf.GradientTape() as tape:

            model = keras.models.Model(inputs=[self.model.inputs],
                                       outputs=[self.model.get_layer(layer_name).output, self.model.output])

            tape.watch(model.get_layer(layer_name).variables)
            input = tf.cast(self.image, tf.float32)
            
            conv_outputs, predictions = model(input)
            #print(predictions.numpy().squeeze())
            #print(bool(tf.cast(predictions < 0.5, "float32").numpy().squeeze()))
            
            loss = predictions[0]


        grads = tape.gradient(loss, conv_outputs)

        cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads
        
        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]
        
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        
        image = self.process_image(size=(224, 224))
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        img = (image.reshape(image.shape[1:]) * 255.0).astype(np.uint8)
        return heatmap, self.overlay_heatmap(heatmap, img)
        
    def overlay_heatmap(self, heatmap, image, alpha=0.3, colormap=cv2.COLORMAP_JET):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return output
    
    def get_conv_layer_name(self, model_id: int = 1):
        # returns last convolutional layer name based on model id (id corresponds to number in 'pages')
        if model_id == 1:
            return self.BASE_CONV_LAYER_NAME
        elif model_id == 2:
            # change to last convolution layer name of efficent net model to implement grad cam
            return None
        elif model_id == 3:
            return self.VGG_CONV_LAYER_NAME
        else:
            return None

    @property
    def gradCAM(self):
        return self.cam
