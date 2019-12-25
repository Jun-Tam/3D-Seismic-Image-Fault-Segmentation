from keras.models import Model
from metrics import cross_entropy_balanced
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from keras import optimizers
from keras.utils.vis_utils import plot_model

class create_model:
    def __init__(self, input_size, lr):
        self.compile_convnet(input_size, lr)
    
    def compile_convnet(self, input_size, lr):
        def Conv3D_with_prm(model_in, num_filters):
            conv = Conv3D(filters=num_filters,
                          kernel_size=(3,3,3),
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(model_in)
            return conv
        
        def conv3d_down(model_in, num_filters, pooling=False):
            pool = []
            conv = Conv3D_with_prm(model_in, num_filters)
            conv = Conv3D_with_prm(conv, num_filters)
            if pooling:
                pool = MaxPooling3D(pool_size=(2,2,2))(conv)
            return conv, pool

        def conv3d_up(model_in, model_merge, num_filters):
            up = UpSampling3D(size=(2,2,2))(model_in)
            merge = concatenate([up, model_merge], axis=4)
            conv = Conv3D_with_prm(merge, num_filters)
            conv = Conv3D_with_prm(conv, num_filters)
            return conv
       
        # Downward
        inputs = Input(input_size)    
        conv1, pool1 = conv3d_down(inputs, 2**4, pooling=True)
        conv2, pool2 = conv3d_down(pool1, 2**5, pooling=True)
        conv3, pool3 = conv3d_down(pool2, 2**6, pooling=True)
        
        # Bottom
        conv4 = Conv3D_with_prm(pool3, 2**9)
        conv4 = Conv3D_with_prm(conv4, 2**9)
        
        # Upward
        conv5 = conv3d_up(conv4, conv3, 2**6)
        conv6 = conv3d_up(conv5, conv2, 2**5)
        conv7 = conv3d_up(conv6, conv1, 2**4)
        outputs = Conv3D(1, kernel_size=(1,1,1), activation='sigmoid')(conv7)
        model = Model(input=inputs, output=outputs)
        model.compile(optimizer=optimizers.Adam(lr=1e-4),
                      loss = cross_entropy_balanced, metrics = ['accuracy'])
        self.model = model

    def visualize_model(self, path_savefig='model_plot.png'):
        self.model.summary()
        plot_model(self.model, to_file=path_savefig, show_shapes=True, show_layer_names=True)         

#model = create_model((128,128,128,1)) 