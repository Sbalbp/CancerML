from cnn import *
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Input
from keras.layers.merge import add
from keras.layers.merge import concatenate
from keras import Model
from keras import backend as K
from keras.regularizers import l2

class DenseNet(Network):

    def dense_block(self, input_t, reps):
        merged = input_t

        for rep in range(reps):
            x = BatchNormalization()(merged)
            x = Activation("relu")(x)
            x = Conv2D(4*self.growth_rate, (1, 1), strides = 1, kernel_initializer = RandomNormal(0, 0.0001))(x)
            x = Dropout(self.drop_conv)(x)
            x = ZeroPadding2D(padding = 1)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(self.growth_rate, (3, 3), strides = 1, kernel_initializer = RandomNormal(0, 0.0001))(x)
            x = Dropout(self.drop_conv)(x)
            # Concatenate inputs increasingly
            merged = concatenate([merged, x])

            self.nconvs += 2

        return merged

    def build_architecture(self, insize, **kwargs):
        self.nconvs = 0

        try:
            self.drop_conv = float(kwargs['drop_conv'])
            self.drop_dense = float(kwargs['drop_dense'])

            self.dense_blocks = repetitions = [int(rep) for rep in kwargs['dense_blocks'].split(',')]
            self.growth_rate = int(kwargs['growth_rate'])
            self.theta = 1 if 'compression' not in kwargs.keys() else float(kwargs['compression'])
        except KeyError as e:
            raise(TypeError('build_architecture: Missing argument: %s' % e))

        input_layer = Input(shape = (insize[0], insize[1], 3), name = 'DenseNet_Input')

        x = ZeroPadding2D(padding = 2, name = 'Pad_init')(input_layer)
        x = BatchNormalization(name = 'BN_init')(x)
        x = Activation("relu", name = 'ReLU_init')(x)
        x = Conv2D(self.growth_rate*2, (7, 7), strides = 2, kernel_initializer = RandomNormal(0, 0.0001), name = 'Conv_init')(x)
        x = Dropout(self.drop_conv)(x)
        x = MaxPooling2D(pool_size = (3, 3), strides = 2, padding="same", name = 'MaxPool_init')(x)

        for i, reps in enumerate(self.dense_blocks):
            # Dense block
            x = self.dense_block(x, reps)

            # Transition layer
            if i != len(self.dense_blocks)-1:
                x = BatchNormalization(name = 'BN_trans_%d' % i)(x)
                x = Activation("relu", name = 'ReLU_trans_%d' % i)(x)
                x = Conv2D(int(K.int_shape(x)[3] * self.theta), (1, 1), strides = 1, kernel_initializer = RandomNormal(0, 0.0001), name = 'Conv_trans_%d' % i)(x)
                x = Dropout(self.drop_conv)(x)
                x = AveragePooling2D(pool_size = (2, 2), strides = 2, padding="same", name = 'AvgPool_trans_%d' % i)(x)
                self.nconvs += 1

        shape_x = K.int_shape(x)
        x = AveragePooling2D(pool_size = (shape_x[1], shape_x[2]), strides = (shape_x[1], shape_x[2]), padding="same", name = 'AvgPool_final')(x)
        print(x)

        x = Flatten()(x)
        x = Dense(units = 2, activation = 'softmax')(x)

        self.nconvs += 2
        self.classifier = Model(inputs = input_layer, outputs = x)
        self.classifier.summary()
        print('DenseNet-%d' % self.nconvs)
