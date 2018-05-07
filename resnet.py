from cnn import *
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Input
from keras.layers.merge import add
from keras import Model
from keras import backend as K
from keras.regularizers import l2

class ResNet(Network):

    def residual_basic(self, x):
        prev = x

        


        x = Conv2D(64, (3, 3), strides = 2, kernel_initializer = RandomNormal(0, 0.0001))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(64, (3, 3), strides = 2, kernel_initializer = RandomNormal(0, 0.0001))(x)
        #x = MaxPooling2D(pool_size = (3, 3), strides = 2, padding="same")(x)

        input_shape = K.int_shape(prev)
        residual_shape = K.int_shape(x)
        stride_width = int(round(input_shape[2] / residual_shape[2]))
        stride_height = int(round(input_shape[1] / residual_shape[1]))
        equal_channels = input_shape[3] == residual_shape[3]

        print(stride_width)
        print(stride_height)

        if stride_width > 1 or stride_height > 1 or not equal_channels:
            prev = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001),
                          name='residual_truncate')(prev)

        x = add([prev,x])

        return x

    def shortcut_conn(self, identity, residual):
        input_shape = K.int_shape(identity)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[2] / residual_shape[2]))
        stride_height = int(round(input_shape[1] / residual_shape[1]))
        equal_channels = input_shape[3] == residual_shape[3]

        if stride_width > 1 or stride_height > 1 or not equal_channels:
            identity = Conv2D(filters=residual_shape[3],
                        kernel_size=(1, 1),
                        strides=(stride_width, stride_height),
                        padding="valid",
                        kernel_initializer="he_normal",
                        kernel_regularizer=l2(0.0001))(identity)

        return add([identity, residual])

    def residual_block(self, input_t, reps, nfilters = 64, first = False):
        x = input_t
        for i, rep in enumerate(range(reps)):
            init_stride = 1 if i != 0 or first else 2
            if not first or i != 0:
                x = BatchNormalization(name = 'BN_A_%d_%d' % (nfilters, i))(x)
                x = Activation("relu", name = 'ReLU_A_%d_%d' % (nfilters, i))(x)
            x = Conv2D(nfilters, (3, 3), strides = init_stride, padding = "same", kernel_initializer = RandomNormal(0, 0.0001), name = 'Conv1_%d_%d' % (nfilters, i))(x)
            x = Dropout(self.drop_conv)(x)
            x = BatchNormalization(name = 'BN_B_%d_%d' % (nfilters, i))(x)
            x = Activation("relu", name = 'ReLU_B_%d_%d' % (nfilters, i))(x)
            x = Conv2D(nfilters, (3, 3), strides = 1, padding = "same", kernel_initializer = RandomNormal(0, 0.0001), name = 'Conv2_%d_%d' % (nfilters, i))(x)
            x = Dropout(self.drop_conv)(x)
            x = self.shortcut_conn(input_t, x)

        return x

    def build_architecture(self, insize, **kwargs):


        try:
            self.drop_conv = float(kwargs['drop_conv'])
            self.drop_dense = float(kwargs['drop_dense'])

            repetitions = [int(rep) for rep in kwargs['repetitions'].split(',')]
            print(repetitions)
        except KeyError as e:
            raise(TypeError('build_architecture: Missing argument: %s' % e))

        input_layer = Input(shape = (insize[0], insize[1], 3), name = 'ResNet_Input')
        #x = input_layer
        x = ZeroPadding2D(padding = 2, name = 'Pad_init')(input_layer)
        x = Conv2D(64, (7, 7), strides = 2, kernel_initializer = RandomNormal(0, 0.0001), name = 'Conv_init')(x)
        x = Dropout(self.drop_conv)(x)
        x = BatchNormalization(name = 'BN_init')(x)
        x = Activation("relu", name = 'ReLU_init')(x)
        x = MaxPooling2D(pool_size = (3, 3), strides = 2, padding="same", name = 'MaxPool_init')(x)

        nfilters = 64
        for i, rep in enumerate(repetitions):
            x = self.residual_block(x, rep, nfilters = nfilters, first = i==0)
            nfilters *= 2

        #x = self.residual_basic(x)

        x = Flatten()(x)
        x = Dense(units = 64, activation = 'relu')(x)
        x = Dropout(self.drop_dense)(x)
        #x = Dropout(drop_dense)(x)
        x = Dense(units = 2, activation = 'softmax')(x)
        #x
        #conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        #pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)


        """
        self.classifier = Sequential()
        self.classifier.add(ZeroPadding2D(input_shape = (insize, insize, 3), padding = 2))
        self.classifier.add(Conv2D(64, (7, 7), strides = 2, kernel_initializer = RandomNormal(0, 0.0001)))
        self.classifier.add(BatchNormalization())
        self.classifier.add(Activation("relu"))
        self.classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2, padding="same"))





        self.classifier.add(Flatten())

        self.classifier.add(Dense(units = 64, activation = 'relu'))
        self.classifier.add(Dropout(drop_dense))
        self.classifier.add(Dense(units = 2, activation = 'softmax'))
        """

        self.classifier = Model(inputs = input_layer, outputs = x)
        self.classifier.summary()
        #conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        #pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)


        """
        self.classifier.add(Conv2D(32, (5, 5), kernel_initializer = RandomNormal(0, 0.0001), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))
        self.classifier.add(Dropout(drop_conv))

        self.classifier.add(ZeroPadding2D(padding = 2))
        self.classifier.add(Conv2D(32, (5, 5), kernel_initializer = RandomNormal(0, 0.01), activation = 'relu'))
        self.classifier.add(AveragePooling2D(pool_size = (3, 3), strides = 2))
        self.classifier.add(Dropout(drop_conv))

        self.classifier.add(ZeroPadding2D(padding = 2))
        self.classifier.add(Conv2D(64, (5, 5), kernel_initializer = RandomNormal(0, 0.0001), activation = 'relu'))
        self.classifier.add(AveragePooling2D(pool_size = (3, 3), strides = 2))
        self.classifier.add(Dropout(drop_conv))

        self.classifier.add(Flatten())

        self.classifier.add(Dense(units = 64, activation = 'relu'))
        self.classifier.add(Dropout(drop_dense))
        self.classifier.add(Dense(units = 2, activation = 'softmax'))
        """
