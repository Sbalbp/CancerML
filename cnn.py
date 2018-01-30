from __future__ import division
# Restrict implcit exports to keras stuff
#__all__ = ['Sequential']

from keras.models import Sequential
from keras.layers import ZeroPadding2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import model_from_json
from keras.initializers import RandomNormal

class Network(object):

    def __init__(self, insize = None, model_json = None, weights_h5 = None, **kwargs):
        print('Using network: %s' % self.__class__.__name__)

        if model_json:
            json_file = open(model_json, 'r')
            classifier_json = json_file.read()
            json_file.close()
            self.classifier = model_from_json(classifier_json)
        else:
            if insize:
                self.build_architecture(insize, **kwargs)
            else:
                raise(TypeError('At least one is required: insize or model_json'))

        if weights_h5:
            self.classifier.load_weights(weights_h5)

    def build_architecture(self, insize, **kwargs):

        #super(ClassName, self).__init__()
        raise NotImplementedError('Class %s should implement method build_architecture' % type(self).__name__)
        """
        self.classifier = Sequential()

        self.classifier.add(ZeroPadding2D(input_shape = (insize, insize, 3), padding = 2))
        self.classifier.add(Conv2D(32, (5, 5), kernel_initializer = RandomNormal(0, 0.0001), activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))
        #self.classifier.add(Dropout(drop_conv))

        self.classifier.add(ZeroPadding2D(padding = 2))
        self.classifier.add(Conv2D(32, (5, 5), kernel_initializer = RandomNormal(0, 0.01), activation = 'relu'))
        self.classifier.add(AveragePooling2D(pool_size = (3, 3), strides = 2))
        #self.classifier.add(Dropout(drop_conv))

        self.classifier.add(ZeroPadding2D(padding = 2))
        self.classifier.add(Conv2D(64, (5, 5), kernel_initializer = RandomNormal(0, 0.0001), activation = 'relu'))
        self.classifier.add(AveragePooling2D(pool_size = (3, 3), strides = 2))
        #self.classifier.add(Dropout(drop_conv))

        self.classifier.add(Flatten())

        self.classifier.add(Dense(units = 64, activation = 'relu'))
        #self.classifier.add(Dropout(drop_dense))
        self.classifier.add(Dense(units = 2, activation = 'softmax'))
        """






    """
    # Command line parameters
    parser = argparse.ArgumentParser( description = 'Training utilities for the cancer dataset' )
    parser.add_argument( '-t', dest = 'training', metavar = 'Training set', required = True, help = 'Location of the training set' )
    parser.add_argument( '-v', dest = 'test', metavar = 'Test set', required = True, help = 'Location of the test set' )
    parser.add_argument( '-e', dest = 'epochs', metavar = 'Epochs', type = int, default = 1, help = 'Number of epochs' )
    parser.add_argument( '-b', dest = 'batch_size', metavar = 'Batch size', type = int, default = 1, help = 'Size of batch' )
    parser.add_argument( '-s', dest = 'img_size', metavar = 'Image size', type = int, default = 32, help = 'Dimensions of image' )
    parser.add_argument( '-lr', dest = 'lr', metavar = 'Learning rate', type = float, default = 0.0001, help = 'Learning rate of the network' )
    parser.add_argument( '-m', dest = 'momentum', metavar = 'Momentum', type = float, default = 0.9, help = 'Momentum of SGD' )
    parser.add_argument( '-dr', dest = 'decay_rate', metavar = 'Decay rate', type = float, default = 4**(-5), help = 'Dacay rate of SGD' )
    parser.add_argument( '--val', dest = 'validate', action = 'store_true', help = 'Use validation during training' )


    args = parser.parse_args()

    training = args.training
    test = args.test
    epochs = args.epochs
    batch_size = args.batch_size
    img_size = args.img_size
    lr = args.lr
    momentum = args.momentum
    decay_rate = args.decay_rate
    validate = args.validate
    """

    


    

    """
    insize = 32
    classifier = Sequential()

    classifier.add(ZeroPadding2D(input_shape = (insize, insize, 3), padding = 2))
    classifier.add(Conv2D(32, (5, 5), kernel_initializer = RandomNormal(0, 0.0001), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (3, 3), strides = 2))

    classifier.add(ZeroPadding2D(padding = 2))
    classifier.add(Conv2D(32, (5, 5), kernel_initializer = RandomNormal(0, 0.01), activation = 'relu'))
    classifier.add(AveragePooling2D(pool_size = (3, 3), strides = 2))

    classifier.add(ZeroPadding2D(padding = 2))
    classifier.add(Conv2D(64, (5, 5), kernel_initializer = RandomNormal(0, 0.0001), activation = 'relu'))
    classifier.add(AveragePooling2D(pool_size = (3, 3), strides = 2))

    classifier.add(Flatten())

    classifier.add(Dense(units = 64, activation = 'relu'))
    classifier.add(Dense(units = 2, activation = 'softmax'))
    """
    

    #print(np.count_nonzero(preds))
    #a= list(map(lambda x: 0 if 'SOB_B' in x else 1, os.listdir(test+'/benign')))
    #print(a)

    """
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    loaded_model.compile(optimizer = SGD(lr = 10**(-6), momentum = 0.9, decay = 4**(-5)), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    print(loaded_model.evaluate_generator(training_set, 6518/128))
    """

    """
    test_set = test_datagen.flow_from_directory(test,
                                                    target_size = (32, 32),
                                                    batch_size = 54,
                                                    class_mode = 'categorical',
                                                    shuffle = False)
    """


