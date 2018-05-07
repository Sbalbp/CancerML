from __future__ import division
import os
import re
import sys
import math
import errno
from six.moves.configparser import ConfigParser
from multiprocessing import Process, Pipe

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import cv2
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator

sys.stderr = stderr

import preprocessing
import evaluator
import pretty_print
from alexnet import AlexNet
from resnet import ResNet
from densenet import DenseNet

CONFIG_DEFAULTS = {
    'ImgSize': '-1',
    'InitialDim': '-1,-1',
    'EndDim': '-1,-1',
    'ImageDim': '-1,-1',
    'Undersample': '1',
    'Oversample': '1'
}

NETWORK = {
    'alexnet': AlexNet,
    'resnet': ResNet,
    'densenet': DenseNet
}

def makedirs_functions(name):
    def makedirs_py2(dirname):
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    def makedirs_py3(dirname):
        os.makedirs(dirname, exist_ok = True)

    return makedirs_py3(name) if sys.version_info >= (3,2) else makedirs_py2(name)

makedirs_exist_ok = makedirs_functions

class LossHistory(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    """
    def on_train_begin(self, logs={}):
        self.losses = []
    """

    def on_epoch_end(self, epoch, logs={}):
        #print(type(self.test_data))
        #print(logs.keys())
        result = evaluator.evaluate_images(self.model, self.test_data, by_patient = True)
        for entry in result['single']:
            print('%s: %s' % (entry['name'], ('%s' % entry['value']).rstrip('0').rstrip('.')))
        #self.losses.append(logs.get('loss'))

def generate_patches(pipe_conn, gen_args, patch_type, dims, qty = None, undersample = 1, oversample = 1):
    sys.stdout = open(os.devnull, 'w')
    train_datagen = ImageDataGenerator()#preprocessing_function = lambda x: x - gen_args[0])
    datagen = train_datagen.flow_from_directory(gen_args[1],
                                                target_size = (gen_args[2][0], gen_args[2][1]),
                                                batch_size = gen_args[3],
                                                class_mode = 'categorical',
                                                shuffle = False)
    sys.stdout = sys.__stdout__

    # Which is the mayority class?
    class_qty = {}
    mayority = -1
    mayority_class = None
    for class_name in datagen.class_indices:
        index = datagen.class_indices[class_name]
        class_qty[index] = np.where(datagen.classes == index)[0].shape[0]
        if class_qty[index] > mayority:
            mayority = class_qty[index]
            mayority_class = index

    # If undersample/oversample is negative we will balance
    if undersample < 0:
        undersample = class_qty[1-mayority_class] / class_qty[mayority_class]

    if oversample < 0:
        oversample = class_qty[mayority_class] / class_qty[1-mayority_class]

    steps = datagen.n / datagen.batch_size
    patches = []

    # For each batch
    for i, batch in enumerate(datagen):
        # Get the class of each image in the batch
        truth_class = np.argmax(batch[1], 1)

        # For each image in the batch
        for j, img in enumerate(batch[0]):
            # Generate the coordinates for qty random patches in the image
            if patch_type == 'random':
                quantity = int(qty * undersample) if truth_class[j] == mayority_class else int(qty * oversample)
                coord = preprocessing.split_random_patches(img, dims[0], dims[1], quantity)
            elif patch_type == 'window':
                coord = preprocessing.split_window_patches(img, dims[0], dims[1])
            elif patch_type == 'grid':
                coord = preprocessing.split_grid_patches(img, dims[0], dims[1])
            else:
                raise Exception('Invalid type of patch selected: %s' % patch_type)
            patches += [{ 'c': coord[1][patch], 'r': coord[0][patch], 'img': '%s/%s' % (gen_args[1], datagen.filenames[i*datagen.batch_size+j]), 'img_index': i*datagen.batch_size+j, 'class': truth_class[j] } for patch in range(len(coord[0]))]
        if i >= steps-1:
            break

    print('Samples: %d' % len(patches))
    filenames = ['%s/%s' % (gen_args[1], filename) for filename in datagen.filenames]
    pipe_conn.send([patches, filenames])
    pipe_conn.close()
    return #patches

class CancerTrainer(object):
    def __init__(self, config_route, train_route = None, test_route = None, avg_img_route = None):
        self.__load_config(config_route)
        if (self.training or train_route) and (self.test or test_route):
            self.set_dataset(train_route, test_route, avg_img_route)

    def create(self, r):
        makedirs(r)

    def __infer_img_size(self):
        if self.img_size == -1:
            regex = re.compile('.*_(\d+)x\d+/.*')
            match = regex.match(self.training)
            if match:
                self.img_size = int(match.group(1))

        if self.net_dim == [-1,-1]:
            regex = re.compile('.*_(\d+)x(\d+)/.*')
            match = regex.match(self.training)
            if match:
                self.net_dim = [int(match.group(1)), int(match.group(2))]

        if self.initial_dim == [-1,-1]:
            self.initial_dim = self.net_dim

    def __load_config(self, config_route):
        # Read parameters from config file
        config = ConfigParser(CONFIG_DEFAULTS)
        config.read(config_route)

        self.training = config.get('training', 'TrainingSet')
        self.test = config.get('training', 'TestSet')
        avg_img_route = config.get('training', 'AverageImage')
        self.epochs = config.getint('training', 'Epochs')
        self.batch_size = config.getint('training', 'BatchSize')
        self.img_size = config.getint('training', 'ImgSize')
        self.net_dim = [int(d) for d in config.get('training', 'ImageDim').split(',')]
        self.preprocessing = config.get('preprocessing', 'Type')
        if self.preprocessing == 'random' or self.preprocessing == 'window':
            self.initial_dim = [int(d) for d in config.get('preprocessing', 'InitialDim').split(',')]
            self.net_dim = [int(d) for d in config.get('preprocessing', 'FinalDim').split(',')]
            self.random_patches = config.getint('preprocessing', 'RandomPatches')
            self.undersample = config.getfloat('preprocessing', 'Undersample')
            self.oversample = config.getfloat('preprocessing', 'Oversample')
        else:
            self.initial_dim = self.net_dim
        self.__infer_img_size()
        self.validate = config.getboolean('training', 'Validate')

        self.avg_img = cv2.imread(avg_img_route)

        self.net_type = config.get('network', 'Network')
        self.lr = config.getfloat('network', 'LearningRate')
        self.momentum = config.getfloat('network', 'Momentum')
        self.decay_rate = config.getfloat('network', 'DecayRate')

        self.network_params = dict(config.items('network specific'))
        #print(self.network_params)
        #self.drop_conv = config.getfloat('network specific', 'drop_conv')
        #self.drop_dense = config.getfloat('network specific', 'drop_dense')

        self.show_patients = config.getboolean('output', 'ShowPatients')

    def set_dataset(self, train_route = None, test_route = None, avg_img_route = None):
        if not train_route and not self.training:
            raise Exception('Training set path is missing')
        if not test_route and not self.test:
            raise Exception('Test set path is missing')

        if train_route:
            self.training = train_route
        if test_route:
            self.test = test_route

        # Try to infer the size from the training set
        self.__infer_img_size()

        if avg_img_route:
            self.avg_img = cv2.imread(avg_img_route)

        # Training set
        train_datagen = ImageDataGenerator(preprocessing_function = lambda x: x - self.avg_img)
        self.training_set = train_datagen.flow_from_directory(self.training,
                                                            target_size = (self.initial_dim[0], self.initial_dim[1]),
                                                            batch_size = self.batch_size,
                                                            class_mode = 'categorical',
                                                            shuffle = not self.preprocessing)
        self.training_steps = self.training_set.samples / self.training_set.batch_size

        # Test set
        test_datagen = ImageDataGenerator(preprocessing_function = lambda x: x - self.avg_img)
        self.test_set = test_datagen.flow_from_directory(self.test,
                                                        target_size = (self.net_dim[0], self.net_dim[1]),
                                                        batch_size = self.batch_size,
                                                        class_mode = 'categorical',
                                                        shuffle = False)
        self.test_steps = self.test_set.samples / self.test_set.batch_size

    def set_network(self, model_file = None, weights_file = None):
        # Network
        if model_file and weights_file:
            self.network = NETWORK[self.net_type](model_json = model_file, weights_h5 = weights_file)
        else:
            self.network = NETWORK[self.net_type](insize = self.net_dim, **self.network_params)
        self.network.classifier.compile(optimizer = SGD(lr = self.lr, momentum = self.momentum, decay = self.decay_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    def patches_generator(self, patches, filenames, batch_size, patch_size, shuffle = True, preload_imgs = True):
        order = np.random.permutation(len(patches))

        if preload_imgs:
            images = np.array([cv2.imread(filename).astype(np.float64) for filename in filenames])#- self.avg_img for filename in filenames])

            print('images', images.mean(), images.dtype)
            #avg = np.sum(images / images.shape[0], 0)
            #images -= avg
            print('images averaged', images.mean(), images.dtype)

        sampled = 0
        while(True):
            if shuffle:
                indexes = order[sampled:min(sampled+batch_size, len(patches))]
            else:
                indexes = range(sampled, min(sampled+batch_size, len(patches)))

            if not preload_imgs:
                images = np.array([cv2.imread(patches[index]['img']) for index in indexes])

                samples = np.array([images[i][patches[index]['r']:patches[index]['r']+patch_size[1], patches[index]['c']:patches[index]['c']+patch_size[0]] for i, index in enumerate(indexes)])
            else:
                samples = np.array([images[patches[index]['img_index']][patches[index]['r']:patches[index]['r']+patch_size[1], patches[index]['c']:patches[index]['c']+patch_size[0]] for i,index in enumerate(indexes)])
            
            labels = np.array([ [1, 0] if patches[index]['class'] == 0 else [0, 1] for index in indexes])

            sampled += batch_size
            if sampled >= len(patches):
                sampled = 0

            yield samples, labels


    def train(self, log_file = None, chkp_model_file = None, chkp_weights_file = None):
        best_on_test = 0

        if log_file:
            makedirs_exist_ok(os.path.dirname(log_file))
            f = open(log_file, 'a')
        log = ''

        if self.preprocessing == 'random' or self.preprocessing == 'window':
            parent_conn, child_conn = Pipe()

            for epoch in range(self.epochs):
                print('EPOCH %d' % (epoch+1))

                # Generate batch for first epoch
                if epoch == 0:
                    process = Process(target = generate_patches, args = (child_conn, [self.avg_img, self.training, self.initial_dim, self.batch_size], self.preprocessing, (self.net_dim[0], self.net_dim[1]), self.random_patches, self.undersample, self.oversample))
                    process.start()

                # Gather batch data computed in parallel
                if epoch == 0 or self.preprocessing == 'random':
                    epoch_data = parent_conn.recv()
                    process.join()

                # Start generating batch for next epoch
                if epoch < self.epochs-1 and self.preprocessing == 'random':
                    process = Process(target = generate_patches, args = (child_conn, [self.avg_img, self.training, self.initial_dim, self.batch_size], self.preprocessing, (self.net_dim[0], self.net_dim[1]), self.random_patches, self.undersample, self.oversample))
                    process.start()

                generator = self.patches_generator(epoch_data[0], epoch_data[1], self.batch_size, (self.net_dim[0],self.net_dim[1]))
                steps = len(epoch_data[0]) / self.batch_size

                self.network.classifier.fit_generator(generator, steps_per_epoch = steps, epochs = 1)
                # for batch:
                    # Generate patches for the batch
                    # Train the batch



            

                parent_conn3, child_conn3 = Pipe()
                process = Process(target = generate_patches, args = (child_conn3, [self.avg_img, self.training, self.initial_dim, self.batch_size], 'grid', (self.net_dim[0], self.net_dim[1]), self.random_patches))
                process.start()
                train_set = parent_conn3.recv()
                process.join()
                val_generator = self.patches_generator(train_set[0], train_set[1], self.batch_size, (self.net_dim[0],self.net_dim[1]), False)
                result = evaluator.evaluate_images_generator(self.network.classifier, val_generator, train_set, self.batch_size, by_patient = True)
                for entry in result['single']:
                    print('%s: %s' % (entry['name'], ('%s' % entry['value']).rstrip('0').rstrip('.')))
                    log = '%s%.4f,' % (log, entry['value'])
                    
                if self.validate:
                    accuracy_acc = 0

                    parent_conn2, child_conn2 = Pipe()
                    process = Process(target = generate_patches, args = (child_conn2, [self.avg_img, self.test, self.initial_dim, self.batch_size], 'grid', (self.net_dim[0], self.net_dim[1]), self.random_patches))
                    process.start()
                    test_set = parent_conn2.recv()
                    process.join()
                    val_generator = self.patches_generator(test_set[0], test_set[1], self.batch_size, (self.net_dim[0],self.net_dim[1]), False)
                    result = evaluator.evaluate_images_generator(self.network.classifier, val_generator, test_set, self.batch_size, by_patient = True)
                    for entry in result['single']:
                        print('%s: %s' % (entry['name'], ('%s' % entry['value']).rstrip('0').rstrip('.')))
                        log = '%s%.4f,' % (log, entry['value'])

                        # Check for improvements in test
                        accuracy_acc += entry['value']
                    if(accuracy_acc > best_on_test):
                        print('NEW BEST FOUND ON TEST!')
                        best_on_test = accuracy_acc

                        if chkp_model_file and chkp_weights_file:
                            self.save_model(chkp_model_file, chkp_weights_file)

                log += '\n'

            if log_file:
                orig_stdout = sys.stdout
                sys.stdout = f
                print(log)
                sys.stdout = orig_stdout
                f.close()


        else:
            history = LossHistory(self.test_set)
            self.network.classifier.fit_generator(self.training_set,
                                        steps_per_epoch = self.training_steps,
                                        epochs = self.epochs,
                                        validation_data = None if not self.validate else self.test_set,
                                        validation_steps = None if not self.validate else self.test_steps, verbose = 2,
                                        callbacks = [history])#, validation_data = test_set, validation_steps = test_steps)

    def evaluate(self, filename = None):
        if self.preprocessing == 'random' or self.preprocessing == 'window':
            parent_conn, child_conn = Pipe()
            process = Process(target = generate_patches, args = (child_conn, [self.avg_img, self.training, self.initial_dim, self.batch_size], 'grid', (self.net_dim[0], self.net_dim[1]), self.random_patches))
            process.start()
            training_set = parent_conn.recv()
            process.join()
            process = Process(target = generate_patches, args = (child_conn, [self.avg_img, self.test, self.initial_dim, self.batch_size], 'grid', (self.net_dim[0], self.net_dim[1]), self.random_patches))
            process.start()
            test_set = parent_conn.recv()
            process.join()
        else:
            # Create a data generator for the training set that is not shuffled
            train_datagen = ImageDataGenerator(preprocessing_function = lambda x: x - self.avg_img)
            training_set = train_datagen.flow_from_directory(self.training,
                                                            target_size = (self.net_dim[0], self.net_dim[1]),
                                                            batch_size = self.batch_size,
                                                            class_mode = 'categorical',
                                                            shuffle = False)
            test_set = self.test_set

        self.evaluate_set(training_set, filename)
        self.evaluate_set(test_set, filename, append = True)

    def evaluate_set(self, dataset, filename = None, append = False):
        if self.preprocessing == 'random' or self.preprocessing == 'window':
            generator = self.patches_generator(dataset[0], dataset[1], self.batch_size, (self.net_dim[0],self.net_dim[1]), False)
            result = evaluator.evaluate_images_generator(self.network.classifier, generator, dataset, self.batch_size, by_patient = True)
        else:
            result = evaluator.evaluate_images(self.network.classifier, dataset, by_patient = True)

        if filename:
            makedirs_exist_ok(os.path.dirname(filename))

            orig_stdout = sys.stdout
            f = open(filename, 'a' if append else 'w')
            sys.stdout = f

            print('-' * 100)
            print(filename.center(100, '-'))
            print('-' * 100 + '\n')

        # print overall stats
        for entry in result['single']:
            pretty_print.print_variable(entry['name'], entry['value'])

        # print confusion matrices
        print('\n\n ***** %s *****\n\n' % result['matrix'][0]['name'])
        pretty_print.print_stats(['Benign', 'Malignant'], result['matrix'][0]['confusion'])

        if self.show_patients and len(result) > 1:
            for patient in result['matrix'][1:]:
                print('\n\n ***** Patient %s *****\n\n' % patient['name'])
                pretty_print.print_stats(['Benign', 'Malignant'], patient['confusion'])

        if filename:
            sys.stdout = orig_stdout
            f.close()


    def save_model(self, model_file = 'model.json', weights_file = 'weights.h5'):
        # serialize model to JSON
        model_json = self.network.classifier.to_json()

        makedirs_exist_ok(os.path.dirname(model_file))
        with open(model_file, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        makedirs_exist_ok(os.path.dirname(weights_file))
        self.network.classifier.save_weights(weights_file)
        print("Saved model to disk")

    """
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=0)
    """

    """
    result = evaluator.evaluate_images(network.classifier, test_set, by_patient=True)

    print('\n\n ***** Overall *****\n\n')
    #(tn, tp, fn, fp) = (result['overall']['tn'], result['overall']['tp'], result['overall']['fn'], result['overall']['fp'])
    print_stats(['Benign', 'Malignant'],result['overall'])

    if show_patients and 'patients' in result.keys():
        for patient in result['patients']:
            print('\n\n ***** Patient %s *****\n\n' % patient)
            #(tn, tp, fn, fp) = (result['patients'][patient]['tn'], result['patients'][patient]['tp'], result['patients'][patient]['fn'], result['patients'][patient]['fp'])
            print_stats(['Benign', 'Malignant'],result['patients'][patient])
    """
