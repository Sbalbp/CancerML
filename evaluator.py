from __future__ import division
import re
import sys
import math
if sys.version_info < (3,0):
    from itertools import izip as zip

import numpy as np

def generate_confusion_matrix(real, predicted, n_classes):
    assert len(real) == len(predicted)

    confusion = np.zeros((n_classes, n_classes), dtype = np.int16)

    for truth, pred in zip(real, predicted):
        confusion[truth][pred] += 1

    return confusion.tolist()

def get_accuracy(confusion):
    conf = np.asarray(confusion)

    return np.sum(np.diagonal(conf)) / np.sum(conf)


def evaluate_images_all(classifier, test_set, by_patient = False):

    test_steps = test_set.samples / test_set.batch_size
    evaluations = classifier.predict_generator(test_set)
    n_classes = len(test_set.class_indices)

    img_classes = test_set.classes
    img_preds = np.argmax(evaluations, 1)

    result = []
    result.append({'name': 'overall', 'confusion': generate_confusion_matrix(img_classes, img_preds, n_classes)})

    return result

def evaluate_images_generator(classifier, generator, gen_data, batch_size, by_patient = False):
    evaluations = classifier.predict_generator(generator, math.ceil(len(gen_data[0])/batch_size))
    n_classes = 2

    whole_img_stats = {}
    # Store the class and evaluation sum for each set of patches belonging to the same image
    for i, nfile in enumerate(gen_data[0]):
        #route_split = nfile['img'].split('/')
        route_split = re.split('[\\\/]', nfile['img'])
        img_name = '%s/%s' % (route_split[-2], route_split[-1][:-4])
        if not img_name in whole_img_stats.keys():
            whole_img_stats[img_name] = {}
            whole_img_stats[img_name]['class'] = nfile['class']
            whole_img_stats[img_name]['eval'] = evaluations[i]
        else:
            whole_img_stats[img_name]['class'] += nfile['class']
            whole_img_stats[img_name]['eval'] = whole_img_stats[img_name]['eval'] + evaluations[i]

    whole_imgs = list(whole_img_stats.keys())

    if by_patient:
        # Parse a list of patients associated to each prediction
        patient_list = whole_imgs
        regexp = re.compile('^(benign|malignant)(/|\\\)SOB_[B|M]_\w+(_|-)\d+-(.*)-(40|100|200|400)-\d+_rescale_.*')
        patients = [regexp.match(f).group(4) for f in patient_list]

        # Get the list of unique patients
        unique_patients = list(set(patients))
        is_patient = np.vectorize(lambda x, y: y == x)

    img_classes = np.asarray([whole_img_stats[img]['class'] if whole_img_stats[img]['class'] == 0 else 1 for img in whole_imgs])
    img_preds = np.asarray([np.argmax(whole_img_stats[img]['eval']) for img in whole_imgs])

    correct = img_classes == img_preds
    incorrect = np.logical_not(correct)

    result = {'single': [], 'matrix': []}
    """
    tn = np.sum(np.logical_and(img_classes == 0, correct))
    tp = np.sum(np.logical_and(img_classes == 1, correct))
    fn = np.sum(np.logical_and(img_classes == 1, incorrect))
    fp = np.sum(np.logical_and(img_classes == 0, incorrect))
    result.append({'name': 'overall', 'confusion': [[tn,fp],[fn,tp]]})
    """

    result['matrix'].append({'name': 'overall', 'confusion': generate_confusion_matrix(img_classes, img_preds, n_classes)})
    result['single'].append({'name': 'image score', 'value': get_accuracy(result['matrix'][-1]['confusion'])})

    if by_patient:
        patient_scores = np.zeros((0))
        for patient in unique_patients:
            patient_records = is_patient(patients, patient)
            """
            tn = np.sum(np.logical_and(patient_records, np.logical_and(img_classes == 0, correct)))
            tp = np.sum(np.logical_and(patient_records, np.logical_and(img_classes == 1, correct)))
            fn = np.sum(np.logical_and(patient_records, np.logical_and(img_classes == 1, incorrect)))
            fp = np.sum(np.logical_and(patient_records, np.logical_and(img_classes == 0, incorrect)))
            result.append({'name': patient, 'confusion': [[tn,fp],[fn,tp]]})
            """
            result['matrix'].append({'name': patient, 'confusion': generate_confusion_matrix(img_classes[patient_records], img_preds[patient_records], n_classes)})

            patient_scores = np.append(patient_scores, get_accuracy(result['matrix'][-1]['confusion']))

        patient_score = np.sum(patient_scores) / patient_scores.shape[0]
        result['single'].append({'name': 'patient score', 'value': patient_score})

    return result

def evaluate_images(classifier, test_set, by_patient = False):

    test_steps = test_set.samples / test_set.batch_size
    evaluations = classifier.predict_generator(test_set)
    n_classes = len(test_set.class_indices)

    whole_img_stats = {}
    # Store the class and evaluation sum for each set of patches belonging to the same image
    for i, fname in enumerate(test_set.filenames):
        name_regex = re.compile('(.*)\d{5}\.png$')
        img_name = name_regex.match(fname).group(1)
        if not img_name in whole_img_stats.keys():
            whole_img_stats[img_name] = {}
            whole_img_stats[img_name]['class'] = test_set.classes[i]
            whole_img_stats[img_name]['eval'] = evaluations[i]
        else:
            whole_img_stats[img_name]['class'] += test_set.classes[i]
            whole_img_stats[img_name]['eval'] = whole_img_stats[img_name]['eval'] + evaluations[i]

    #print('key length')
    #print(len(whole_img_stats.keys()))

    whole_imgs = list(whole_img_stats.keys())

    if by_patient:
        # Parse a list of patients associated to each prediction
        patient_list = whole_imgs
        regexp = re.compile('^(benign|malignant)(/|\\\)SOB_[B|M]_\w+(_|-)\d+-(.*)-(40|100|200|400)-\d+_rescale_.*')
        patients = [regexp.match(f).group(4) for f in patient_list]

        # Get the list of unique patients
        unique_patients = list(set(patients))
        is_patient = np.vectorize(lambda x, y: y == x)

    img_classes = np.asarray([whole_img_stats[img]['class'] if whole_img_stats[img]['class'] == 0 else 1 for img in whole_imgs])
    img_preds = np.asarray([np.argmax(whole_img_stats[img]['eval']) for img in whole_imgs])

    correct = img_classes == img_preds
    incorrect = np.logical_not(correct)

    result = {'single': [], 'matrix': []}
    """
    tn = np.sum(np.logical_and(img_classes == 0, correct))
    tp = np.sum(np.logical_and(img_classes == 1, correct))
    fn = np.sum(np.logical_and(img_classes == 1, incorrect))
    fp = np.sum(np.logical_and(img_classes == 0, incorrect))
    result.append({'name': 'overall', 'confusion': [[tn,fp],[fn,tp]]})
    """

    result['matrix'].append({'name': 'overall', 'confusion': generate_confusion_matrix(img_classes, img_preds, n_classes)})
    result['single'].append({'name': 'image score', 'value': get_accuracy(result['matrix'][-1]['confusion'])})

    if by_patient:
        patient_scores = np.zeros((0))
        for patient in unique_patients:
            patient_records = is_patient(patients, patient)
            """
            tn = np.sum(np.logical_and(patient_records, np.logical_and(img_classes == 0, correct)))
            tp = np.sum(np.logical_and(patient_records, np.logical_and(img_classes == 1, correct)))
            fn = np.sum(np.logical_and(patient_records, np.logical_and(img_classes == 1, incorrect)))
            fp = np.sum(np.logical_and(patient_records, np.logical_and(img_classes == 0, incorrect)))
            result.append({'name': patient, 'confusion': [[tn,fp],[fn,tp]]})
            """
            result['matrix'].append({'name': patient, 'confusion': generate_confusion_matrix(img_classes[patient_records], img_preds[patient_records], n_classes)})

            patient_scores = np.append(patient_scores, get_accuracy(result['matrix'][-1]['confusion']))

        patient_score = np.sum(patient_scores) / patient_scores.shape[0]
        result['single'].append({'name': 'patient score', 'value': patient_score})

    return result


class CancerEvaluator(object):

    def __init__(self, classifier, batch_size, img_size):
        self.classifier = classifier
        self.batch_size = batch_size
        self.img_size = img_size
        self.patches_per_img = (350 // (img_size / 2) - 1) * (230 // (img_size / 2) - 1)
        self.grid_patches_per_img = (350 // img_size) * (230 // img_size)

    def __pred_by_sum(self, evaluation):
        prediction = np.argmax(np.sum(evaluation.reshape(-1, self.patches_per_img, 2), 1), 1)
        return prediction

    def __pred_by_majority(self, evaluation):
        prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, minlength = 2)),
                                            axis = 1,
                                            arr = evaluation.argsort()[:,1].reshape(-1, self.patches_per_img))
        return prediction

    def  evaluate_images(self, test_set, by_patient = False):
        # First evaluate the model on the test set
        """
        test_datagen = ImageDataGenerator(preprocessing_function = lambda x: x)

        test_set = test_datagen.flow_from_directory(test,
                                                    target_size = (self.img_size, self.img_size),
                                                    batch_size = self.batch_size,
                                                    class_mode = 'categorical',
                                                    shuffle = False)
        """

        test_steps = test_set.samples / test_set.batch_size
        evaluations = self.classifier.predict_generator(test_set, test_steps)

        self.patches_per_img = 2

        """
        IDEA: to make sure the images are in order (if we dont trust Keras):
        iterate test_set.filenames, evaluations and test_set.classes at once
        regexp the original file name (eg, anything before .png,)
        add an entry to dictionary with the parsed filename if it doesnt exist
        append both the corresponding evaluation and test_set.class to 2 lists inside the dictionary for that key
        When the structure is completed, iterate through its keys (every single 350x230 image) and calculate the pred_by_sum on the evaluations, getting the pred
        the new img_classes will be an array with the first element os test_set.classes of each filename
        the new img_preds shall be an array with the pred just calculated
        BAM
        """


        whole_img_stats = {}
        # Store the class and evaluation sum for each set of patches belonging to the same image
        for i, fname in enumerate(test_set.filenames):
            name_regex = re.compile('(.*)\d{5}\.png$')
            img_name = name_regex.match(fname).group(1)
            if not img_name in whole_img_stats.keys():
                whole_img_stats[img_name] = {}
                whole_img_stats[img_name]['class'] = test_set.classes[i]
                whole_img_stats[img_name]['eval'] = evaluations[i]
            else:
                whole_img_stats[img_name]['class'] += test_set.classes[i]
                whole_img_stats[img_name]['eval'] = whole_img_stats[img_name]['eval'] + evaluations[i]

        print('key length')
        print(len(whole_img_stats.keys()))

        whole_imgs = list(whole_img_stats.keys())


        if by_patient:
            # Parse a list of patients associated to each prediction
            patient_list = whole_imgs
            regexp = re.compile('^(benign|malignant)(/|\\\)SOB_[B|M]_\w+(_|-)\d+-(.*)-(40|100|200|400)-\d+_rescale_.*')
            patients = [regexp.match(f).group(4) for f in patient_list]

            #print(patients)

            # Get the list of unique patients
            unique_patients = list(set(patients))
            is_patient = np.vectorize(lambda x, y: y == x)

        img_classes = np.asarray([whole_img_stats[img]['class'] if whole_img_stats[img]['class'] == 0 else 1 for img in whole_imgs])
        img_preds = np.asarray([np.argmax(whole_img_stats[img]['eval']) for img in whole_imgs])

        #print(img_classes)
        #print(img_preds)


        """
        IDEA IMPLEMENTED ABOVE
        """

        """

        if by_patient:
            # Parse a list of patients associated to each prediction
            patient_list = test_set.filenames
            regexp = re.compile('^(benign|malignant)(/|\\\)SOB_[B|M]_\w+(_|-)\d+-(.*)-(40|100|200|400)-\d+_rescale_.*')
            patients = [regexp.match(f).group(4) for f in patient_list]
            patients = np.asarray(patients).reshape(-1, self.patches_per_img)[:,0]

            # Get the list of unique patients
            unique_patients = list(set(patients))
            is_patient = np.vectorize(lambda x, y: y == x)

        # Since we are dealing with patches, group them to get the final prediction for the whole image
        img_classes = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, minlength = 2)), axis = 1, arr = test_set.classes.reshape(-1, self.patches_per_img))
        img_preds = self.__pred_by_sum(evaluations)

        """

        correct = img_classes == img_preds
        incorrect = np.logical_not(correct)

        result = {'overall': {}}

        tn = np.sum(np.logical_and(img_classes == 0, correct))
        tp = np.sum(np.logical_and(img_classes == 1, correct))
        fn = np.sum(np.logical_and(img_classes == 1, incorrect))
        fp = np.sum(np.logical_and(img_classes == 0, incorrect))
        result['overall'] = [[tn,fp],[fn,tp]]

        if by_patient:
            result['patients'] = {}
            for patient in unique_patients:
                result['patients'][patient] = {}
                patient_records = is_patient(patients, patient)
                tn = np.sum(np.logical_and(patient_records, np.logical_and(img_classes == 0, correct)))
                tp = np.sum(np.logical_and(patient_records, np.logical_and(img_classes == 1, correct)))
                fn = np.sum(np.logical_and(patient_records, np.logical_and(img_classes == 1, incorrect)))
                fp = np.sum(np.logical_and(patient_records, np.logical_and(img_classes == 0, incorrect)))
                result['patients'][patient] = [[tn,fp],[fn,tp]]

        return result