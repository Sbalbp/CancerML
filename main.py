from six.moves.configparser import ConfigParser

from cancer import CancerTrainer

if __name__ == "__main__":


    # Read the training config
    train_config = ConfigParser()
    train_config.read('train.cfg')

    experiment_name = train_config.get('parameters','Name')
    config_routes = [cfg_file.strip() for cfg_file in train_config.get('parameters','NetworkConfig').split(',')]
    dataset_path = train_config.get('parameters','DatasetPath')
    folds_dir = train_config.get('parameters','FoldsDir')
    averages_dir = train_config.get('parameters','AveragesDir')
    output_dir = train_config.get('parameters','OutputDir')
    folds = [int(fold) for fold in train_config.get('parameters','Folds').split(',')]
    resolutions = [int(resolution) for resolution in train_config.get('parameters','Resolutions').split(',')]
    patch_types = [ptype.strip() for ptype in train_config.get('parameters','PatchTypes').split(',')]
    patch_sizes = [int(psize) for psize in train_config.get('parameters','PatchSizes').split(',')]

    print(config_routes)
    print(folds)
    print(resolutions)
    print(patch_types)
    print(patch_sizes)

    for config_route in config_routes:
        cancer_tr = CancerTrainer(config_route)
        for resolution in resolutions:
            for fold in folds:
                path_str_b = '%dX' % resolution

                path_str_a_train = '/fold%d' % fold
                avg_img = '%s/%s/%s/average_fold%d_train_%s.png' % (dataset_path, averages_dir, folds_dir, fold, path_str_b)

                train_dir = '%s/%s%s/%s/%s' % (dataset_path, folds_dir, path_str_a_train, 'train', path_str_b)
                test_dir = '%s/%s%s/%s/%s' % (dataset_path, folds_dir, path_str_a_train, 'test', path_str_b)

                
                model_dir = '%s/%s/%s%s/%s/model' % (dataset_path, output_dir, folds_dir, path_str_a_train, path_str_b)
                log_dir = '%s/%s/%s%s/%s/log' % (dataset_path, output_dir, folds_dir, path_str_a_train, path_str_b)
                

                
                print(train_dir)
                print(test_dir)
                
                


                cancer_tr.set_dataset(train_dir, test_dir, avg_img)
                cancer_tr.set_network()

                params_str = '%s_%s_unders-%s_overs-%s_lr-%s_ep-%d_bs-%d_%s_%dx%d' % ( experiment_name,
                                        cancer_tr.net_type,
                                        str(1 if not cancer_tr.undersample else cancer_tr.undersample).rstrip('0').rstrip('.'),
                                        str(1 if not cancer_tr.oversample else cancer_tr.oversample).rstrip('0').rstrip('.'),
                                        str(cancer_tr.lr).rstrip('0'),
                                        cancer_tr.epochs,
                                        cancer_tr.batch_size,
                                        'no_preprocessing' if not cancer_tr.preprocessing else cancer_tr.preprocessing,
                                        cancer_tr.net_dim[0],
                                        cancer_tr.net_dim[1])

                results_file = '%s/%s/%s%s/%s/%s_results.txt' % (dataset_path, output_dir, folds_dir, path_str_a_train, path_str_b, params_str)
                json_file = '%s/%s_model.json' % (model_dir, params_str)
                h5_file = '%s/%s_weights.h5' % (model_dir, params_str)
                log_file = '%s/%s_log.csv' % (log_dir, params_str)

                print(results_file)
                print(json_file)
                print(h5_file)
                print(log_file)

                cancer_tr.train(log_file, json_file[:-5]+'_best_test'+json_file[-5:], h5_file[:-3]+'_best_test'+h5_file[-3:])
                cancer_tr.save_model(json_file, h5_file)
                cancer_tr.evaluate(filename = results_file)
                # Evaluate with the best
                if cancer_tr.validate:
                    cancer_tr.set_network(json_file[:-5]+'_best_test'+json_file[-5:], h5_file[:-3]+'_best_test'+h5_file[-3:])
                    cancer_tr.evaluate(filename = (results_file[:-12]+'_best_test'+results_file[-12:]))

            #cancer_tr.set_dataset(eval_train_dir, eval_test_dir, avg_img)

    """
    cancer_tr = CancerTrainer(config_route)
    #cancer_tr.set_dataset('BreaKHis_v1_rescaled_350x230_patch_window_32x32/train','BreaKHis_v1_rescaled_350x230_patch_window_32x32/test')
    cancer_tr.set_dataset('mkfold/train/40X','mkfold/test/40X','average_40X.png')
    cancer_tr.set_network()
    cancer_tr.train()
    #cancer_tr.set_network('./model1/model1.json','./model1/model1.h5')
    
    #cancer_tr.evaluate(filename = './train_res')
    cancer_tr.save_model('./model1/model1.json','./model1/model1.h5')
    """
















    """

    network = Network(img_size, drop_conv, drop_dense)


    network.classifier.compile(optimizer = SGD(lr = lr, momentum = momentum, decay = decay_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    train_datagen = ImageDataGenerator(preprocessing_function = lambda x: x)
    #test_datagen = ImageDataGenerator(preprocessing_function = lambda x: x)

    training_set = train_datagen.flow_from_directory(training,
                                                    target_size = (img_size, img_size),
                                                    batch_size = batch_size,
                                                    class_mode = 'categorical')

    
    #test_set = test_datagen.flow_from_directory(test,
    #                                                target_size = (img_size, img_size),
    #                                                batch_size = batch_size,
    #                                                class_mode = 'categorical')


    training_steps = training_set.samples / training_set.batch_size


    test_datagen = ImageDataGenerator(preprocessing_function = lambda x: x)
    test_set = test_datagen.flow_from_directory(test,
                                                target_size = (img_size, img_size),
                                                batch_size = batch_size,
                                                class_mode = 'categorical',
                                                shuffle = False)

    test_steps = test_set.samples / test_set.batch_size

    history = LossHistory()
    network.classifier.fit_generator(training_set,
                                    steps_per_epoch = training_steps,
                                    epochs = epochs,
                                    validation_data = None if not validate else test_set,
                                    validation_steps = None if not validate else test_steps)
                                    #callbacks = [history])#, validation_data = test_set, validation_steps = test_steps)



    # serialize model to JSON
    model_json = network.classifier.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    network.classifier.save_weights("model.h5")
    print("Saved model to disk")

    #eval = CancerEvaluator(network.classifier, batch_size, img_size)
    result = evaluator.evaluate_images(network.classifier, test_set, by_patient=True)
    #print(result)
    #(tn, tp, fn, fp) = (result['overall']['tn'], result['overall']['tp'], result['overall']['fn'], result['overall']['fp'])

    #print('benign: %f' % (tn/float(fp+tn)))
    #print('malignant: %f' % (tp/float(tp+fn)))

    print('\n\n ***** Overall *****\n\n')
    #(tn, tp, fn, fp) = (result['overall']['tn'], result['overall']['tp'], result['overall']['fn'], result['overall']['fp'])
    print_stats(['Benign', 'Malignant'],result['overall'])

    if show_patients and 'patients' in result.keys():
        for patient in result['patients']:
            print('\n\n ***** Patient %s *****\n\n' % patient)
            #(tn, tp, fn, fp) = (result['patients'][patient]['tn'], result['patients'][patient]['tp'], result['patients'][patient]['fn'], result['patients'][patient]['fp'])
            print_stats(['Benign', 'Malignant'],result['patients'][patient])

    """