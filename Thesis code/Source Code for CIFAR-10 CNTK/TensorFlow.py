from __future__ import print_function
import numpy as np
import cntk
from cntk import ops, layers, io, learning, Trainer, utils

def create_reader(path, is_training, input_dim, label_dim):
    return io.MinibatchSource(io.CTFDeserializer(path, io.StreamDefs(
        features=io.StreamDef(field='features', shape=input_dim),
        labels=io.StreamDef(field='labels', shape=label_dim)
    )), randomize=is_training, epoch_size=io.INFINITELY_REPEAT if is_training else io.FULL_DATA_SWEEP)

def convnet_cifar10(debug_output=False):
    utils.set_computation_network_trace_level(0)
    
    image_height = 32
    image_width = 32
    num_channels = 3
    input_dim = image_height * image_width * num_channels
    num_output_classes = 10

    input_var = ops.input_variable((num_channels, image_height, image_width), np.float32)
    label_var = ops.input_variable(num_output_classes, np.float32)

    input_removemean = ops.minus(input_var, ops.constant(128))
    scaled_input = ops.element_times(ops.constant(0.00392156), input_removemean)

    with layers.default_options(activation=ops.relu, pad=False):
        model = layers.Sequential([
            layers.Convolution2D((3, 3), 32, pad=True),
            layers.Convolution2D((3, 3), 32),
            layers.MaxPooling((2, 2), (2, 2)),
            layers.Dropout(0.25),
            layers.Convolution2D((3, 3), 64, pad=True),
            layers.Convolution2D((3, 3), 64),
            layers.MaxPooling((2, 2), (2, 2)),
            layers.Dropout(0.25),
            layers.Dense(512),
            layers.Dropout(0.5),
            layers.Dense(num_output_classes, activation=None)
        ])(scaled_input)

    ce = ops.cross_entropy_with_softmax(model, label_var)
    pe = ops.classification_error(model, label_var)

    reader_train = create_reader("Train_cntk_text.txt", True, input_dim, num_output_classes)
    
    epoch_size = 50000
    minibatch_size = 32
    lr_per_sample = [0.01]
    lr_schedule = learning_rate_schedule(lr_per_sample, learner.UnitType.sample, epoch_size)

    learner = learning.adagrad(model.parameters, lr_schedule)
    trainer = Trainer(model, (ce, pe), learner)

    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    max_epochs = 40
    utils.log_number_of_parameters(model)
    print()

    progress_printer = utils.ProgressPrinter(tag='Training', log_to_file='log.txt', num_epochs=max_epochs)

    for epoch in range(max_epochs):
        sample_count = 0

        while sample_count < epoch_size:
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map)
            trainer.train_minibatch(data)
            sample_count += trainer.previous_minibatch_sample_count

        progress_printer.update_with_trainer(trainer, with_metric=True)
        progress_printer.epoch_summary(with_metric=True)
        
        model.save_model(os.path.join("Models", "ConvNet_CIFAR10_{}.dnn".format(epoch)))

    reader_test = create_reader("Test_cntk_text.txt", False, input_dim, num_output_classes)
    
    input_map_test = {
        input_var: reader_test.streams.features,
        label_var: reader_test.streams.labels
    }

    epoch_size_test = 10000
    minibatch_size_test = 32

    metric_numer = 0
    metric_denom = 0
    sample_count_test = 0
    minibatch_index_test = 0

    while sample_count_test < epoch_size_test:
        current_minibatch_test = min(minibatch_size_test, epoch_size_test - sample_count_test)
        data_test = reader_test.next_minibatch(current_minibatch_test, input_map=input_map_test)

        metric_numer += trainer.test_minibatch(data_test) * current_minibatch_test
        metric_denom += current_minibatch_test
        sample_count_test += data_test[label_var].num_samples
        minibatch_index_test += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(
        minibatch_index_test + 1, (metric_numer * 100.0) / metric_denom, metric_denom))
    print("")

    return metric_numer / metric_denom

if __name__ == '__main__':
    convnet_cifar10()
