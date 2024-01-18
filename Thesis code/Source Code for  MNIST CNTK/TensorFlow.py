from __future__ import print_function
import numpy as np
import cntk
from cntk import ops, layers, io, learning, Trainer, utils

def create_reader(path, is_training, input_dim, label_dim):
    return io.MinibatchSource(io.CTFDeserializer(path, io.StreamDefs(
        features=io.StreamDef(field='features', shape=input_dim),
        labels=io.StreamDef(field='labels', shape=label_dim)
    )), randomize=is_training, epoch_size=io.INFINITELY_REPEAT if is_training else io.FULL_DATA_SWEEP)

def convnet_mnist(debug_output=False):
    abs_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(abs_path, "MNIST")
    model_path = os.path.join(abs_path, "Models")

    image_height, image_width, num_channels = 28, 28, 1
    input_dim = image_height * image_width * num_channels
    num_output_classes = 10

    input_var = ops.input_variable((num_channels, image_height, image_width), np.float32)
    label_var = ops.input_variable(num_output_classes, np.float32)

    scaled_input = ops.element_times(ops.constant(0.00392156), input_var)

    with layers.default_options(activation=ops.relu, pad=False):
        conv1 = layers.Convolution2D((5, 5), 32, pad=True)(scaled_input)
        pool1 = layers.MaxPooling((3, 3), (2, 2))(conv1)
        conv2 = layers.Convolution2D((3, 3), 48)(pool1)
        pool2 = layers.MaxPooling((3, 3), (2, 2))(conv2)
        conv3 = layers.Convolution2D((3, 3), 64)(pool2)
        f4 = layers.Dense(96)(conv3)
        drop4 = layers.Dropout(0.5)(f4)
        z = layers.Dense(num_output_classes, activation=None)(drop4)

    ce = ops.cross_entropy_with_softmax(z, label_var)
    pe = ops.classification_error(z, label_var)

    reader_train = create_reader(os.path.join(data_path, 'Train-28x28_cntk_text.txt'), True, input_dim, num_output_classes)

    epoch_size = 60000
    minibatch_size = 128

    lr_schedule = learning.learning_rate_schedule([0.01], learning.UnitType.sample, epoch_size)
    learner = learning.adagrad(z.parameters, lr_schedule)
    trainer = Trainer(z, (ce, pe), learner)

    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    utils.log_number_of_parameters(z)
    print()

    max_epochs = 20
    progress_printer = utils.ProgressPrinter(tag='Training', log_to_file='log.txt', num_epochs=max_epochs)

    for epoch in range(max_epochs):
        sample_count = 0

        while sample_count < epoch_size:
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map)
            trainer.train_minibatch(data)
            sample_count += data[label_var].num_samples

        progress_printer.update_with_trainer(trainer, with_metric=True)
        progress_printer.epoch_summary(with_metric=True)

        z.save(os.path.join(model_path, "ConvNet_MNIST_{}.dnn".format(epoch)))

    reader_test = create_reader(os.path.join(data_path, 'Test-28x28_cntk_text.txt'), False, input_dim, num_output_classes)

    input_map_test = {
        input_var: reader_test.streams.features,
        label_var: reader_test.streams.labels
    }

    epoch_size_test = 10000
    minibatch_size_test = 128

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
    convnet_mnist()
