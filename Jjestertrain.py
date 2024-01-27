import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras

def create_model(captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=2):
    input_tensor = keras.Input(input_shape)
    x = input_tensor
    for i, module_length in enumerate([module_size] * model_depth):
        for j in range(module_length):
            x = keras.layers.Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Flatten()(x)
    x = [keras.layers.Dense(captcha_num_symbols, activation='softmax', name=f'char_{i + 1}')(x) for i in range(captcha_length)]
    model = keras.Model(inputs=input_tensor, outputs=x)
    return model

class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, dictionary, batch_size, captcha_length, captcha_symbols, captcha_width, captcha_height):
        self.directory_name = directory_name
        self.dictionary = dictionary
        self.batch_size = batch_size
        self.captcha_length = captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height

        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        return int(numpy.floor(self.count / self.batch_size)) - 1

    def __getitem__(self, idx):
        X = numpy.zeros((self.batch_size, self.captcha_height, self.captcha_width, 3), dtype=numpy.float32)
        y = [numpy.zeros((self.batch_size, len(self.captcha_symbols)), dtype=numpy.uint8) for _ in range(self.captcha_length)]

        for i in range(self.batch_size):
            random_image_label = random.choice(list(self.files.keys()))
            random_image_file = self.files[random_image_label]

            self.used_files.append(self.files.pop(random_image_label))

            raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            processed_data = numpy.array(rgb_data) / 255.0
            X[i] = processed_data

            random_image_label = self.dictionary[random_image_label].ljust(self.captcha_length, '&')

            for j, ch in enumerate(random_image_label):
                y[j][i, :] = 0
                y[j][i, self.captcha_symbols.find(ch)] = 1

        return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='length limit of captchas in characters', type=int)
    parser.add_argument('--batch-size', help='How many images in training captcha batches', type=int)
    parser.add_argument('--train-dataset', help='Where to look for the training image dataset', type=str)
    parser.add_argument('--validate-dataset', help='Where to look for the validation image dataset', type=str)
    parser.add_argument('--output-model-name', help='Where to save the trained model', type=str)
    parser.add_argument('--input-model', help='Where to look for the input model to continue training', type=str)
    parser.add_argument('--epochs', help='How many training epochs to run', type=int)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--train-map', help='name for the training map', type=str)
    parser.add_argument('--validate-map', help='name for the validate map', type=str)
    args = parser.parse_args()

    required_args = [args.width, args.height, args.length, args.batch_size, args.epochs, args.train_dataset, args.validate_dataset, args.output_model_name, args.symbols, args.train_dict, args.validate_dict]
    if None in required_args:
        print("Please provide all required arguments")
        exit(1)

    captcha_symbols = None
    with open(args.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline()

    trai_dict = {}
    with open(args.train_dict) as train_dict_file:
        lines = train_dict_file.readlines()
        for line in lines:
            line_split = line.split(' ')
            trai_dict[line_split[0]] = line_split[1].replace('\n', '')

    val_dict = {}
    with open(args.validate_dict) as validate_dict_file:
        lines = validate_dict_file.readlines()
        for line in lines:
            line_split = line.split(' ')
            val_dict[line_split[0]] = line_split[1].replace('\n', '')

    with tf.device('/device:GPU:0'):
        cap_model = create_model(args.length, len(captcha_symbols), (args.height, args.width, 3))

        if args.input_model is not None:
            cap_model.load_weights(args.input_model)

        cap_model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])

        cap_model.summary()

        tran_data = ImageSequence(args.train_dataset, trai_dict, args.batch_size, args.length, captcha_symbols, args.width, args.height)
        val_data = ImageSequence(args.validate_dataset, val_dict, args.batch_size, args.length, captcha_symbols, args.width, args.height)

        callbacks = [keras.callbacks.EarlyStopping(patience=3),
                     keras.callbacks.ModelCheckpoint(args.output_model_name + '.h5', save_best_only=False)]

        with open(args.output_model_name + ".json", "w") as json_file:
            json_file.write(cap_model.to_json())

        try:
            cap_model.fit(x=tran_data,
                      validation_data=val_data,
                      epochs=args.epochs,
                      callbacks=callbacks,
                      use_multiprocessing=True)
        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' + args.output_model_name + '_resume.h5')
            cap_model.save_weights(args.output_model_name + '_resume.h5')

if __name__ == '__main__':
    main()
