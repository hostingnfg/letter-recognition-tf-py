import tensorflow as tf
from tensorflow.keras import backend as K
import os
import asyncio
from helpers import get_alphabet_array, get_random_images_from_bin, normalize_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

correction_str = ""
s_id = 1

async def start(is_init):
    global s_id, correction_str

    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    s_id = max([int(item.split("-")[1]) for item in os.listdir(models_dir)], default=0) + 1

    letters = get_alphabet_array()
    letters_data = normalize_data([
        get_random_images_from_bin(l, 3000)
        if l['letter'] in correction_str
        else get_random_images_from_bin(l, 300)
        for l in letters
    ])

    images = []
    labels = []
    num_classes = 26

    for data in letters_data:
        for image in data['images']:
            images.append(image)
            labels.append(ord(data['l']['letter']) - 97)

    images_nested = [tf.reshape(image, [32, 32, 1]) for image in images]

    alphabet_images_nested = [tf.reshape(data['images'][index], [32, 32, 1]) for index, data in enumerate(letters_data)]

    train_images = tf.stack(images_nested)

    labels_tensor = tf.constant(labels, dtype=tf.int32)
    train_labels = tf.one_hot(labels_tensor, depth=num_classes)

    if is_init:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32, 32, 1)),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=num_classes, activation='softmax')
        ])
    else:
        model = tf.keras.models.load_model(f'models/model-{s_id-1}')

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=5, batch_size=45, validation_split=0.1, callbacks=[

    ]).history

    on_epoch_end(model, alphabet_images_nested)

    model.save(f'models/model-{s_id}')
    await start(False)


def on_epoch_end(model, alphabet_images_nested):
    global correction_str

    alphabet_images = tf.stack(alphabet_images_nested)
    alphabet_predictions = model.predict(alphabet_images)
    alphabet_prediction_labels = tf.argmax(alphabet_predictions, axis=-1)
    alphabet_prediction_labels_array = alphabet_prediction_labels.numpy()

    correction_str = ""
    for i, val in enumerate(alphabet_prediction_labels_array):
        if val != i:
            correction_str += chr(i + 97)

    print(correction_str)

    K.clear_session()


asyncio.run(start(True))
