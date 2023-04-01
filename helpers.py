import random


def get_random_int(min_value, max_value):
    min_value = int(min_value)
    max_value = int(max_value)
    return random.randint(min_value, max_value)


def letter_to_hex(letter):
    code = ord(letter)
    return hex(code)[2:]


def get_alphabet_array():
    alphabet = []
    for i in range(97, 123):
        letter = chr(i)
        alphabet.append({
            'letter': letter,
            'hex': letter_to_hex(letter)
        })
    return alphabet


def get_random_images_from_bin(letter, count):
    with open(f'dataL/{letter["letter"]}.bin', 'rb') as f:
        data = f.read()

    random_indices = set()
    while len(random_indices) < count and len(random_indices) < (len(data) - 1) // 1024:
        random_index = random.randint(0, len(data) // 1024 - 1)
        random_indices.add(random_index)

    images = []
    for index in random_indices:
        pixels = data[index*1024:(index+1)*1024]
        image = []
        for item in pixels:
            if random.random() > 0.995:
                image.append(item)
            elif random.random() > 0.99:
                image.append(random.randint(1, 254))
            else:
                image.append(255 - item)
        images.append(image)

    return {'l': letter, 'images': images}


def normalize_data(data):
    return [
        {
            'l': letter_data['l'],
            'images': [
                [pixel / 255 for pixel in img]
                for img in letter_data['images']
            ]
        }
        for letter_data in data
    ]
