import os
import numpy
import random
import cv2
import argparse
from captcha.image import ImageCaptcha


def generate_captcha_symbols(symbols_file):
    with open(symbols_file, 'r') as file:
        return file.readline().strip()


def generate_captcha_images(width, height, captcha_symbols, lower_length, upper_length, count, output_dir, dict_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    captcha_generate = ImageCaptcha(width=width, height=height, fonts=['Jester.ttf'])
    cap_dict = {}

    for i in range(count):
        random_str = ''.join([random.choice(captcha_symbols) for _ in range(random.randint(lower_length, upper_length))])
        image_path = os.path.join(output_dir, f"{i}.png")
        cap_dict[i] = random_str

        image = numpy.array(captcha_generate.generate_image(random_str))
        cv2.imwrite(image_path, image)

    with open(dict_name, 'w') as dict_file:
        for index, captcha_text in cap_dict.items():
            dict_file.write(f"{index} {captcha_text}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--upper-length', help='Upper length limit of captchas in characters', type=int)
    parser.add_argument('--lower-length', help='Lower length limit of captchas in characters', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--map-name', help='Name for the dictionary map, translating captcha ids into symbols', type=str)
    args = parser.parse_args()

    if None in [args.width, args.height, args.upper_length, args.lower_length, args.count, args.output_dir, args.symbols, args.dict_name]:
        print("Please provide all required arguments")
        exit(1)

    captcha_symbols = generate_captcha_symbols(args.symbols)
    print(f"Generating captchas with symbol set {{{captcha_symbols}}}")
    generate_captcha_images(args.width, args.height, captcha_symbols, args.lower_length, args.upper_length, args.count, args.output_dir, args.dict_name)


if __name__ == '__main__':
    main()
