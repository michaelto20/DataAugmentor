import albumentations as A
import cv2
import os
import glob

def augment_data(image, bboxes, class_labels):
    transformed_images = []
    h,w,_ = image.shape
    crop_width = int(w * 0.9)
    crop_height = int(h * 0.9)
    min_area = crop_height * crop_width
    for i in range(10):
        transform = A.Compose([
            A.RandomCrop(width=crop_width, height=crop_height),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', min_area=min_area - 1, min_visibility=0.2, label_fields=['class_labels']))

        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_images.append(transformed)
    return transformed_images

def get_augmentation(path_to_bboxes, path_to_images):
    '''
    for each image in images folder, get corresponding bbox file and pass them both
    to augment_data()
    '''
    transformed_images = []
    class_labels = [0]
    files = glob.glob(f'{path_to_images}/*')
    for f in files:
        filename = f.split('\\')[-1]
        filename_no_extension = filename.split('.')[0]
        image = cv2.imread(f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox_path = os.path.join(path_to_bboxes, filename_no_extension + '.txt')
        bboxes = get_bbox(bbox_path)
        trans = augment_data(image, bboxes, class_labels)
        transformed_images += trans
    
    return transformed_images

def save_augmentations(augmentations):
    count = 0
    for transformed in augmentations:
        img = transformed['image']
        image_path = os.path.join('output', f'{count}.png')
        cv2.imwrite(image_path, img)
        count += 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_path = os.path.join('output', f'{count}.png')
        cv2.imwrite(image_path, img)
        count += 1


def get_bbox(bbox_filepath):
    with open(bbox_filepath, 'r') as reader:
        line = reader.readline()
        bboxes = []
        while line != '':
            tokens = [float(x) for x in line.strip().split(' ')]
            bbox = tokens[1:] # remove first value as it is the class
            bbox.append(tokens[0])
            bboxes.append(bbox)
            line = reader.readline()
        return bboxes


def cleanup_output_folder():
    '''
    Iterate through output folder and delete everthingy
    '''
    files = glob.glob('output/*')
    for f in files:
        os.remove(f)

if __name__ == "__main__":
    cleanup_output_folder()
    
    path_to_images = r'C:\Development\labelImg\output'
    path_to_bboxes = r'C:\Development\labelImg\numbers_annotated'
    augmentations = get_augmentation(path_to_bboxes, path_to_images)

    save_augmentations(augmentations)
