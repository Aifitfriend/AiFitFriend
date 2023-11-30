import json
import funcy
from sklearn.model_selection import train_test_split


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def split_coco(original_path, train_path, validation_path, split):
    with open(original_path, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info'] if 'info' in coco else {}
        licenses = coco['licenses'] if 'licenses' in coco else {}
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        x, y = train_test_split(images, train_size=split)

        save_coco(train_path, info, licenses, x, filter_annotations(annotations, x), categories)
        save_coco(validation_path, info, licenses, y, filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(len(x), train_path, len(y), validation_path))


if __name__ == "__main__":
    original_path = "/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/person_keypoints_val2017.json"
    train_path = "/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/train_data.json"
    validation_path = "/Users/dharrensandhi/PycharmProjects/model_1_keypoint_detection/validation_data.json"
    split = 0.7
    split_coco(original_path, train_path, validation_path, split)