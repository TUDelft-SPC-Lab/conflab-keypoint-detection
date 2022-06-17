import json
import funcy
from sklearn.model_selection import train_test_split


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'w', encoding='UTF-8') as coco:
        json.dump(
            {
                'info': info,
                'licenses': licenses,
                'images': images,
                'annotations': annotations,
                'categories': categories
            },
            coco,
            indent=2)


def filter_annotations(annotations, images):
    image_ids = set(map(lambda i: int(i['id']), images))
    return list(filter(lambda a: int(a['image_id']) in image_ids, annotations))


def split(annotation_file, train_file, test_file, split=0.8):
    with open(annotation_file, 'r', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco.get('licenses', [])
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        x, y = train_test_split(images, train_size=split)

        save_coco(train_file, info, licenses, x,
                  filter_annotations(annotations, x), categories)
        save_coco(test_file, info, licenses, y,
                  filter_annotations(annotations, y), categories)

        print("Saved {} entries in {} and {} in {}".format(
            len(x), train_file, len(y), test_file))
