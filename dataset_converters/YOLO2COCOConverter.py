import json
import os

import cv2

from dataset_converters.ConverterBase import ConverterBase


class YOLO2COCOConverter(ConverterBase):

    formats = ['YOLO2COCO']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)

    def _parse_obj_data_file(self, obj_data_file_path):
        with open(obj_data_file_path, 'r') as f:
            obj_data_file = f.readlines()

        result = {}
        for line in obj_data_file:
            if '=' not in line:
                continue
            k, v = line.strip().split('=')
            result[k.replace(' ', '')] = v.replace(' ', '')
        return result

    def _read_labels(self, filename):
        with open(filename, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        labels = []
        for i, line in enumerate(lines):
            labels.append({'supercategory': 'none', 'id': i+1, 'name': line})

        return labels

    def _read_annotations(self, input_folder, annotations_path):
        with open(annotations_path, 'r') as f:
            lines = [os.path.join(input_folder, l.strip()) for l in f.readlines()]
        instances = {
            "root": os.path.dirname(lines[0]),
            "imgs": {}
        }

        for line in lines:
            filename = os.path.basename(line)
            label_file = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(input_folder, instances["root"], label_file)
            with open(label_path, 'r') as f:
                anns = [l.strip().split() for l in f.readlines()]

            instances["imgs"][filename] = []
            for ann in anns:
                bbox = [float(s) for s in ann[1:]]  # bbox in yolo format: <x_center> <y_center> <width> <height> in range 0..1 
                instances["imgs"][filename].append({'class': int(ann[0]), 'bbox': bbox})

        return instances

    def _yolo_bbox_to_coco(self, yolo_bbox, img_w, img_h):
        x_center, y_center, w, h = yolo_bbox
        x_center *= img_w
        y_center *= img_h
        w *= img_w
        h *= img_h

        x = max(x_center - w / 2, 0.0)
        y = max(y_center - h / 2, 0.0)

        return [x, y, w, h]


    def _convert_subset(self, input_folder, annotations_path):
        to_dump = {'images': [], 'type': 'instances', 'annotations': [], 'categories': self.labels}
        
        image_counter = 1
        instance_counter = 1
        instances = self._read_annotations(input_folder, annotations_path)
        img_root = instances["root"]

        folder = os.path.splitext(os.path.basename(annotations_path))[0]
        subset_image_folder = os.path.join(self.output_folder, folder)
        self._ensure_folder_exists_and_is_clear(subset_image_folder)

        for filename, anns in instances["imgs"].items():
            full_image_path = os.path.join(input_folder, img_root, filename)
            image = cv2.imread(full_image_path)
            to_dump['images'].append(
                {
                    'file_name': filename,
                    'height': image.shape[0],
                    'width': image.shape[1],
                    'id': image_counter
                }
            )
            for ann in anns:
                bbox = self._yolo_bbox_to_coco(ann["bbox"], image.shape[1], image.shape[0])
                x, y, w, h = bbox

                if any([b < 0 for b in bbox]):
                    print("Point 2", bbox, ann["bbox"])

                xmin = x
                xmax = x + w
                ymin = y
                ymax = y + h
                to_dump['annotations'].append(
                    {
                        'segmentation': [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]],
                        'area': w * h,
                        'iscrowd': 0,
                        'image_id': image_counter,
                        'bbox': bbox,
                        'category_id': ann["class"],
                        'id': instance_counter,
                        'ignore': 0
                    }
                )
                instance_counter += 1
            self.copy(full_image_path, subset_image_folder)
            image_counter += 1

        with open(os.path.join(self.annotations_folder, '{0}.json'.format(folder)), 'w') as f:
            json.dump(to_dump, f, indent=4)

    def _run(self, input_folder, output_folder, FORMAT):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.annotations_folder = os.path.join(output_folder, 'annotations')

        self._ensure_folder_exists_and_is_clear(output_folder)
        self._ensure_folder_exists_and_is_clear(self.annotations_folder)

        obj_data_file = os.path.join(input_folder, 'obj.data')
        obj_data = self._parse_obj_data_file(obj_data_file)
        labels_path = os.path.join(input_folder, obj_data["names"])

        self.labels = self._read_labels(labels_path)

        subsets = [
            os.path.join(input_folder, obj_data["train"]),
            os.path.join(input_folder, obj_data["valid"]),
        ]
        for subset in subsets:
            self._convert_subset(input_folder, subset)
        

        
