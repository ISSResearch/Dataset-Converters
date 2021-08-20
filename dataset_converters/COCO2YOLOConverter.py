from itertools import groupby
import json
import os
import re

from dataset_converters.ConverterBase import ConverterBase


class COCO2YOLOConverter(ConverterBase):

    formats = ['COCO2YOLO']

    def __init__(self, copy_fn):
        ConverterBase.__init__(self, copy_fn)

    def _run(self, input_folder, output_folder, FORMAT):
        annotations_dir = os.path.join(input_folder, 'annotations')
        annotations_list = os.listdir(annotations_dir)

        self._ensure_folder_exists_and_is_clear(output_folder)
        for filename in annotations_list:
            image_folder = os.path.join(input_folder, filename[:-5])
            out_folder = os.path.join(output_folder, filename[:-5])
            self._ensure_folder_exists_and_is_clear(out_folder)
            out_img_folder = os.path.join(output_folder, filename[:-5])
            self._ensure_folder_exists_and_is_clear(out_img_folder)
            annotation_file = os.path.join(annotations_dir, filename)

            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

            cats = {cat["id"]: cat["name"] for cat in annotations["categories"]}
            imgs_list = ""

            for img in annotations["images"]:
                i_w, i_h = img["width"], img["height"]
                i_src = os.path.join(image_folder, img["file_name"])
                labels = ""
                anns = [ann for ann in annotations["annotations"] if ann["image_id"] == img["id"]]
                for ann in anns:
                    # Bbox to yolo format
                    cls_id = ann["category_id"]
                    x, y, width, height = ann["bbox"]
                    x_center, y_center = x + width//2, y + height//2
                    labels += "{0} {1} {2} {3} {4}\n".format(cls_id, x_center/i_w, y_center/i_h, width/i_w, height/i_h)
                dst_clear_name = img["file_name"]

                i_dst = os.path.join(out_img_folder, dst_clear_name)
                l_path = i_dst[:i_dst.rfind(".")] + ".txt"
                imgs_list += os.path.join('.', filename[:-5], dst_clear_name) + "\n"


                self.copy(i_src , i_dst)
                with open(l_path, "w") as f:
                    f.write(labels)

            data = (
                'classes = {0}'.format(len(cats)),
                'train  = ./train.txt',
                'names = ./obj.names',
                'backup = backup/',
            )
            with open(os.path.join(output_folder, "obj.data"), "w") as f:
                    f.write("\n".join(data))

            with open(os.path.join(output_folder, "train.txt"), "w") as f:
                    f.write(imgs_list)

            names = "\n".join([cats[i] for i in range(min(cats.keys()), max(cats.keys())+1)])
            with open(os.path.join(output_folder, "obj.names"), "w") as f:
                f.write(names)

            
