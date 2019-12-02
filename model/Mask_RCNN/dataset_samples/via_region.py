import os
import json
import shutil

import numpy as np
import skimage
import urllib.request
import zipfile

from ..mrcnn import utils


class ViaRegionDataset(utils.Dataset):

    def load_via_region(self, image_dir, annotation_file):
        """Load a VGG Image Annotator(VIA) dataset.
        image_dir: Directory of the image.
        annotation_file: Annotation file to load (.json)
        """
        # Load annotations
        # VGG Image Annotator (version 2.0.8) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {'category_name': 'object_name'},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       '1': {
        #           'region_attributes': {'category_name': 'object_name'},
        #           'shape_attributes': {
        #               'x': [...],
        #               'y': [...],
        #               'width': [...],
        #               'height': [...],
        #               'name': 'rect'}},
        #       '2': {
        #           'region_attributes': {'category_name': 'object_name'},
        #           'shape_attributes': {
        #               'cx': [...],
        #               'cy': [...],
        #               'y': [...],
        #               'name': 'circle'}},
        #       '3': {
        #           'region_attributes': {'category_name': 'object_name'},
        #           'shape_attributes': {
        #               'cx': [...],
        #               'cy': [...],
        #               'rx': [...],
        #               'ry': [...],
        #               'name': 'ellips'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(annotation_file))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        classes = {}  # id: name

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                shapes = [r['shape_attributes']
                          for r in a['regions'].values()]
                categories = [r['region_attributes']
                              for r in a['regions'].values()]
            else:
                shapes = [r['shape_attributes']
                          for r in a['regions']]
                categories = [r['region_attributes']
                              for r in a['regions']]

            for category in categories:
                class_name, object_name = list(category.items())[0]
                if class_name not in list(classes.values()):
                    # Add classes. We have only one class to add.
                    self.add_class("via_region", len(classes)+1,
                                   class_name)
                    classes.update({len(classes)+1: class_name})

            # load_mask() needs the image size to convert shapes to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(image_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "via_region",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                shapes=shapes,
                categories=categories)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "via_region":
            return super(self.__class__, self).load_mask(image_id)

        # Convert shapes to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["shapes"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["shapes"]):
            if p['name'] == 'polygon':
                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = skimage.draw.polygon(p['all_points_y'],
                                              p['all_points_x'])
                mask[rr, cc, i] = 1
            elif p['name'] == 'rect':
                # Get indexes of pixels inside the rect and set them to 1
                rr, cc = skimage.draw.rectangle((p['y'], p['x']),
                                                (p['y'] + p['height'],
                                                 p['x'] + p['width']))
                mask[rr, cc, i] = 1
            elif p['name'] == 'circle':
                # Get indexes of pixels inside the ellipse and set them to 1
                rr, cc = skimage.draw.circle(p['cy'], p['cx'],
                                             p['r'])
                mask[rr, cc, i] = 1
            elif p['name'] == 'ellipse':
                # Get indexes of pixels inside the ellipse and set them to 1
                rr, cc = skimage.draw.ellipse(p['cy'], p['cx'],
                                              p['ry'], p['ry'])
                mask[rr, cc, i] = 1
            else:
                print(p['name'])
                assert False, 'wrong image shapes, name {} is strange.'.format(
                        p['name'])

        # Return mask, and array of class IDs of each instance.
        class_names = [d['name'] for d in self.class_info]
        class_ids = np.array([class_names.index(list(c.keys())[0])
                              for c in info['categories']])
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "via_region":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
