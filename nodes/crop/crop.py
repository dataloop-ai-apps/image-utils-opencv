import dtlpy as dl
import tempfile
import cv2
import os
import logging
from multiprocessing.pool import ThreadPool
import numpy as np

logger = logging.getLogger('crop-image')


class ServiceRunner(dl.BaseServiceRunner):

    def __init__(self):
        pass

    def crop_images(self, item: dl.Item):
        """
        Crop images based on bounding box annotations and upload the cropped images to the same dataset.

        This function takes an image item, downloads it, and crops it according to the bounding box annotations associated with the item. 
        Each crop is then uploaded as a new item to the same dataset. The metadata of the cropped images includes the annotation details 
        from the original image.

        Parameters:
        item (dl.Item): The item object representing the image to be cropped.

        Returns:
        list: A list of uploaded items representing the cropped images.
        """

        logger.info('Running service')
        image = cv2.imread(item.download(overwrite=True))
        remote_path = os.path.splitext(item.filename)[0]
        temp_items_path = tempfile.mkdtemp()
        logger.info("downloaded item to: {}".format(temp_items_path))
        filters = dl.Filters(resource='annotations')
        filters.add(field='type', values='box')
        annotations = item.annotations.list(filters=filters)
        logger.info("creating crops for {} annotations".format(len(annotations)))
        pool = ThreadPool(processes=6)
        async_results = list()

        if "user" not in item.metadata:
            item.metadata["user"] = {}
                
        for annotation in annotations:

            crop = image[
                int(annotation.top): int(annotation.bottom),
                int(annotation.left): int(annotation.right),
            ]
            file_path = os.path.join(temp_items_path, "{}.jpg".format(annotation.id))
            cv2.imwrite(file_path, crop)
            attributes = annotation.attributes
            if annotation.metadata.get("system", {}).get("attributes", False):
                attributes = annotation.metadata["system"]["attributes"]
            async_results.append(
                pool.apply_async(
                    item.dataset.items.upload,
                    kwds={
                        "local_path": file_path,
                        "remote_path": remote_path,
                        "overwrite": True,
                        "item_metadata": {
                            "user": {
                                "parentItemId": item.id,
                                "originalAnnotationId": annotation.id,
                                "originalAnnotationType": annotation.type,
                                "originalAnnotationLabel": annotation.label,
                                "originalAnnotationAttributes": attributes,
                                "originalAnnotationCoordinates": annotation.coordinates,
                            }
                        },
                    },
                )
            )
        pool.close()
        pool.join()
        uploads = list()
        for async_result in async_results:
            upload = async_result.get()
            uploads.append(upload)

        return list(uploads)

    def add_annotation_to_orig_image(self, item: dl.Item):
        """
        Add annotations from a cropped image to the original image at the appropriate location.

        This function takes a cropped image item, identifies its parent item (the original image from which it was cropped), 
        and transfers the annotations from the cropped image back to the original image at the correct coordinates. 
        It adjusts the coordinates of the annotations to fit the original image's dimensions.

        Parameters:
        item (dl.Item): The item object representing the cropped image with annotations.

        Returns:
        dl.Item: The item object representing the original image with the new annotations.
        """
        logger.info('Running service')
        if item.metadata['system'].get('parentItemId', False):
            parent_item = dl.items.get(item_id=item.metadata['system']['parentItemId'])
        else:
            logger.error('Item has no parent item')
            raise KeyError('Item has no parent item')

        annotations = item.annotations.list()
        new_annotations = dl.AnnotationCollection()

        orig = item.metadata['system']['originalAnnotationCoordinates']

        for annotation in annotations:
            if annotation.type == 'class':
                new_ann = dl.Classification(label=annotation.label, attributes=annotation.attributes, description=annotation.description)
            elif annotation.type == 'point':
                new_ann = dl.Point(x=annotation.x + orig[0]['x'], y=annotation.y + orig[0]['y'], label=annotation.label, attributes=annotation.attributes, description=annotation.description)

            elif annotation.type == 'box':
                new_ann = dl.Box(top=annotation.top + orig[0]['y'], left=annotation.left + orig[0]['x'], bottom=annotation.bottom + orig[0]['y'], right=annotation.right + orig[0]['x'], label=annotation.label, angle=annotation.angle, attributes=annotation.attributes, description=annotation.description)

            elif annotation.type == 'cube':
                new_ann = dl.Cube(front_bl=[annotation.x[0] + orig[0]['x'], annotation.y[0] + orig[0]['y']], front_br=[annotation.x[1] + orig[0]['x'], annotation.y[1] + orig[0]['y']], front_tr=[annotation.x[2] + orig[0]['x'], annotation.y[2] + orig[0]['y']], front_tl=[annotation.x[3] + orig[0]['x'], annotation.y[3] + orig[0]['y']], back_bl=[annotation.x[4] + orig[0]['x'], annotation.y[4] + orig[0]['y']], back_br=[annotation.x[5] + orig[0]['x'], annotation.y[5] + orig[0]['y']], back_tr=[annotation.x[6] + orig[0]['x'], annotation.y[6] + orig[0]['y']], back_tl=[annotation.x[7] + orig[0]['x'], annotation.y[7] + orig[0]['y']], label=annotation.label, angle=annotation.angle, attributes=annotation.attributes, description=annotation.description)

            elif annotation.type == 'binary':
                new_geo = np.zeros((parent_item.height, parent_item.width))

                # Get the top-left corner coordinates
                x, y = int(round(orig[0]['x'])), int(round(orig[0]['y']))

                # Calculate the bounds of the smaller array within the larger one
                x_end = min(x + annotation.geo.shape[1], parent_item.width)
                y_end = min(y + annotation.geo.shape[0], parent_item.height)

                # Place the smaller array in the larger one using slicing
                new_geo[y:y_end, x:x_end] = annotation.geo[:y_end - y, :x_end - x]

                # Create the new segmentation and add it to the annotations
                new_ann = dl.Segmentation(geo=new_geo, color=annotation.color, label=annotation.label, attributes=annotation.attributes, description=annotation.description)

            elif annotation.type == 'segment':
                new_geo = annotation.geo.astype(np.float64)
                new_geo[:, 0] += orig[0]['x']
                new_geo[:, 1] += orig[0]['y']

                new_ann = dl.Polygon(geo=new_geo, label=annotation.label, attributes=annotation.attributes, description=annotation.description)

            elif annotation.type == 'polyline':
                new_geo = annotation.geo.astype(np.float64)
                new_geo[:, 0] += orig[0]['x']
                new_geo[:, 1] += orig[0]['y']

                new_ann = dl.Polyline(geo=new_geo, label=annotation.label, attributes=annotation.attributes, description=annotation.description)

            elif annotation.type == 'ellipse':
                new_ann = dl.Ellipse(x=annotation.x + orig[0]['x'], y=annotation.y + orig[0]['y'], rx=annotation.rx, ry=annotation.ry, angle=annotation.angle, label=annotation.label, attributes=annotation.attributes, description=annotation.description)

            else:
                logger.warning('Annotation type not supported: {}'.format(annotation.type))

            new_annotations.add(new_ann)

        parent_item.annotations.upload(annotations=new_annotations)

        return parent_item
