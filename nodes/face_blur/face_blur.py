import os
import cv2 as cv
import dtlpy as dl
import logging

logger = logging.getLogger('blur-faces')


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        logger.info('Initializing service')
        self.model = cv.CascadeClassifier(
            cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logger.info('Service initialized')

    def blur_faces(self, item: dl.Item, context: dl.Context):
        logger.info('Running service')
        node = context.node
        remote_path = node.metadata['customNodeConfig']['remotePath']

        dataset_id = item.dataset.id if dataset_id is None else dataset_id
        dataset = dl.datasets.get(dataset_id=dataset_id)

        remote_path = os.path.join(
            item.dir, remote_path) if remote_path is not None else item.dir

        if remote_path == item.dir:
            overwrite = True
        else:
            overwrite = False

        # Download the image
        img = item.download(save_locally=False, to_array=True)
        if img.ndim == 2:
            # Image is already grayscale
            gray = img
        else:
            # Convert RGB or RGBA to grayscale
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        # Detect faces
        faces = self.model.detectMultiScale(gray, 1.1, 4)

        # Copy the image for blurring
        blur = img.copy()

        # Apply blur to the regions with faces
        for (x, y, w, h) in faces:
            roi = img[y:y + h, x:x + w]
            blur[y:y + h, x:x + w] = cv.blur(roi, (11, 11))

        # Determine the output file name and format
        extension = os.path.splitext(item.name)[1].lower()
        name = item.name if extension in [
            '.png', '.jpg', '.jpeg'] else os.path.splitext(item.name)[0] + '.jpeg'
        output_path = os.path.join(os.getcwd(), name)

        # Write the modified image to file
        cv.imwrite(output_path, cv.cvtColor(blur, cv.COLOR_RGB2BGR))

        # Upload the image to the dataset
        dataset.items.upload(local_path=output_path,
                             remote_path=remote_path, overwrite=overwrite)

        return item
