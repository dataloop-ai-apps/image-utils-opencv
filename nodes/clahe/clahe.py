import os
import cv2
import dtlpy as dl
import shutil


class ServiceRunner(dl.BaseServiceRunner):
    def __init__(self):
        ...

    @staticmethod
    def apply_clahe(item: dl.Item, context: dl.Context):
        node = context.node
        remote_path = node.metadata['customNodeConfig']['remote_path']
        # Load the image in grayscale
        local_path = os.path.join(os.getcwd(), f'clahe_{item.id}')
        os.makedirs(local_path, exist_ok=True)
        image_path = item.download(local_path=local_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Create a CLAHE object
        clahe = cv2.createCLAHE()

        # Apply CLAHE to the image
        equalized_image = clahe.apply(image)

        # Save the resulting image
        output_path = os.path.join(local_path, f'clahe_{item.name}')
        cv2.imwrite(output_path, equalized_image)
        clahe_image = item.dataset.items.upload(local_path=output_path,
                                                remote_path=remote_path,
                                                item_metadata={'user': {
                                                    'original_item_id': item.id
                                                }}
                                                )
        shutil.rmtree(local_path)
        return clahe_image
