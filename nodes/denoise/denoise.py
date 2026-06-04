import dtlpy as dl
import cv2
import numpy as np
import io
import logging

logger = logging.getLogger(__name__)

class ServiceRunner(dl.BaseServiceRunner):
    def denoise(self, item: dl.Item, context: dl.Context) -> dl.Item:
        
        node = context.node
        custom_config = node.metadata['customNodeConfig']
        filter_type = custom_config['filter_type']
        print(f"Starting denoising for item: {item.id} with filter: {filter_type}")
        logger.info(f"Using filter: {filter_type}")
        print(f"Using filter: {filter_type} - print")
        buffer = item.download(save_locally=False)
        img_array = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Failed to decode image from item {item.id}")

        if filter_type == 'nlmeans':
            h = custom_config['h']
            hColor = custom_config['hColor']
            templateWindowSize = custom_config['templateWindowSize']
            searchWindowSize = custom_config['searchWindowSize']
            denoised_img = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)
        elif filter_type == 'bilateral':
            d = custom_config['d']
            sigmaColor = custom_config['sigmaColor']
            sigmaSpace = custom_config['sigmaSpace']
            denoised_img = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
        elif filter_type == 'median':
            kSize = custom_config['kSize']
            denoised_img = cv2.medianBlur(img, kSize)
        else:
            raise ValueError(f"Unsupported filter: {filter_type}. Choose 'nlmeans', 'bilateral', or 'median'.")

        is_success, encoded_img = cv2.imencode('.jpg', denoised_img)
        if not is_success:
            raise RuntimeError("Failed to encode the denoised image.")
            
        b_io = io.BytesIO(encoded_img.tobytes())
        b_io.name = f"denoised_{filter_type}_{item.name}"

        dataset = item.dataset
        new_item = dataset.items.upload(
            local_path=b_io, 
            remote_path=item.dir
        )
        
        logger.info(f"Successfully uploaded denoised item: {new_item.id}")
        return new_item
