import dtlpy as dl
import cv2
import numpy as np
import io
import logging
import os

logger = logging.getLogger(__name__)

class ServiceRunner(dl.BaseServiceRunner):
    def denoise(self, item: dl.Item, context: dl.Context) -> dl.Item:
        
        node = context.node
        dataset = item.dataset
        file_format = os.path.splitext(item.filename)[1].lower()
        metadata = dict()
        custom_config = node.metadata['customNodeConfig']
        filter_type = custom_config['filter_type']
        
        if not item.mimetype.startswith('image'):
            raise ValueError(f"Item {item.id} is not an image (mimetype: {item.mimetype}). This node only supports image items.")
        
        buffer = item.download(save_locally=False)
        img_array = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Failed to decode image from item {item.id}")

        if filter_type == 'nlmeans':
            h = custom_config['h']
            h_color = custom_config['h_color']
            template_window_size = custom_config['template_window_size']
            search_window_size = custom_config['search_window_size']
            metadata['denoising_strength_brightness'] = h
            metadata['denoising_strength_color'] = h_color
            metadata['patch_size'] = template_window_size
            metadata['search_area_size'] = search_window_size
            denoised_img = cv2.fastNlMeansDenoisingColored(img, None, h, h_color, template_window_size, search_window_size)
        elif filter_type == 'bilateral':
            d = custom_config['d']
            sigma_color = custom_config['sigma_color']
            sigma_space = custom_config['sigma_space']
            metadata['neighborhood_size'] = d
            metadata['color_sensitivity'] = sigma_color
            metadata['spatial_influence'] = sigma_space
            denoised_img = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        elif filter_type == 'median':
            k_size = custom_config['k_size']
            metadata['kernel_size'] = k_size
            denoised_img = cv2.medianBlur(img, k_size)
        else:
            raise ValueError(f"Unsupported filter: {filter_type}. Choose 'nlmeans', 'bilateral', or 'median'.")

        is_success, encoded_img = cv2.imencode(file_format, denoised_img)
        if not is_success:
            raise RuntimeError("Failed to encode the denoised image.")
            
        b_io = io.BytesIO(encoded_img.tobytes())
        b_io.name = f"denoised_{filter_type}_{item.name}"

        new_item = dataset.items.upload(
            local_path=b_io, 
            remote_path=item.dir,
            item_metadata={'denoise_metadata': metadata}
        )
        
        logger.info(f"Successfully uploaded denoised item: {new_item.id} using {filter_type} filter")
        return new_item
