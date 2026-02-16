import dtlpy as dl
from PIL import Image
from PIL.ExifTags import TAGS
import json

import dtlpy as dl


class ServiceRunner:

    def run(self, item):
        image = Image.open(item.download())
        exif_data = image.getexif()
        exif_dict = {}
        # Iterate through the EXIF data and convert tag numbers to names
        for tag_id, value in exif_data.items():
            # Get the tag name
            tag = TAGS.get(tag_id, tag_id)
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8')
                except UnicodeDecodeError:
                    value = str(value)
            try:
                _ = json.loads(value)
            except Exception:
                value = str(value)
            # Store the tag and its value in the dictionary
            # print(type(tag), type(value))
            exif_dict[tag] = value
            print(f"Tag: {tag}, Value: {value}")

        # Update dict even if empty, for visibility on the item
        if 'user' not in item.metadata:
            item.metadata['user'] = {}
        item.metadata['user']['exif'] = exif_dict
        item.update()


if __name__ == "__main__":

    runner = ServiceRunner()
    runner.run(dl.items.get(item_id='ITEM_ID'))

    pipeline = dl.pipelines.get(pipeline_id='PIPELINE_ID')
    ex = pipeline.execute(execution_input={'item': 'ITEM_ID'})
    ex = ex.wait()
