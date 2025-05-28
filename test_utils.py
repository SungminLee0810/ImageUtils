import os
import cv2
import numpy as np
import unittest

# Assuming utils.py is in the same directory or accessible in PYTHONPATH
from utils import imgfmt_convert

class TestImgfmtConvert(unittest.TestCase):

    def test_convert_png_to_jpg(self):
        # Dummy image parameters
        img_height = 64
        img_width = 64
        img_channels = 3
        input_filename = "dummy_input.png"
        output_filename_base = "dummy_input" # imgfmt_convert appends the extension
        target_format = "jpg"
        output_filepath_expected = f"./{output_filename_base}.{target_format}" # Added ./ to match the output of imgfmt_convert

        # Create a dummy PNG image (black square)
        dummy_image = np.zeros((img_height, img_width, img_channels), dtype=np.uint8)
        cv2.imwrite(input_filename, dummy_image)

        # Ensure input file was created
        self.assertTrue(os.path.exists(input_filename), f"Setup failed: {input_filename} was not created.")

        # Call imgfmt_convert
        # Assuming tgt_path is the current directory for simplicity
        converted_image_path = imgfmt_convert(input_filename, ".", tgt_fmt=target_format)

        # 1. Assert that the output file exists
        self.assertTrue(os.path.exists(converted_image_path), f"Output file {converted_image_path} does not exist.")
        
        # 2. Assert that the output file has the .jpg extension
        self.assertTrue(converted_image_path.endswith(f".{target_format}"),
                        f"Output file {converted_image_path} does not have the .{target_format} extension.")
        self.assertEqual(converted_image_path, output_filepath_expected)

        # 3. Clean up
        if os.path.exists(input_filename):
            os.remove(input_filename)
        if os.path.exists(converted_image_path):
            os.remove(converted_image_path)

if __name__ == '__main__':
    unittest.main()
