import numpy as np

class BasePostprocessor:
    def process(self, inference_output):
        pass

class ImagePostprocessor(BasePostprocessor):
    def process(self, tensor_output):
        # tensor_output: (1, 3, H, W)
        image = tensor_output.permute(0, 2, 3, 1).cpu().numpy()
        image = image[0][..., ::-1] # RGB to BGR for OpenCV
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        return image
