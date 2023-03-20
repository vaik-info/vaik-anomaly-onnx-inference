from typing import List, Dict, Tuple
import onnxruntime as rt
from PIL import Image, ImageOps
import numpy as np


class OnnxModel:
    def __init__(self, input_saved_model_path: str = None):
        self.model = self.__load(input_saved_model_path)
        self.model_input_shape = tuple(self.model.get_inputs()[0].shape)
        self.input_name = self.model.get_inputs()[0].name
        self.model_output_shape = tuple(self.model.get_outputs()[0].shape)
        self.output_name = self.model.get_outputs()[0].name

    def __load(self, input_saved_model_path):
        model = rt.InferenceSession(input_saved_model_path, providers=rt.get_available_providers())
        return model

    def inference(self, input_image_list: List[np.ndarray], batch_size: int = 8) -> Tuple[List[Dict], Dict]:
        resized_image_array = self.__preprocess_image_list(input_image_list, self.model_input_shape[1:3])
        raw_pred = self.__inference(resized_image_array, batch_size)
        output = self.__output_parse(raw_pred)
        return output, raw_pred

    def __inference(self, resize_input_tensor: np.ndarray, batch_size: int) -> np.ndarray:
        if len(resize_input_tensor.shape) != 4:
            raise ValueError('dimension mismatch')

        output_tensor = np.zeros((resize_input_tensor.shape[0],) + self.model_output_shape[1:])
        for index in range(0, resize_input_tensor.shape[0], batch_size):
            batch = resize_input_tensor[index:index + batch_size, :, :, :]
            batch_pad = np.zeros(((batch_size, ) + self.model_input_shape[1:]), dtype=np.float32)
            batch_pad[:batch.shape[0], :, :, :] = batch
            raw_pred = self.model.run([self.output_name], {self.input_name: batch_pad})[0]
            output_tensor[index:index + batch.shape[0], :] = np.stack(raw_pred[:batch.shape[0]], axis=0)
        return output_tensor

    def __preprocess_image_list(self, input_image_list: List[np.ndarray],
                                resize_input_shape: Tuple[int, int]) -> np.ndarray:
        resized_image_list = []
        for input_image in input_image_list:
            resized_image = self.__preprocess_image(input_image, resize_input_shape)
            resized_image_list.append(resized_image)
        return np.stack(resized_image_list)

    def __preprocess_image(self, input_image: np.ndarray, resize_input_shape: Tuple[int, int]) -> Tuple[
        np.ndarray, Tuple[float, float]]:
        if len(input_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_image.dtype}')

        pil_image = Image.fromarray(input_image)
        width, height = pil_image.size

        scale = min(resize_input_shape[0] / height, resize_input_shape[1] / width)

        resize_width = int(width * scale)
        resize_height = int(height * scale)

        resize_pil_image = pil_image.resize((resize_width, resize_height))
        padding_bottom, padding_right = resize_input_shape[0] - resize_height, resize_input_shape[1] - resize_width
        resize_pil_image = ImageOps.expand(resize_pil_image, (0, 0, padding_right, padding_bottom), fill=0)
        resize_image = np.array(resize_pil_image)
        return resize_image

    def __output_parse(self, pred: np.ndarray) -> List[Dict]:
        output_dict_list = []
        for index in range(pred.shape[0]):
            output_dict = {'score': np.mean(pred[index])}
            output_dict_list.append(output_dict)
        return output_dict_list
