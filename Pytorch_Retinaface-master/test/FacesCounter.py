import os.path
import unittest
from ..counters.MetaCounter import DictToObject,FacesCounter
from ..counters.SpecificCounter import RetinaFaceCounter, MobileNetV1FacesCounter

class TestDictToObject(unittest.TestCase):

    def test_dict_to_object(self):
        test_dict = {'key1': 1, 'key2': {'nested_key': 2}}
        obj = DictToObject(test_dict)

        self.assertEqual(obj.key1, 1)
        self.assertEqual(obj.key2.nested_key, 2)

# class TestFacesCounter(unittest.TestCase):
#
#     def setUp(self):
#         # Create a FacesCounter instance with necessary configurations for testing
#         pass
#
#     def test_check_single_image(self):
#         # Test the check method for a single image
#         result = self.faces_counter_instance.check('/path/to/single_image.jpeg')
#         self.assertEqual(result, 0)  # Adjust the assertion based on expected results
#
#     def test_check_image_folder(self):
#         # Test the check method for an image folder
#         result = self.faces_counter_instance.check('/path/to/image_folder')
#         self.assertEqual(result, 1)  # Adjust the assertion based on expected results
#
#     # Add more test methods to cover other functionalities of FacesCounter

class TestRetinaFaceCounter(unittest.TestCase):

    def setUp(self):
        # Create a RetinaFaceCounter instance with necessary configurations for testing
        pass

    def test_model_loading(self):
        # Test if the model is loaded correctly
        tmp_dict = {
            'confidence_threshold': 0.02,
            'top_k': 5000,
            'keep_top_k': 750,
            'vis_thres': 0.6,
            'nms_threshold': 0.4,
        }
        self.counter = RetinaFaceCounter(tmp_dict,
                                       r'pth_model/Retinaface_model_v2-20231202T083246Z-001/Retinaface_model_v2/Resnet50_Final.pth')
        self.assertTrue(isinstance(self.counter, RetinaFaceCounter))

    def test_model_process_folder(self,root = 'testdataset'):
        self.assertTrue( self.counter.check(root)==1)
    def test_model_proce_pic(self,path = 'testdataset/test.jpg'):
        self.assertTrue(self.counter.check(path) == 0)

    def test_process_single_img(self,path = 'testdataset/test.jpg'):
        self.assertEqual(self.counter.process_single_img(path),0)

    def test_single_pic_infer_process(self,path = 'testdataset/test.jpg'):
        self.assertEqual(self.counter(path))

    def test_folder_infer_process(self,path = 'testdataset'):
        pics = [os.path.join(path,i) for i in os.listdir(path)]
        for pic in pics:
            if pic.endswith('result.png')  or pic.endswith('result.jpg')  or pic.endswith('result.jpeg') :
                os.remove(pic)
        self.assertEqual(self.counter(path))

# class TestMobileNetV1FacesCounter(unittest.TestCase):
#
#     def setUp(self):
#         # Create a MobileNetV1FacesCounter instance with necessary configurations for testing
#         pass
#
#     def test_model_loading(self):
#         # Test if the model is loaded correctly
#         self.assertTrue(isinstance(self.mobilenet_counter_instance.model, RetinaFace))  # Adjust the assertion based on expected results
#
#     # Add more test methods to cover other functionalities of MobileNetV1FacesCounter

if __name__ == '__main__':
    unittest.main()
