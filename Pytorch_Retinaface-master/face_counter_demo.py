import argparse
import sys
from counters.SpecificCounter import RetinaFaceCounter,MobileNetV1FacesCounter
parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-f','--file_path',default=r'E:\projects\python_pro\Pytorch_Retinaface-master\Pytorch_Retinaface-master\curve\test.jpg',
                    type=str,required=False, help='file_path could be a specific pic path or a floder path')
parser.add_argument('-m', '--trained_model', default=r'E:\projects\python_pro\Pytorch_Retinaface-master\Pytorch_Retinaface-master\pth_model\Retinaface_model_v2-20231202T083246Z-001\Retinaface_model_v2\Resnet50_Final.pth',   # default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

if __name__ == '__main__':
# args_1 = sys.argv
    if args.network == 'resnet50':
        counter = RetinaFaceCounter(args)#args.trained_model)
    elif args.network == 'mobile0.25':
        counter = MobileNetV1FacesCounter(args)#args.trained_model)
    counter(args.file_path)
    # print(counter.faces(args.file_path))