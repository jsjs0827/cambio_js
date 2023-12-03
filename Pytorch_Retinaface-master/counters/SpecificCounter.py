from .MetaCounter import FacesCounter
from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace#,MobileNetV1
import torch
from .detect import check_keys,remove_prefix,load_model
import traceback
import logging
# logging.basicConfig(filename='test.log',filemode='a',format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)


class RetinaFaceCounter(FacesCounter):

    def __init__(self,count_cfg,model_path:str=r'pth_model/Retinaface_model_v2-20231202T083246Z-001/Retinaface_model_v2/Resnet50_Final.pth',model_cfg:dict=None):
        super(RetinaFaceCounter,self).__init__(count_cfg)
        self.cfg = model_cfg if model_cfg else cfg_re50
        self.model = RetinaFace(self.cfg, phase='test')
        # assert model_path,'Please input the necessary param:model_path'
        # if model_path:
        try:
            if self.device!='cpu':
                pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(self.device))
            if "state_dict" in pretrained_dict.keys():
                pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
            else:
                pretrained_dict = remove_prefix(pretrained_dict, 'module.')
            check_keys(self.model, pretrained_dict)
            self.model.load_state_dict(pretrained_dict, strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)
            logging.info('Finished loading {} model!'.format(self.cfg['name']))
            print('Finished loading {} model!'.format(self.cfg['name']))
        except Exception as e:
            logging.error('load {} model failed,error_type:{},error_mes:{}'.format(self.cfg['name'],type(e).__name__,str(e)))
            traceback.print_exc()

class MobileNetV1FacesCounter(FacesCounter):

    def __init__(self,count_cfg,model_path:str=r'pth_model/Retinaface_model_v2-20231202T083246Z-001/Retinaface_model_v2/mobilenet0.25_Final.pth',model_cfg:dict=None):
        super(MobileNetV1FacesCounter,self).__init__(count_cfg)
        self.cfg = model_cfg if model_cfg else cfg_mnet
        # assert model_path, 'Please input the necessary param:model_path'
        try:
            self.model =  RetinaFace(self.cfg, phase='test')
            self.model  = load_model(self.model , model_path, True if self.device=='cpu' else False)
            self.model.eval()
            self.model = self.model.to(self.device)
            logging.info('Finished loading {} model!'.format(self.cfg['name']))
            print('Finished loading {} model!'.format(self.cfg['name']))
        except Exception as e:
            logging.error(
                'load {} model failed,error_type:{},error_mes:{}'.format(self.cfg['name'], type(e).__name__, str(e)))
            traceback.print_exc()

if __name__ == '__main__':
    # counter = RetinaFaceCounter(r'E:\projects\python_pro\Pytorch_Retinaface-master\Pytorch_Retinaface-master\pth_model\Retinaface_model_v2-20231202T083246Z-001\Retinaface_model_v2\Resnet50_Final.pth')
    counter = MobileNetV1FacesCounter(r"E:\projects\python_pro\Pytorch_Retinaface-master\Pytorch_Retinaface-master\pth_model\Retinaface_model_v2-20231202T083246Z-001\Retinaface_model_v2\mobilenet0.25_Final.pth")
    counter(r'E:\projects\python_pro\Pytorch_Retinaface-master\Pytorch_Retinaface-master\curve\test.jpg')
    print(counter.faces(r'E:\projects\python_pro\Pytorch_Retinaface-master\Pytorch_Retinaface-master\curve\test.jpg'))