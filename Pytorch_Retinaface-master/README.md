# Cambio ML Interview Question

A [PyTorch](https://pytorch.org/) implementation of [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641). Model size only 1.7M, when Retinaface use mobilenet0.25 as backbone net. We also provide resnet50 as backbone net to get better result. The official code in Mxnet can be found [here](https://github.com/deepinsight/insightface/tree/master/RetinaFace).

Here are simple codes to call RetinaFace to count faces in the given pictures.At first, establish the specific 
counter for the task. According to the config,ResNet50 and Mobilenet backbone are available. Secondly, call the 
instance established in the first step, with a parameter of picture path of folders, then the instance would infer
the picture or the pictures in the floders and output the picture path and count of faces in the terminal.

## details
Create the counters folder, statement the metaclass--FacesCounter,with many functions.

instance func            aim
    |------__init__     set the key information for the task
    |------faces        get the number of faces in the current picture
    |------check        check the pic or the pictures are in the format of 'jpg' ,'png','jpeg', and save the format in the instance
    |------process_single_img
                        infer the single_img
    |------show_results post process and save the result
    |------—__call__    connect the above funtions to do the whole procedure to complete the mission

RetinaFaceCounter and MobileNetV1FacesCounter in SpecificCounter.py are the subclass of FacesCounter,the
whole procedure is similar, all I need to do is to overwrite the __init__ function the load the specific
instance according to the args.

## others
There is only simple testunits in the test folder for RetinaFaceCounter ,because the one of 
MobileNetV1FacesCounter is similar, I omit it, but it is necessary for the real project.

After run the demo, there would be a test log which saved the info and error information good for debugging.

About the environment, I generated the requirement.txt by Pycharm, you can isntall it by pip install -r requirement.txt in your conda environment.
About the system, I used the pathlib to address the path problem '\' '/' in different os.
About the gui, I wrote a simple code to establish a simple gui to infer the single pic.
