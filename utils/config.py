from easydict import EasyDict

def get_config() -> EasyDict: 
    conf = EasyDict()

    # Face detection configuration
    conf.face_detection_size = 640
    conf.min_face_detection_size = 512
    conf.face_conf_thres = 0.5

    return conf
