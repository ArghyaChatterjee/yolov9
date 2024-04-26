import sys
from pathlib import Path
from segment.predict import run  # Assuming predict.py is in the same directory

def webcam_stream():
    # Set parameters for webcam usage
    webcam_params = {
        'weights': Path('./weights/gelan-c-seg.pt'),  # path to model
        'source': '0',  # webcam source
        'data': Path('./data/coco128.yaml'),  # dataset path
        'imgsz': (640, 640),  # image size
        'conf_thres': 0.25,  # confidence threshold
        'iou_thres': 0.45,  # IOU threshold
        'max_det': 1000,  # maximum detections per image
        'device': '',  # computation device
        'view_img': True,  # view image during inference
        'save_txt': False,  # save results to text file
        'project': Path('./runs/predict-seg'),  # directory to save results
        'name': 'webcam_exp',  # name of the experiment
        'exist_ok': True,  # allow existing files
        'line_thickness': 3,  # line thickness for boxes
        'hide_labels': False,  # whether to hide labels
        'hide_conf': False,  # whether to hide confidences
    }

    # Run the webcam stream processing
    run(**webcam_params)

if __name__ == "__main__":
    webcam_stream()
