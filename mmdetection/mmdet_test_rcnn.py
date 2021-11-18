from mmdet.apis import init_detector, inference_detector
import pathlib
# BASE_DIR will be like '/home/jovyan/DemoExample/'
BASE_DIR = pathlib.Path(__file__).parent.absolute()
print(f"Working dir: {BASE_DIR}")

config_file = BASE_DIR / 'mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = BASE_DIR / 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
device = 'cuda:0'
# init a detector
model = init_detector(f"{config_file}", f"{checkpoint_file}", device=device)
# inference the demo image
image = BASE_DIR / "mmdetection/demo/demo.jpg"
print(inference_detector(model, f'{image}'))