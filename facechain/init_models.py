import torch
import os
from controlnet_aux import OpenposeDetector
from diffusers import (ControlNetModel, PNDMScheduler, DDIMScheduler, AutoencoderKL, StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline, StableDiffusionPipeline)
from transformers import pipeline as tpipeline

from modelscope import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from face_adapter import FaceAdapter_v1, Face_Extracter_v1


def init_models():
    dtype = torch.float16
    model_dir = snapshot_download(
            'damo/face_chain_control_model', revision='v1.0.1')
    model_dir1 = snapshot_download(
            'ly261666/cv_wanx_style_model', revision='v1.0.3')
    controlnet = [
            ControlNetModel.from_pretrained(
                os.path.join(model_dir,
                             'model_controlnet/control_v11p_sd15_openpose'),
                torch_dtype=dtype),
            ControlNetModel.from_pretrained(
                os.path.join(model_dir1, 'contronet-canny'), torch_dtype=dtype)
        ]
    
    # depth_estimator = tpipeline(
    #         'depth-estimation',
    #         os.path.join(model_dir, 'model_controlnet/dpt-large'))
    
    segmentation_pipeline = pipeline(
            Tasks.image_segmentation,
            'damo/cv_resnet101_image-multiple-human-parsing',
            model_revision='v1.0.1')

    image_face_fusion = pipeline('face_fusion_torch',
                                     model='damo/cv_unet_face_fusion_torch', model_revision='v1.0.3')
    
    openpose = OpenposeDetector.from_pretrained(
            os.path.join(model_dir, 'model_controlnet/ControlNet')).to('cuda')
    
    model_dir = snapshot_download(
            'ly261666/cv_wanx_style_model', revision='v1.0.2')
        
    fr_weight_path = snapshot_download('yucheng1996/FaceChain-FACT', revision='v1.0.0')
    fr_weight_path = os.path.join(fr_weight_path, 'ms1mv2_model_TransFace_S.pt')
        
    fact_model_path = snapshot_download('yucheng1996/FaceChain-FACT', revision='v1.0.0')
    face_adapter_path = os.path.join(fact_model_path, 'adapter_maj_mask_large_new_reg001_faceshuffle_00290001.ckpt')
    face_extracter = Face_Extracter_v1(fr_weight_path=fr_weight_path, fc_weight_path=face_adapter_path)
    
    face_detection = pipeline(
            task=Tasks.face_detection,
            model='damo/cv_ddsar_face-detection_iclr23-damofd',
            model_revision='v1.1')
    face_detection0 = pipeline(task=Tasks.face_detection, model='damo/cv_resnet50_face-detection_retinaface')
    skin_retouching = pipeline(
            'skin-retouching-torch',
            model='damo/cv_unet_skin_retouching_torch',
            model_revision='v1.0.1.1')
    fair_face_attribute_func = pipeline(Tasks.face_attribute_recognition,
            snapshot_download('damo/cv_resnet34_face-attribute-recognition_fairface', revision='v2.0.2'))
    
    face_adapter = FaceAdapter_v1(face_detection0, segmentation_pipeline, face_extracter, face_adapter_path, 'cuda', True)
    face_adapter.delayed_face_condition = 0.0
    
    base_model_path = snapshot_download('YorickHe/majicmixRealistic_v6', revision='v1.0.0')
    base_model_path = os.path.join(base_model_path, 'realistic')
    
    pipe_pose = StableDiffusionControlNetPipeline.from_pretrained(
                base_model_path,
                safety_checker=None,
                controlnet=controlnet[0],
                torch_dtype=dtype).to('cuda')
    pipe_pose.scheduler = PNDMScheduler.from_config(
                pipe_pose.scheduler.config)
    
    pipe_all = StableDiffusionControlNetInpaintPipeline(
            vae=pipe_pose.vae,
            text_encoder=pipe_pose.text_encoder,
            tokenizer=pipe_pose.tokenizer,
            unet=pipe_pose.unet,
            scheduler=pipe_pose.scheduler,
            controlnet=controlnet,
            safety_checker=pipe_pose.safety_checker,
            feature_extractor=pipe_pose.feature_extractor,
            requires_safety_checker=pipe_pose.requires_safety_checker
    ).to('cuda')
    
    face_adapter.set_adapter(pipe_all)
    face_adapter.load_adapter(pipe_all)

    return pipe_pose, pipe_all, face_adapter, openpose, segmentation_pipeline, image_face_fusion, face_detection, skin_retouching, fair_face_attribute_func