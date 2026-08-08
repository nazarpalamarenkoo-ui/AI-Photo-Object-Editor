from enum import Enum


class MLTaskType(str, Enum):
    DETECTION = "detection"

    SEGMENTATION = "segmentation"                    
    SEGMENTATION_PROMPT = "segmentation_prompt"       
    SEGMENTATION_POLYGON = "segmentation_polygon"     
    SEGMENTATION_HYBRID = "segmentation_hybrid"       

    SAM_REMOVE_OBJECT = "sam_remove_object"          
    SAM_REPLACE_OBJECT = "sam_replace_object"         
    DIFFUSION = "diffusion"                          

    REMOVE_OBJECT = "remove_object"                   
    REMOVE_MULTIPLE_OBJECTS = "remove_multiple_objects"  
    REPLACE_OBJECT = "replace_object"                 

    EXTRACT_OBJECT = "extract_object"                 