import asyncio
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from io import BytesIO

from app.ml.experiment_tracker import ExperimentTracker, get_tracker

class SAM2Segmentor:
    
    """
    SAM 2 Segmentor
    
    Wrapper around Meta SAM2 for instance segmentation.
    Supports:
        1. Automatic segmentation (everything mode — no prompts required)
        2. Point/bounding-box prompt segmentation
        3. Async interface + MLflow tracking
    """
    
    def __init__(
        self,
        model_path: str = 'weights/sam2.1_hiera_s.pt',
        device: str = 'cpu',
        tracker: Optional[ExperimentTracker] = None
    ):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            
            self.device = device
            self.tracker = tracker or get_tracker()
            self.segmentor = build_sam2(
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            model_path,
            device=device)
            
            # Predictor for point/bbox prompts
            self.predictor = SAM2ImagePredictor(self.segmentor)
            
            # Auto mask generator — without prompts
            self.auto_generator = SAM2AutomaticMaskGenerator(
                model = self.segmentor,
                points_per_side = 32,
                pred_iou_thresh = 0.88,
                stability_score_thresh = 0.92,
                min_mask_region_area = 100,
            )
            print(f"SAM2 loaded on {device}")
            
        except ImportError as e:
            raise RuntimeError(
                f"sam2 not installed: {e}. Please install the SAM2 package")
            
    async def segment_auto(
        self,
        image_bytes: bytes,
        track_metrics: bool = True
    ) -> Dict:
        
        """
        Automatic segmentation of the entire image (without prompts).

        Returns:
            Dict:
            - segments: List[Dict] with the following keys:
            {
                'mask_id': int
                'bbox': {'x1', 'y1', 'x2', 'y2'}
                'area': int  (in pixels)
                'stability_score': float
                'mask_bytes': bytes  (PNG, binary mask)
            }
            - metrics: Dict
        """
        start_time = time.time()
        
        segments = await asyncio.to_thread(
            self._segment_auto_sync, image_bytes
        )

        inference_time = (time.time() - start_time) * 1000
        metrics = self._calculate_metrics(segments, inference_time)

        if track_metrics:
            await self._track_metrics(metrics)

        return {"segments": segments, "metrics": metrics}
    
    def _segment_auto_sync(self, image_bytes: bytes) -> List[Dict]:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)

        masks = self.auto_generator.generate(img_array)

        segments = []
        for idx, mask_data in enumerate(masks):
            mask = mask_data["segmentation"].astype(np.uint8) * 255
            bbox_xywh = mask_data["bbox"]  # SAM returns [x, y, w, h]

            x, y, w, h = [int(v) for v in bbox_xywh]
            bbox = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}

            # Convert mask to PNG bytes
            mask_img = Image.fromarray(mask, mode="L")
            buf = BytesIO()
            mask_img.save(buf, format="PNG")
            mask_bytes = buf.getvalue()

            segments.append({
                "mask_id": idx,
                "bbox": bbox,
                "area": int(mask_data["area"]),
                "stability_score": float(mask_data["stability_score"]),
                "predicted_iou": float(mask_data["predicted_iou"]),
                "mask_bytes": mask_bytes,
            })

        # Sort segments by area (largest first)
        segments.sort(key=lambda s: s["area"], reverse=True)
        return segments
    
    async def segment_with_prompt(
        self,
        image_bytes: bytes,
        point_coords: Optional[List[Tuple[int, int]]] = None,
        point_labels: Optional[List[int]] = None,
        bbox: Optional[Dict[str, int]] = None,
        track_metrics: bool = True
    ) -> Dict:
        
        """
        Prompt-based segmentation (points or bounding box).

        Args:
            point_coords: List of points [(x, y), ...]
            point_labels: 1 = foreground, 0 = background for each point
            bbox: {'x1', 'y1', 'x2', 'y2'} — if provided, used as a prompt

        Returns:
            Dict:
            - segments: List[Dict] (typically 1–3 masks)
            - metrics: Dict
        """
        start_time = time.time()

        segments = await asyncio.to_thread(
            self._segment_prompt_sync,
            image_bytes, point_coords, point_labels, bbox
        )

        inference_time = (time.time() - start_time) * 1000
        metrics = self._calculate_metrics(segments, inference_time)

        if track_metrics:
            await self._track_metrics(metrics)

        return {"segments": segments, "metrics": metrics}
    
    def _segment_prompt_sync(
        self,
        image_bytes: bytes,
        point_coords: Optional[List[Tuple[int, int]]],
        point_labels: Optional[List[int]],
        bbox: Optional[Dict[str, int]]
    ) -> List[Dict]:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(img)

        self.predictor.set_image(img_array)

        np_points = np.array(point_coords) if point_coords else None
        np_labels = np.array(point_labels) if point_labels else None
        np_bbox = None
        if bbox:
            np_bbox = np.array([
                bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            ])

        masks, scores, _ = self.predictor.predict(
            point_coords=np_points,
            point_labels=np_labels,
            box=np_bbox,
            multimask_output=True,  # return 3 masks for each prompt
        )

        segments = []
        for idx, (mask, score) in enumerate(zip(masks, scores)):
            mask_u8 = mask.astype(np.uint8) * 255
            ys, xs = np.where(mask_u8 > 0)

            if len(xs) == 0:
                continue

            seg_bbox = {
                "x1": int(xs.min()), "y1": int(ys.min()),
                "x2": int(xs.max()), "y2": int(ys.max()),
            }
            buf = BytesIO()
            Image.fromarray(mask_u8, mode="L").save(buf, format="PNG")

            segments.append({
                "mask_id": idx,
                "bbox": seg_bbox,
                "area": int(mask_u8.sum() // 255),
                "stability_score": float(score),
                "predicted_iou": float(score),
                "mask_bytes": buf.getvalue(),
            })

        segments.sort(key=lambda s: s["stability_score"], reverse=True)
        return segments
    
    def _calculate_metrics(self, segments: List[Dict], inference_time_ms: float) -> Dict:
        if not segments:
            return {
                "num_segments": 0,
                "avg_stability": 0.0,
                "inference_time_ms": inference_time_ms,
            }
        avg_stability = sum(s["stability_score"] for s in segments) / len(segments)
        return {
            "num_segments": len(segments),
            "avg_stability": avg_stability,
            "inference_time_ms": inference_time_ms,
            "total_area_px": sum(s["area"] for s in segments),
        }

    async def _track_metrics(self, metrics: Dict) -> None:
        def log_sync():
            self.tracker.log_metrics({
                "sam2_num_segments": metrics["num_segments"],
                "sam2_avg_stability": metrics["avg_stability"],
                "sam2_inference_ms": metrics["inference_time_ms"],
            })
        await asyncio.to_thread(log_sync)


_segmentor_instance = None

def get_segmentor(
    model_path: str = "weights/sam2.1_hiera_s.pt",
    device: str = "cpu",
    tracker: Optional[ExperimentTracker] = None
) -> SAM2Segmentor:
    global _segmentor_instance
    if _segmentor_instance is None:
        _segmentor_instance = SAM2Segmentor(model_path, device, tracker)
    return _segmentor_instance