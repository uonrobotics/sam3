"""Local, serialized reference-image SAM3 inference. No dataset batch or Gemini."""

import argparse
import importlib.util
import io
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = Path("/media/uon/data1/Dataset_2026/review/real_v1")
DEFAULT_METADATA = DEFAULT_REFERENCE_ROOT.parent / "2026_real_world_test_objects_metadata.csv"
DEFAULT_SCENE = "Home/Uon_1F_Home/floor"


def checkpoint_default():
    candidates = sorted((Path.home() / ".cache/huggingface/hub/models--facebook--sam3/snapshots").glob("*/sam3.pt"))
    return candidates[0] if candidates else ROOT / "checkpoints/sam3.pt"


def scene_index(root, scene):
    relative = Path(scene)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("reference_scene must be relative to reference_root")
    index = (root / relative / "reference/reference_index.json").resolve()
    if not index.is_relative_to(root.resolve()):
        raise ValueError("Reference scene escapes reference_root")
    return index


class ReferenceEngine:
    def __init__(self, args):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.args = args
        self.cache = {}
        self.inference_calls = 0
        started = time.perf_counter()
        model = build_sam3_image_model(
            checkpoint_path=str(args.checkpoint),
            load_from_HF=False,
            bpe_path=str(ROOT / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        )
        self.processor = Sam3Processor(model, resolution=1008, confidence_threshold=args.confidence_threshold)
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started

    def reference_bundle(self, class_id, scene):
        from sam3.reference_exemplar import load_reference_bundle

        index = scene_index(self.args.reference_root, scene)
        key = (str(index), class_id)
        if key not in self.cache:
            self.cache[key] = load_reference_bundle(
                class_id,
                reference_index_path=index,
                metadata_path=self.args.metadata,
                tile_width=1008,
                tile_height=504,
                input_size=1008,
            )
        return self.cache[key]

    def reference_info(self, class_id, scene):
        """Validate reference files without a model prediction or image capture."""
        bundle = self.reference_bundle(class_id, scene)
        return dict(
            success=True,
            class_id=class_id,
            reference_scene=scene,
            reference_path=str(bundle.reference_path),
            identity=bundle.identity.metadata(),
            inference_performed=False,
        )

    def predict(self, rgb, class_id, scene):
        import numpy as np
        import torch
        from PIL import Image
        from sam3.reference_exemplar import letterbox, map_boxes_to_original, restore_masks

        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
            raise ValueError("rgb must be uint8 HxWx3")
        bundle = self.reference_bundle(class_id, scene)
        # Require the canonical class name, never a dataset-local obj_NNN join.
        if bundle.identity.applied_key != class_id:
            raise ValueError("class_id must be the canonical reference object name")
        target, transform = letterbox(Image.fromarray(rgb), 1008, 504)
        canvas = Image.new("RGB", (1008, 1008))
        canvas.paste(bundle.reference_tile, (0, 0))
        canvas.paste(target, (0, 504))
        started = time.perf_counter()
        self.inference_calls += 1
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(canvas)
            output = self.processor.add_geometric_prompt(state=state, box=bundle.normalized_prompt, label=True)
        keep = (output["boxes"][:, 1] + output["boxes"][:, 3]) / 2 >= 504
        boxes = map_boxes_to_original(output["boxes"][keep].float(), transform, tile_y=504).detach().cpu().numpy()
        masks = restore_masks(output["masks"][keep], transform, 504, 504, 1008)
        masks = np.stack(masks)[:, None] if masks else np.zeros((0, 1, *rgb.shape[:2]), bool)
        scores = output["scores"][keep].float().detach().cpu().numpy()
        metadata = dict(
            class_id=class_id,
            reference_scene=scene,
            reference_image=str(bundle.reference_path),
            reference_index=str(scene_index(self.args.reference_root, scene)),
            reference_identity=bundle.identity.metadata(),
            confidence_threshold=self.args.confidence_threshold,
            inference_seconds=time.perf_counter() - started,
            inference_calls=self.inference_calls,
        )
        return dict(masks=masks, boxes_xyxy=boxes, scores=scores, metadata=np.array(json.dumps(metadata)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=13347)
    parser.add_argument("--checkpoint", type=Path, default=checkpoint_default())
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("confidence-threshold must be in [0,1]")
    missing = [
        str(p)
        for p in (args.checkpoint, args.metadata, ROOT / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
        if not p.is_file()
    ]
    missing += [
        m
        for m in ("torch", "numpy", "PIL", "sam3", "einops", "psutil", "cc_torch")
        if importlib.util.find_spec(m) is None
    ]
    if not args.reference_root.is_dir():
        missing.append(str(args.reference_root))
    if missing:
        sys.exit("SAM3 prerequisites missing: " + ", ".join(missing))
    if args.check:
        print("SAM3 check: OK\nSource: " + str(ROOT), flush=True)
        return
    engine = ReferenceEngine(args)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, data, content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/reference":
                try:
                    query = parse_qs(parsed.query)
                    result = engine.reference_info(query["class_id"][0], query.get("scene", [DEFAULT_SCENE])[0])
                    self.reply(200, json.dumps(result).encode())
                except (ValueError, KeyError, OSError) as error:
                    self.reply(400, json.dumps(dict(error=str(error))).encode())
                return
            if self.path != "/health":
                self.reply(404, b"{}")
                return
            self.reply(
                200,
                json.dumps(
                    dict(
                        status="ready",
                        backend="sam3_reference",
                        source=str(ROOT),
                        load_seconds=engine.load_seconds,
                        inference_calls=engine.inference_calls,
                    )
                ).encode(),
            )

        def do_POST(self):
            if self.path != "/segment-reference":
                self.reply(404, b"{}")
                return
            try:
                import numpy as np

                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 40 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                with np.load(io.BytesIO(self.rfile.read(length)), allow_pickle=False) as packet:
                    result = engine.predict(packet["rgb"], str(packet["class_id"]), str(packet["reference_scene"]))
                stream = io.BytesIO()
                np.savez_compressed(stream, **result)
                self.reply(200, stream.getvalue(), "application/octet-stream")
            except (ValueError, KeyError, FileNotFoundError) as error:
                self.reply(400, json.dumps(dict(error=str(error))).encode())
            except Exception as error:
                traceback.print_exc()
                self.reply(500, json.dumps(dict(error=str(error))).encode())

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print("SAM3_READY " + json.dumps(dict(port=args.port, source=str(ROOT))), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
