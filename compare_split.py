"""데이터 누수 비교: Roboflow 무작위 분할 vs 구간 분할.
같은 설정으로 무작위 분할 모델을 학습하고, 각자의 valid 점수와 처음 보는 데모 영상 검출률을 비교."""
from collections import Counter
from pathlib import Path
import yaml
from ultralytics import YOLO

SRC = Path("dataset_rf").resolve()
names = yaml.safe_load(open(SRC / "data.yaml"))["names"]
RANDOM_YAML = str(SRC / "data_fixed.yaml")
yaml.safe_dump({"path": str(SRC), "train": "train/images", "val": "valid/images",
                "test": "test/images", "names": names}, open(RANDOM_YAML, "w"))
BLOCK_YAML = str(Path("dataset_block/data.yaml").resolve())

m = YOLO("yolov8n.pt")
m.train(data=RANDOM_YAML, epochs=50, imgsz=640, seed=0, device="mps", name="random", exist_ok=True)
models = {"random": Path(m.trainer.save_dir) / "weights/best.pt", "block": Path("runs/detect/block/weights/best.pt")}
yamls = {"random": RANDOM_YAML, "block": BLOCK_YAML}

def demo_rate(weights):
    hit, n = Counter(), 0
    for r in YOLO(weights).predict("demo/demo_5s.mp4", conf=0.25, device="mps", stream=True, verbose=False):
        n += 1
        hit.update({r.names[int(c)] for c in r.boxes.cls})
    return {k: hit[k] / n for k in names}

lines = [f"{'split':<8}{'valid mAP50':>12}{'valid mAP50-95':>16}   demo 검출률 " + " / ".join(names)]
for k in ["random", "block"]:
    v = YOLO(models[k]).val(data=yamls[k], device="mps", name=f"{k}_val", exist_ok=True)
    d = demo_rate(models[k])
    lines.append(f"{k:<8}{v.box.map50:>12.3f}{v.box.map:>16.3f}   " + " / ".join(f"{d[c]:.0%}" for c in names))
report = "\n".join(lines)
print("\n" + report)
Path("runs/detect/compare_split.txt").write_text(report)
