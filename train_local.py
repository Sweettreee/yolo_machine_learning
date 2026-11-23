"""train_yolo.ipynb의 본 학습 과정을 맥(Apple GPU, MPS)에서 실행.
구간 분할 → yolov8n 전이학습 → test 성능 → 데모 영상 추적+궤적."""
import os, re, random, shutil
from collections import defaultdict, Counter
from pathlib import Path

import cv2, numpy as np, yaml
from ultralytics import YOLO

BLOCK, EPOCHS, DEVICE = 10, 50, "mps"
SRC, DST = Path("dataset_rf"), Path("dataset_block")
names = yaml.safe_load(open(SRC / "data.yaml"))["names"]

# 1. 구간 분할: 영상별 연속 10프레임(약 3초)을 한 묶음으로 train/valid/test 배정 (이웃 프레임 누수 방지)
blocks = defaultdict(list)
for p in SRC.glob("*/images/*.jpg"):
    m = re.match(r"([a-z]+)_(\d+)_jpg", p.name)
    assert m, f"예상과 다른 파일명: {p.name}"
    blocks[(m[1], int(m[2]) // BLOCK)].append(p)

rng, split_of = random.Random(0), {}
for video in sorted({v for v, _ in blocks}):
    keys = sorted(k for k in blocks if k[0] == video)
    rng.shuffle(keys)
    for i, k in enumerate(keys):
        split_of[k] = "train" if i < 0.7 * len(keys) else "valid" if i < 0.9 * len(keys) else "test"

shutil.rmtree(DST, ignore_errors=True)
count = Counter()
for k, imgs in blocks.items():
    s = split_of[k]
    for p in imgs:
        for f, sub in [(p, "images"), (p.parent.parent / "labels" / (p.stem + ".txt"), "labels")]:
            (DST / s / sub).mkdir(parents=True, exist_ok=True)
            if f.exists():
                shutil.copy(f, DST / s / sub / f.name)
        count[s] += 1
DATA = str((DST / "data.yaml").resolve())
yaml.safe_dump({"path": str(DST.resolve()), "train": "train/images", "val": "valid/images",
                "test": "test/images", "names": names}, open(DATA, "w"))
print("구간 분할:", dict(count), flush=True)

# 2. 학습 (사전학습 yolov8n = 전이학습)
model = YOLO("yolov8n.pt")
model.train(data=DATA, epochs=EPOCHS, imgsz=640, seed=0, device=DEVICE, name="block", exist_ok=True)
RUN = Path(model.trainer.save_dir)
BEST = RUN / "weights" / "best.pt"

# 3. test 세트 성능
m = YOLO(BEST).val(data=DATA, split="test", device=DEVICE, name="block_test", exist_ok=True)
lines = [f"{'class':<8}{'mAP50':>8}{'mAP50-95':>10}"]
lines += [f"{names[c]:<8}{m.box.ap50[i]:>8.3f}{m.box.ap[i]:>10.3f}" for i, c in enumerate(m.box.ap_class_index)]
lines.append(f"{'all':<8}{m.box.map50:>8.3f}{m.box.map:>10.3f}")

# 4. 데모 영상 추적 + 이동 궤적
def track_video(weights, src, dst, conf=0.25):
    model = YOLO(weights)
    cap = cv2.VideoCapture(src)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter("_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trail, hit, n = defaultdict(list), Counter(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf, device=DEVICE, verbose=False)[0]
        n += 1
        hit.update({r.names[c] for c in r.boxes.cls.int().tolist()})
        img = r.plot(line_width=3)
        if r.boxes.id is not None:
            for (x, y, _, _), tid in zip(r.boxes.xywh.tolist(), r.boxes.id.int().tolist()):
                trail[tid] = (trail[tid] + [(int(x), int(y))])[-90:]   # 최근 90프레임(1.5초) 궤적
                cv2.polylines(img, [np.array(trail[tid], np.int32)], False, (0, 255, 255), 4)
        out.write(img)
    cap.release(); out.release()
    os.system(f"ffmpeg -loglevel error -y -i _raw.mp4 -c:v libx264 -pix_fmt yuv420p {dst} && rm _raw.mp4")
    res = [f"{dst}: {n}프레임, track ID {len(trail)}개 (3개에 가까울수록 추적 안정)"]
    res += [f"  {k:<12} 검출 프레임 {v}/{n} ({v/n:.0%})" for k, v in hit.most_common()]
    return res

lines += [""] + track_video(BEST, "demo/demo_5s.mp4", "demo/demo_tracked.mp4")
report = "\n".join(lines)
print("\n" + report)
(RUN / "summary.txt").write_text(report)
shutil.copy(BEST, "best.pt")
