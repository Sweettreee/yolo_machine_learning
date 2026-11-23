"""학습 전 기본 yolov8n(COCO)이 우리 물체를 얼마나 인식하는지 측정 (보고서의 학습 전후 비교용)."""
from collections import Counter
from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
CUP = {"cup", "wine glass"}

for folder in ["glasscup", "watch", "cap", "together"]:
    imgs = sorted(Path("frames", folder).glob("*.jpg"))
    seen, hit = Counter(), 0
    for r in model.predict(imgs, conf=0.25, verbose=False, stream=True):
        names = {r.names[int(c)] for c in r.boxes.cls}
        seen.update(names)
        hit += bool(names & CUP)
    print(f"\n[{folder}] {len(imgs)}장 | cup/wine glass 검출: {hit}장 ({hit/len(imgs):.0%})")
    print("  검출된 COCO 클래스(이미지 수):", dict(seen.most_common(6)))

# 보고서용 예시 이미지: runs/detect/baseline/
model.predict(sorted(Path("frames/glasscup").glob("*.jpg"))[::20], conf=0.25,
              save=True, name="baseline", exist_ok=True, verbose=False)

# 데모 영상(=실제 과제 상황)에서 프레임별 cup 검출률
frames = [bool({r.names[int(c)] for c in r.boxes.cls} & CUP)
          for r in model.predict("demo/demo_5s.mp4", conf=0.25, verbose=False, stream=True)]
print(f"\n[demo_5s.mp4] {len(frames)}프레임 | cup/wine glass 검출: {sum(frames)}프레임 ({sum(frames)/len(frames):.0%})")
