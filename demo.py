import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import argparse
import inspect
import gradio as gr
from retinanet import Retinanet
from utils.config import str2bool


def parse_args():
    parser = argparse.ArgumentParser(description='Launch the Gradio RetinaNet face mask detection demo.')
    parser.add_argument('--config', default='configs/mask_retinanet.yaml',
                        help='專案設定檔路徑')
    parser.add_argument('--weights', default=None,
                        help='推論權重路徑')
    parser.add_argument('--confidence', type=float, default=None,
                        help='bbox 信心度門檻')
    parser.add_argument('--share', type=str2bool, default=False,
                        help='是否建立 Gradio share link')
    parser.add_argument('--server-name', default=None,
                        help='Gradio server name，例如 0.0.0.0')
    parser.add_argument('--server-port', type=int, default=None,
                        help='Gradio server port')
    return parser.parse_args()


args = parse_args()
model_kwargs = {'config_path': args.config}
if args.weights:
    model_kwargs['model_path'] = args.weights
if args.confidence is not None:
    model_kwargs['confidence'] = args.confidence
model = Retinanet(**model_kwargs)

def predict(image):
    return model.detect_image(image)

DESCRIPTION = """
<div class="mask-demo-hero">
  <div class="mask-demo-copy">
    <div class="mask-demo-kicker">RETINANET / RESNET50-FPN</div>
    <h1>口罩配戴狀態偵測</h1>
    <p>多人場景三類別偵測，針對正確配戴、未配戴與配戴不正確狀態輸出 bounding boxes。</p>
  </div>
  <div class="mask-demo-metrics" aria-label="model metrics">
    <div>
      <span>mAP@0.5</span>
      <strong>76.05%</strong>
    </div>
    <div>
      <span>Input</span>
      <strong>600×600</strong>
    </div>
    <div>
      <span>Classes</span>
      <strong>3</strong>
    </div>
  </div>
</div>
"""

ARTICLE = """
<div class="mask-demo-footer">
  <div class="mask-demo-legend">
    <div><i class="swatch swatch-ok"></i><span>with_mask</span><em>正確配戴</em></div>
    <div><i class="swatch swatch-alert"></i><span>without_mask</span><em>未配戴</em></div>
    <div><i class="swatch swatch-warn"></i><span>mask_weared_incorrect</span><em>配戴不正確</em></div>
  </div>
  <div class="mask-demo-spec">
    <span>Face Mask Detection Dataset</span>
    <span>TensorFlow 1.13.2 + Keras 2.1.5</span>
    <span>NVIDIA RTX 2070 8GB</span>
  </div>
</div>
"""

CUSTOM_CSS = """
:root {
  --mask-ink: #17211d;
  --mask-muted: #66716c;
  --mask-line: #d9e0dc;
  --mask-panel: #fbfcfa;
  --mask-canvas: #f2f5f1;
  --mask-teal: #12806a;
  --mask-red: #c84c48;
  --mask-amber: #c9952c;
}

body,
.gradio-container {
  background:
    linear-gradient(180deg, rgba(255,255,255,0.72), rgba(255,255,255,0) 34%),
    var(--mask-canvas) !important;
  color: var(--mask-ink) !important;
}

.gradio-container {
  max-width: 1120px !important;
  margin: 0 auto !important;
  padding: 28px 22px 34px !important;
  font-family: "Noto Sans TC", "Segoe UI", "Microsoft JhengHei", sans-serif !important;
}

.mask-demo-hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 22px;
  align-items: end;
  padding: 26px 28px;
  margin: 0 auto 18px;
  border: 1px solid var(--mask-line);
  border-radius: 18px;
  background: linear-gradient(135deg, #ffffff 0%, #f7faf7 58%, #edf5f1 100%);
  box-shadow: 0 18px 50px rgba(20, 37, 31, 0.10);
}

.mask-demo-kicker {
  margin-bottom: 8px;
  color: var(--mask-teal);
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.08em;
}

.mask-demo-copy h1 {
  margin: 0;
  color: var(--mask-ink);
  font-size: 2.25rem;
  line-height: 1.12;
  letter-spacing: 0;
}

.mask-demo-copy p {
  max-width: 620px;
  margin: 10px 0 0;
  color: var(--mask-muted);
  font-size: 0.98rem;
  line-height: 1.7;
}

.mask-demo-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(94px, 1fr));
  gap: 8px;
  min-width: 330px;
}

.mask-demo-metrics div {
  padding: 13px 14px;
  border: 1px solid rgba(18, 128, 106, 0.16);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.76);
}

.mask-demo-metrics span,
.mask-demo-spec span {
  display: block;
  color: var(--mask-muted);
  font-size: 0.74rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.mask-demo-metrics strong {
  display: block;
  margin-top: 4px;
  color: var(--mask-ink);
  font-size: 1.06rem;
}

.mask-demo-footer {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 18px;
  align-items: center;
  max-width: 1040px;
  margin: 16px auto 0;
  padding: 16px 18px;
  border: 1px solid var(--mask-line);
  border-radius: 14px;
  background: rgba(255,255,255,0.72);
}

.mask-demo-legend {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.mask-demo-legend div {
  display: grid;
  grid-template-columns: 10px minmax(0, auto);
  column-gap: 8px;
  row-gap: 1px;
  align-items: center;
  color: var(--mask-ink);
  font-size: 0.88rem;
  font-weight: 800;
}

.mask-demo-legend em {
  grid-column: 2;
  color: var(--mask-muted);
  font-size: 0.76rem;
  font-style: normal;
  font-weight: 500;
}

.swatch {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.swatch-ok { background: var(--mask-teal); }
.swatch-alert { background: var(--mask-red); }
.swatch-warn { background: var(--mask-amber); }

.mask-demo-spec {
  display: grid;
  gap: 6px;
  text-align: right;
}

.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container .panel,
.gradio-container .block,
.gradio-container .gr-box {
  border-radius: 12px !important;
}

.gradio-container button {
  border: 1px solid rgba(18, 128, 106, 0.24) !important;
  background: #ffffff !important;
  color: var(--mask-ink) !important;
  font-weight: 700 !important;
}

.gradio-container button.primary,
.gradio-container button[class*="primary"] {
  background: var(--mask-teal) !important;
  color: #ffffff !important;
  box-shadow: 0 10px 24px rgba(18, 128, 106, 0.22) !important;
}

@media (max-width: 860px) {
  .mask-demo-hero,
  .mask-demo-footer {
    grid-template-columns: 1fr;
  }

  .mask-demo-metrics,
  .mask-demo-legend {
    grid-template-columns: 1fr;
    min-width: 0;
  }

  .mask-demo-copy h1 {
    font-size: 1.85rem;
  }

  .mask-demo-spec {
    text-align: left;
  }
}
"""

def interface_supports_kwarg(name):
    try:
        return name in inspect.signature(gr.Interface.__init__).parameters
    except (TypeError, ValueError):
        return False


def image_component(label):
    if hasattr(gr, 'Image'):
        return gr.Image(type="pil", label=label)
    return gr.inputs.Image(type="pil", label=label)


interface_kwargs = dict(
    fn=predict,
    inputs=image_component("上傳圖片"),
    outputs=image_component("偵測結果"),
    title=None,
    description=DESCRIPTION,
    article=ARTICLE,
    examples=[
        ["figure/demo_input.jpg"],
    ],
    cache_examples=False,
)

if hasattr(gr, 'themes'):
    interface_kwargs['theme'] = gr.themes.Soft(primary_hue="blue", secondary_hue="sky")

if interface_supports_kwarg('css'):
    interface_kwargs['css'] = CUSTOM_CSS

demo = gr.Interface(**interface_kwargs)

if __name__ == "__main__":
    demo.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)
