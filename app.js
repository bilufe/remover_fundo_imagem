import {
  ImageSegmenter,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const canvas = document.getElementById("canvas");
const fotoOriginalCanvas = document.getElementById("fotoOriginal");
const fOriginalCTX = fotoOriginalCanvas.getContext("2d");
const ctx = canvas.getContext("2d");
const upload = document.getElementById("arquivo");

const vision = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
);

const segmenter = await ImageSegmenter.createFromOptions(vision, {
  baseOptions: {
    modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
  },
  outputConfidenceMasks: true
});

upload.addEventListener("change", e => {
  const img = new Image();
  img.src = URL.createObjectURL(e.target.files[0]);

  img.onload = async () => {
    canvas.width = img.width;
    canvas.height = img.height;

    fotoOriginalCanvas.width = img.width;
    fotoOriginalCanvas.height = img.height;

    ctx.drawImage(img, 0, 0);

    fOriginalCTX.drawImage(img, 0, 0);

    const result = await segmenter.segment(img);
    const mask = result.confidenceMasks[0].getAsFloat32Array();

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    for (let i = 0; i < mask.length; i++) {
      if (mask[i] < 0.5) {
        const p = i * 4;
        data[p] = 255;
        data[p + 1] = 255;
        data[p + 2] = 255;
        data[p + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  };
});
