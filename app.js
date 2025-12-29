import {
  ImageSegmenter,
  FaceLandmarker,
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

const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
  baseOptions: {
    modelAssetPath:
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
  },
  runningMode: "IMAGE",
  numFaces: 2
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
    await verificaEspecificacoes(img);
  };

});

async function verificaEspecificacoes(img) {

  const exibeMensagem = document.getElementById('info-imagem');
  exibeMensagem.innerHTML = "";

  if (img.width < 400 || img.height < 500) {
    exibeMensagem.innerHTML += `<p style="color:red;font-weight:bold; margin: 2vh">
      ERRO: Resolução insuficiente para 3x4 a 300 DPI
    </p>`;
  } 

  const faceResult = await faceLandmarker.detect(img);
  const faces = faceResult.faceLandmarks || [];

  if (faces.length === 0) {
    exibeMensagem.innerHTML += `<p style="color:red;font-weight:bold; margin: 2vh">
      ERRO: Nenhum rosto foi detectado.
    </p>`;
  } else if (faces.length > 1) {
    exibeMensagem.innerHTML += `<p style="color:red;font-weight:bold; margin: 2vh">
      ERRO: A imagem deve conter apenas um rosto.
    </p>`;
  } 

  const avgBackgroundColor = getAverageBackgroundColor(img);

  if (!isNearWhite(avgBackgroundColor)) {
    exibeMensagem.innerHTML += `<p style="color:red;font-weight:bold; margin: 2vh">
      ERRO: O fundo da imagem deve ser branco.
    </p>`;
  }


  // Nesta parte do código, vai ser verificado se o rosto ocupa a maior parte da foto, como é esperado para foto
  // que será utilizada para documento
  const faceLandmarks = faces[0]; // já validado que há exatamente 1 rosto
  const box = calcularBoundingBox(faceLandmarks, img.width, img.height);

  const faceArea = box.width * box.height;
  const imageArea = img.width * img.height;

  const areaRatio = faceArea / imageArea;
  const heightRatio = box.height / img.height;

  // Face ocupa pouca área → mostra corpo demais
  if (heightRatio < 0.30 || areaRatio < 0.10) {
    exibeMensagem.innerHTML += `<p style="color:red;font-weight:bold;margin:2vh">
    ERRO: O rosto está muito distante. 
    A foto mostra mais que o rosto, fora do padrão de documento.
  </p>`;
  } 


  // Face grande demais → muito próxima ou cortada
  if (heightRatio > 0.50 || areaRatio > 0.30) {
    exibeMensagem.innerHTML += `<p style="color:red;font-weight:bold;margin:2vh">
    ERRO: O rosto está muito próximo ou parcialmente cortado.
  </p>`;
  }


}


function getAverageBackgroundColor(source) {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = source.width;
  tempCanvas.height = source.height;

  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(source, 0, 0);

  const borderSize = 10;
  let r = 0, g = 0, b = 0, count = 0;

  const regions = [
    [0, 0, source.width, borderSize],
    [0, source.height - borderSize, source.width, borderSize],
    [0, 0, borderSize, source.height],
    [source.width - borderSize, 0, borderSize, source.height]
  ];

  for (const [x, y, w, h] of regions) {
    const data = tempCtx.getImageData(x, y, w, h).data;
    for (let i = 0; i < data.length; i += 4) {
      r += data[i];
      g += data[i + 1];
      b += data[i + 2];
      count++;
    }
  }

  return { r: r / count, g: g / count, b: b / count };
}


// Função para verificar se a cor é "quase branca"
function isNearWhite({ r, g, b }) {
  // Branco quente: r,g,b altos, branco frio: azul um pouco mais alto
  return (
    r > 200 && g > 200 && b > 200 && // todos altos
    Math.abs(r - g) < 30 && Math.abs(g - b) < 40 // não muito colorido
  );
}


// Função para calcular o quanto do rosto faz parte da imagem
// Importante para saber se a imagem carregada foge do padrão para documentos
function calcularBoundingBox(landmarks, imgWidth, imgHeight) {
  let minX = Infinity, minY = Infinity;
  let maxX = -Infinity, maxY = -Infinity;

  for (const p of landmarks) {
    const x = p.x * imgWidth;
    const y = p.y * imgHeight;

    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY
  };
}


// Evento ao clicar no título da imagem manipulada
document.getElementById('tituloFundoRemovido').addEventListener('click', () => {
  const ffrem = document.getElementById('canvas');
  if (ffrem.style.display === 'none') {
    ffrem.style.display = 'table';
  } else {
    ffrem.style.display = 'none';
  }
});