import * as tf from '@tensorflow/tfjs';

const MOBILENET_MODEL_PATH =
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;

let mobilenet;
async function mobilenetDemo() {
    // 1. load model
    mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
    
    // 2. predict
    const catElement = document.getElementById('cat');
    catElement.style.display = '';
    predict(catElement);
}

/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {

  // 1. imgElement -> pred_tensors 를 익명함수로 정의
  const logits = tf.tidy(() => {
    // 1. image tensor
    let img = tf.browser.fromPixels(imgElement).toFloat();

    // 2. preprocess
    const offset = tf.scalar(127.5);
    img = img.sub(offset).div(offset);
    const imgs = img.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    // 3. prediction
    return mobilenet.predict(imgs);
  });

  // 2. 익명함수를 실행해서 데이터를 뽑는다.
  const values = await logits.data();
  printMsg(values.length + ": " + values[0] + ", " + values[1]);  
}


function printMsg(msg) {
    const demoStatusElement = document.getElementById('status');
    demoStatusElement.innerText += (msg + "\n");    
}

mobilenetDemo();
