import "./styles.css";

import * as tf from "@tensorflow/tfjs";
import * as Plotly from "plotly.js-dist";
import * as tfvis from "@tensorflow/tfjs-vis";

const perceptron = ({ x, w, bias }) => {
  const product = tf.dot(x, w).dataSync()[0];
  return product + bias < 0 ? 0 : 1;
};

const sigmoidPerceptron = ({ x, w, bias }) => {
  const product = tf.dot(x, w).dataSync()[0];
  return tf.sigmoid(product + bias).dataSync()[0];
};

const oneHot = (val, categoryCount) =>
  Array.from(tf.oneHot(val, categoryCount).dataSync());

const renderActivationFunction = (xs, ys, title, container) => {
  const data = [
    {
      x: xs,
      y: ys,
      line: { shape: "spline", color: "royalblue" },
      mode: "lines"
    }
  ];

  Plotly.newPlot(container, data, {
    title: title
  });
};

const renderSigmoid = () => {
  const xs = [...Array(20).keys()].map(x => x - 10);
  const ys = tf.sigmoid(xs).dataSync();

  renderActivationFunction(xs, ys, "Sigmoid", "sigmoid-cont");
};

const renderReLU = () => {
  const xs = [...Array(20).keys()].map(x => x - 10);
  const ys = tf.relu(xs).dataSync();

  renderActivationFunction(xs, ys, "ReLU", "relu-cont");
};

const renderLeakyReLU = () => {
  const xs = [...Array(20).keys()].map(x => x - 10);
  const ys = tf.leakyRelu(xs).dataSync();

  renderActivationFunction(xs, ys, "Leaky ReLU", "leaky-relu-cont");
};

const renderLayer = (model, layerName, container) => {
  tfvis.show.layer(
    document.getElementById(container),
    model.getLayer(layerName)
  );
};

const run = async () => {
  console.log(
    perceptron({
      x: [0, 1],
      w: [0.5, 0.9],
      bias: -0.5
    })
  );

  console.log(
    sigmoidPerceptron({
      x: [0.6, 0.9],
      w: [0.5, 0.9],
      bias: -0.5
    })
  );

  renderSigmoid();
  renderReLU();
  renderLeakyReLU();

  const X = tf.tensor2d([
    // pink, small
    [0.1, 0.1],
    [0.3, 0.3],
    [0.5, 0.6],
    [0.4, 0.8],
    [0.9, 0.1],
    [0.75, 0.4],
    [0.75, 0.9],
    [0.6, 0.9],
    [0.6, 0.75]
  ]);

  // 0 - no buy, 1 - buy
  const y = tf.tensor([0, 0, 1, 1, 0, 0, 1, 1, 1].map(y => oneHot(y, 2)));

  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      name: "hidden-layer",
      inputShape: [2],
      units: 3,
      activation: "relu"
    })
  );

  model.add(
    tf.layers.dense({
      units: 2,
      activation: "softmax"
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.1),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"]
  });

  const lossContainer = document.getElementById("loss-cont");

  await model.fit(X, y, {
    shuffle: true,
    epochs: 20,
    validationSplit: 0.1,
    callbacks: tfvis.show.fitCallbacks(
      lossContainer,
      ["loss", "val_loss", "acc", "val_acc"],
      {
        callbacks: ["onEpochEnd"]
      }
    )
  });

  const hiddenLayer = model.getLayer("hidden-layer");
  const [weights, biases] = hiddenLayer.getWeights(true);
  console.log(weights.shape);
  console.log(biases.shape);

  renderLayer(model, "hidden-layer", "hidden-layer-container");

  const predProb = model.predict(tf.tensor2d([[0.1, 0.6]])).dataSync();

  console.log(predProb);
};

if (document.readyState !== "loading") {
  run();
} else {
  document.addEventListener("DOMContentLoaded", run);
}
