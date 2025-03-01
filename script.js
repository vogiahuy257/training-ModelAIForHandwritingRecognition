import {MnistData} from './data.js';

async function showExamples(data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

function getModel() {
    const model = tf.sequential();
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;  

    //inputShape. 
    // Hình dạng của dữ liệu sẽ chuyển vào lớp đầu tiên của mô hình. Trong trường hợp này, ví dụ về
    //MNIST của chúng tôi là ảnh đen trắng 28x28 pixel. Định dạng chuẩn cho dữ liệu hình ảnh là [row, column, depth]
    //vì vậy, ở đây chúng ta muốn định cấu hình hình dạng của [28, 28, 1]. 28 hàng và cột cho số lượng
    //pixel trong mỗi chiều và độ sâu là 1 vì hình ảnh của chúng ta chỉ có 1 kênh màu. Lưu ý rằng chúng ta không chỉ
    //định kích thước lô trong hình dạng dữ liệu nhập. Lớp được thiết kế không phụ thuộc vào kích thước lô để trong quá
    //trình suy luận, bạn có thể truyền một tensor có kích thước lô bất kỳ vào.

    //kernelSize. 
    // Kích thước của các cửa sổ bộ lọc tích chập trượt được áp dụng cho dữ liệu đầu vào. Ở đây, chúng ta
    //đặt kernelSize là 5. Giá trị này chỉ định một cửa sổ tích chập hình vuông có kích thước 5x5.

    //filters. 
    // Số lượng cửa sổ bộ lọc có kích thước kernelSize để áp dụng cho dữ liệu đầu vào. Ở đây, chúng ta sẽ
    //áp dụng 8 bộ lọc cho dữ liệu.

    //strides. "Kích thước bước" của cửa sổ trượt, tức là số lượng pixel mà bộ lọc sẽ dịch chuyển mỗi khi di chuyển
    //qua hình ảnh. Ở đây, chúng tôi xác định các bước là 1, có nghĩa là bộ lọc sẽ trượt trên hình ảnh theo các bước 1 pixel.

    //activation.
    //. Hàm kích hoạt để áp dụng cho dữ liệu sau khi tích chập hoàn tất. 
    // Trong trường hợp này, chúng ta sẽ áp dụng hàm Đơn vị tuyến tính đã chỉnh sửa (ReLU). 
    // Đây là hàm kích hoạt rất phổ biến trong các mô hình học máy.

    //kernelInitializer
    // Phương thức sử dụng để khởi tạo ngẫu nhiên trọng số mô hình, rất quan trọng trong việc huấn luyện động lực học. 
    // Chúng ta sẽ không đi vào thông tin chi tiết về việc khởi chạy ở đây, nhưng VarianceScaling (được sử dụng ở đây) 
    // thường là một lựa chọn tốt về trình khởi tạo.

    // In the first layer of our convolutional neural network we have 
    // to specify the input shape. Then we specify some parameters for 
    // the convolution operation that takes place in this layer.
    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
  
    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averaging.  
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Repeat another conv2d + maxPooling stack. 
    // Note that we have more filters in the convolution.
    model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Bây giờ chúng ta làm phẳng đầu ra từ bộ lọc 2D thành một vectơ 1D để chuẩn bị
    // cho đầu vào vào lớp cuối cùng của chúng ta. Đây là thông lệ phổ biến khi đưa
    // dữ liệu có chiều cao hơn vào lớp đầu ra phân loại cuối cùng.
    model.add(tf.layers.flatten());
  
    // Lớp cuối cùng của chúng ta là một lớp dày đặc có 10 đơn vị đầu ra, mỗi đơn vị cho một
    // lớp đầu ra (tức là 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }));
  
    
    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  
    return model;
  }

  async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;
  
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [
        d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [
        d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
        d.labels
      ];
    });
  
    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks
    });
  }

document.addEventListener('DOMContentLoaded', run);

async function run() {  
    const data = new MnistData();
    
    await data.load();
    await showExamples(data);

    const model = getModel();
    tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
    await train(model, data);
    await model.save('localstorage://mnist-model'); // Lưu mô hình sau khi huấn luyện
    await showAccuracy(model, data);
    await showConfusion(model, data);
    await loadModel();
    document.getElementById('predictButton').addEventListener('click', predictImage);
}

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}


async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});

  labels.dispose();
}

// xử lý frontend
let model;
const imageInput = document.getElementById('imageInput');
const predictButton = document.getElementById('predictButton');
const canvas = document.getElementById('canvas');
const predictionResult = document.getElementById('predictionResult');
// Ẩn các phần tử khi đang tải mô hình
imageInput.style.display = "none";
predictButton.style.display = "none";
canvas.style.display = "none";
predictionResult.style.display = "none";

async function loadModel() {
  const statusElement = document.getElementById('modelStatus');

  try {
      model = await tf.loadLayersModel('localstorage://mnist-model'); // Hoặc đường dẫn mô hình đã lưu
      statusElement.innerText = 'Mô hình đã tải xong!';

      // Hiển thị lại các phần tử sau khi tải xong
      if(model)
      {
        imageInput.style.display = "block";
        predictButton.style.display = "block";
        canvas.style.display = "block";
        predictionResult.style.display = "block";
      }
  } catch (error) {
      statusElement.innerText = 'Lỗi khi tải mô hình!';
      console.error('Lỗi khi tải mô hình:', error);
  }
}


async function predictImage() {
  const input = document.getElementById('imageInput');
  if (!input.files || input.files.length === 0) {
      alert('Vui lòng chọn một hình ảnh.');
      return;
  }

  const file = input.files[0];
  const img = new Image();
  img.src = URL.createObjectURL(file);

  img.onload = async function() {
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');

      // Resize và chuyển đổi ảnh về grayscale
      ctx.drawImage(img, 0, 0, 28, 28);
      const imageData = ctx.getImageData(0, 0, 28, 28);
      const grayscaleImage = [];

      for (let i = 0; i < imageData.data.length; i += 4) {
          const gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
          grayscaleImage.push(gray / 255.0);
      }

      // Chuyển đổi sang tensor với shape phù hợp [1, 28, 28, 1]
      const inputTensor = tf.tensor4d(grayscaleImage, [1, 28, 28, 1]);

      // Dự đoán
      const prediction = model.predict(inputTensor);
      const predictedClass = prediction.argMax(1).dataSync()[0];

      // Hiển thị số dự đoán
      document.getElementById('predictionResult').innerText = `Dự đoán: ${classNames[predictedClass]}`;

      // Giải phóng bộ nhớ
      inputTensor.dispose();
      prediction.dispose();
  };
}


// Chạy khi trang tải xong