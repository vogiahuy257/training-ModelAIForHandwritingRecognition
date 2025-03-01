# Training Model AI for Handwriting Recognition

## Yêu cầu hệ thống

Để đảm bảo mô hình chạy mượt mà và đạt hiệu suất tốt nhất, bạn nên sử dụng trình duyệt Google Chrome có hỗ trợ GPU.

### 1. **Cài đặt Google Chrome**
Nếu bạn chưa cài đặt Chrome, hãy tải xuống và cài đặt từ đường dẫn sau:
[Google Chrome Download](https://www.google.com/chrome/)

### 2. **Kích hoạt tăng tốc GPU trong Chrome**
TensorFlow.js có thể tận dụng GPU để tăng tốc độ xử lý mô hình. Để đảm bảo Chrome sử dụng GPU, hãy làm theo các bước sau:

#### ✅ Kiểm tra xem Chrome đã bật GPU chưa
1. Mở Chrome và truy cập **chrome://gpu**
2. Kiểm tra phần **Graphics Feature Status**
   - Nếu các mục như **WebGL, WebGL2, and WebGPU** được đánh dấu là **Hardware accelerated**, nghĩa là GPU đã được bật.
   - Nếu không, tiếp tục với các bước bên dưới.

#### 🔧 Bật tăng tốc GPU thủ công
1. Mở Chrome và nhập **chrome://flags** vào thanh địa chỉ.
2. Tìm kiếm **Override software rendering list** và bật nó (**Enabled**).
3. Tìm **WebGL Draft Extensions** và bật nó (**Enabled**).
4. Nhấn **Relaunch** để khởi động lại trình duyệt.

### 3. **Cài đặt và chạy dự án**
#### 🔹 Cài đặt Node.js và Yarn/NPM
Bạn cần cài đặt [Node.js](https://nodejs.org/) và Yarn/NPM để quản lý dependencies.

```sh
# Cài đặt các dependencies
npm install  # Hoặc yarn install
```

#### 🔹 Chạy ứng dụng
```sh
npm start  # Hoặc yarn start
```

Sau khi chạy lệnh, mở trình duyệt và truy cập `http://localhost:3000/` để sử dụng mô hình AI nhận dạng chữ viết tay.

---

🎯 **Lưu ý:** Nếu bạn gặp lỗi khi sử dụng mô hình AI, hãy đảm bảo rằng Chrome đã bật GPU và kiểm tra lại bằng `chrome://gpu`.

