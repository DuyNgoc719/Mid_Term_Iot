import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import time

# Đường dẫn đến dữ liệu đã tiền xử lý và mô hình
processed_data_path = '/home/ubuntu/bidmc_project/data/processed'
model_path = '/home/ubuntu/bidmc_project/models'
figures_path = '/home/ubuntu/bidmc_project/code/figures'

# Tạo thư mục nếu chưa tồn tại
os.makedirs(model_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# Tải mô hình CVAE đã định nghĩa
from cvae_model import CVAE, build_encoder, build_decoder

# Tải dữ liệu đã tiền xử lý
print("Đang tải dữ liệu đã tiền xử lý...")
X_train = np.load(os.path.join(processed_data_path, 'ppg_train.npy'))
X_test = np.load(os.path.join(processed_data_path, 'ppg_test.npy'))
hr_train = np.load(os.path.join(processed_data_path, 'hr_train.npy'))
hr_test = np.load(os.path.join(processed_data_path, 'hr_test.npy'))
rr_train = np.load(os.path.join(processed_data_path, 'rr_train.npy'))
rr_test = np.load(os.path.join(processed_data_path, 'rr_test.npy'))

print(f"Kích thước dữ liệu huấn luyện: {X_train.shape}")
print(f"Kích thước dữ liệu kiểm thử: {X_test.shape}")

# Tham số mô hình
input_dim = X_train.shape[1]  # Độ dài đoạn tín hiệu PPG
condition_dim = 2  # HR và RR
latent_dim = 32  # Kích thước không gian tiềm ẩn
hidden_units = [256, 128, 64]  # Số đơn vị ẩn trong các lớp
batch_size = 64
epochs = 50
learning_rate = 0.001

# Chuẩn bị dữ liệu điều kiện
condition_train = np.column_stack((hr_train, rr_train))
condition_test = np.column_stack((hr_test, rr_test))

# Xây dựng mô hình
print("Đang xây dựng mô hình CVAE...")
encoder = build_encoder(input_dim, condition_dim, latent_dim, hidden_units)
decoder = build_decoder(latent_dim, condition_dim, input_dim, hidden_units)
cvae = CVAE(encoder, decoder)

# Biên dịch mô hình
cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

# Tạo TensorBoard callback
log_dir = os.path.join(model_path, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Tạo ModelCheckpoint callback
checkpoint_path = os.path.join(model_path, "cvae_checkpoint.weights.h5")
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True
)

# Tạo EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=True
)

# Huấn luyện mô hình
print("\nBắt đầu huấn luyện mô hình...")
start_time = time.time()

history = cvae.fit(
    x=[X_train, condition_train],
    y=None,  # y là None vì CVAE tự tính toán loss
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([X_test, condition_test], None),
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback]
)

training_time = time.time() - start_time
print(f"\nHuấn luyện hoàn tất trong {training_time:.2f} giây.")

# Lưu mô hình
cvae.save_weights(os.path.join(model_path, 'cvae_final.weights.h5'))
print(f"Đã lưu mô hình tại: {os.path.join(model_path, 'cvae_final.weights.h5')}")

# Vẽ biểu đồ quá trình huấn luyện
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history.history['reconstruction_loss'])
plt.title('Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(history.history['kl_loss'])
plt.title('KL Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'training_history.png'))
plt.close()

# Lưu thông tin huấn luyện
with open(os.path.join(model_path, 'training_info.txt'), 'w') as f:
    f.write("THÔNG TIN HUẤN LUYỆN MÔ HÌNH CVAE\n")
    f.write("=================================\n\n")
    
    f.write("Tham số huấn luyện:\n")
    f.write(f"- Kích thước batch: {batch_size}\n")
    f.write(f"- Số epoch: {epochs}\n")
    f.write(f"- Tốc độ học: {learning_rate}\n\n")
    
    f.write("Kết quả huấn luyện:\n")
    f.write(f"- Số epoch đã huấn luyện: {len(history.history['loss'])}\n")
    f.write(f"- Loss cuối cùng (train): {history.history['loss'][-1]:.4f}\n")
    f.write(f"- Loss cuối cùng (validation): {history.history['val_loss'][-1]:.4f}\n")
    f.write(f"- Reconstruction loss cuối cùng: {history.history['reconstruction_loss'][-1]:.4f}\n")
    f.write(f"- KL loss cuối cùng: {history.history['kl_loss'][-1]:.4f}\n")
    f.write(f"- Thời gian huấn luyện: {training_time:.2f} giây\n\n")
    
    f.write("Đường dẫn đến mô hình đã lưu:\n")
    f.write(f"- Mô hình cuối cùng: {os.path.join(model_path, 'cvae_final.weights.h5')}\n")
    f.write(f"- Mô hình checkpoint: {checkpoint_path}\n")

print("\nQuá trình huấn luyện đã hoàn tất. Thông tin huấn luyện đã được lưu vào file training_info.txt.")
