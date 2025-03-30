import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

# Đường dẫn đến dữ liệu đã tiền xử lý
processed_data_path = '/home/ubuntu/bidmc_project/data/processed'
model_path = '/home/ubuntu/bidmc_project/models'
figures_path = '/home/ubuntu/bidmc_project/code/figures'

# Tạo thư mục nếu chưa tồn tại
os.makedirs(model_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

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
epochs = 20
learning_rate = 0.001

# Chuẩn bị dữ liệu điều kiện
condition_train = np.column_stack((hr_train, rr_train))
condition_test = np.column_stack((hr_test, rr_test))

# Định nghĩa mô hình CVAE đơn giản hóa
class SimplifiedCVAE(Model):
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_units):
        super(SimplifiedCVAE, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Dense(hidden_units[0], activation='relu'),
            layers.Dense(hidden_units[1], activation='relu'),
            layers.Dense(hidden_units[2], activation='relu'),
            layers.Dense(latent_dim * 2)  # Mean và log_var
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(hidden_units[2], activation='relu'),
            layers.Dense(hidden_units[1], activation='relu'),
            layers.Dense(hidden_units[0], activation='relu'),
            layers.Dense(input_dim, activation='tanh')
        ])
        
        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def encode(self, inputs):
        # Kết hợp tín hiệu PPG và điều kiện
        x, condition = inputs
        combined_input = tf.concat([x, condition], axis=1)
        
        # Encoder
        h = self.encoder(combined_input)
        
        # Tách mean và log_var
        z_mean, z_log_var = tf.split(h, num_or_size_splits=2, axis=1)
        
        # Lấy mẫu từ không gian tiềm ẩn
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        return z_mean, z_log_var, z
    
    def decode(self, inputs):
        # Kết hợp vector tiềm ẩn và điều kiện
        z, condition = inputs
        combined_input = tf.concat([z, condition], axis=1)
        
        # Decoder
        reconstruction = self.decoder(combined_input)
        
        return reconstruction
    
    def call(self, inputs):
        x, condition = inputs
        z_mean, z_log_var, z = self.encode([x, condition])
        reconstruction = self.decode([z, condition])
        return reconstruction
    
    def train_step(self, data):
        if isinstance(data, tuple):
            x, condition = data
        else:
            raise ValueError("Expected a tuple of (x, condition)")
        
        with tf.GradientTape() as tape:
            # Encoder
            z_mean, z_log_var, z = self.encode([x, condition])
            
            # Decoder
            reconstruction = self.decode([z, condition])
            
            # Tính toán loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mse(x, reconstruction), axis=1
                )
            )
            
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
                )
            )
            
            total_loss = reconstruction_loss + kl_loss
        
        # Cập nhật trọng số
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Cập nhật metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        if isinstance(data, tuple):
            x, condition = data
        else:
            raise ValueError("Expected a tuple of (x, condition)")
        
        # Encoder
        z_mean, z_log_var, z = self.encode([x, condition])
        
        # Decoder
        reconstruction = self.decode([z, condition])
        
        # Tính toán loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.mse(x, reconstruction), axis=1
            )
        )
        
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
            )
        )
        
        total_loss = reconstruction_loss + kl_loss
        
        # Cập nhật metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def generate(self, condition, z=None):
        if z is None:
            # Tạo vector ngẫu nhiên từ không gian tiềm ẩn
            batch_size = condition.shape[0]
            z = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # Tạo tín hiệu PPG từ vector z và điều kiện
        return self.decode([z, condition])

# Xây dựng mô hình
print("Đang xây dựng mô hình CVAE đơn giản hóa...")
cvae = SimplifiedCVAE(input_dim, condition_dim, latent_dim, hidden_units)

# Biên dịch mô hình
cvae.compile(optimizer=Adam(learning_rate=learning_rate))

# Tạo generator để cung cấp dữ liệu theo batch
def data_generator(x, condition, batch_size):
    num_samples = x.shape[0]
    while True:
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield x[batch_indices], condition[batch_indices]

# Huấn luyện mô hình
print("\nBắt đầu huấn luyện mô hình...")
start_time = time.time()

# Tạo generator
train_gen = data_generator(X_train, condition_train, batch_size)
val_gen = data_generator(X_test, condition_test, batch_size)

# Số lượng batch mỗi epoch
steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_test) // batch_size

# Huấn luyện mô hình
history = {"loss": [], "reconstruction_loss": [], "kl_loss": [], "val_loss": []}

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    # Training
    train_loss = []
    train_reconstruction_loss = []
    train_kl_loss = []
    
    for _ in range(steps_per_epoch):
        x_batch, condition_batch = next(train_gen)
        metrics = cvae.train_step((x_batch, condition_batch))
        train_loss.append(metrics["loss"].numpy())
        train_reconstruction_loss.append(metrics["reconstruction_loss"].numpy())
        train_kl_loss.append(metrics["kl_loss"].numpy())
    
    # Validation
    val_loss = []
    
    for _ in range(validation_steps):
        x_batch, condition_batch = next(val_gen)
        metrics = cvae.test_step((x_batch, condition_batch))
        val_loss.append(metrics["loss"].numpy())
    
    # Cập nhật history
    history["loss"].append(np.mean(train_loss))
    history["reconstruction_loss"].append(np.mean(train_reconstruction_loss))
    history["kl_loss"].append(np.mean(train_kl_loss))
    history["val_loss"].append(np.mean(val_loss))
    
    # In kết quả
    print(f"loss: {history['loss'][-1]:.4f}, reconstruction_loss: {history['reconstruction_loss'][-1]:.4f}, kl_loss: {history['kl_loss'][-1]:.4f}, val_loss: {history['val_loss'][-1]:.4f}")

training_time = time.time() - start_time
print(f"\nHuấn luyện hoàn tất trong {training_time:.2f} giây.")

# Lưu mô hình
model_save_path = os.path.join(model_path, 'cvae_model')
tf.saved_model.save(cvae, model_save_path)
print(f"Đã lưu mô hình tại: {model_save_path}")

# Vẽ biểu đồ quá trình huấn luyện
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history["loss"])
plt.plot(history["val_loss"])
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history["reconstruction_loss"])
plt.title('Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(history["kl_loss"])
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
    f.write(f"- Số epoch đã huấn luyện: {len(history['loss'])}\n")
    f.write(f"- Loss cuối cùng (train): {history['loss'][-1]:.4f}\n")
    f.write(f"- Loss cuối cùng (validation): {history['val_loss'][-1]:.4f}\n")
    f.write(f"- Reconstruction loss cuối cùng: {history['reconstruction_loss'][-1]:.4f}\n")
    f.write(f"- KL loss cuối cùng: {history['kl_loss'][-1]:.4f}\n")
    f.write(f"- Thời gian huấn luyện: {training_time:.2f} giây\n\n")
    
    f.write("Đường dẫn đến mô hình đã lưu:\n")
    f.write(f"- Mô hình: {model_save_path}\n")

print("\nQuá trình huấn luyện đã hoàn tất. Thông tin huấn luyện đã được lưu vào file training_info.txt.")
