import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import time

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
epochs = 50
learning_rate = 0.001

# Chuẩn bị dữ liệu điều kiện
condition_train = np.column_stack((hr_train, rr_train))
condition_test = np.column_stack((hr_test, rr_test))

# Định nghĩa lớp Sampling để lấy mẫu từ không gian tiềm ẩn
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Xây dựng mô hình CVAE
class CVAE(tf.keras.Model):
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_units):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        
        # Encoder layers
        self.encoder_inputs = layers.InputLayer(input_shape=(input_dim,), name='encoder_input')
        self.condition_inputs = layers.InputLayer(input_shape=(condition_dim,), name='condition_input')
        self.concat_layer = layers.Concatenate()
        
        self.encoder_layers = []
        for units in hidden_units:
            self.encoder_layers.append(layers.Dense(units, activation='relu'))
        
        self.z_mean_layer = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var_layer = layers.Dense(latent_dim, name='z_log_var')
        self.sampling = Sampling()
        
        # Decoder layers
        self.latent_inputs = layers.InputLayer(input_shape=(latent_dim,), name='latent_input')
        self.decoder_concat_layer = layers.Concatenate()
        
        self.decoder_layers = []
        for units in reversed(hidden_units):
            self.decoder_layers.append(layers.Dense(units, activation='relu'))
        
        self.decoder_outputs = layers.Dense(input_dim, activation='tanh')
        
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
        x, condition = inputs
        x = self.encoder_inputs(x)
        condition = self.condition_inputs(condition)
        concat = self.concat_layer([x, condition])
        
        for layer in self.encoder_layers:
            concat = layer(concat)
        
        z_mean = self.z_mean_layer(concat)
        z_log_var = self.z_log_var_layer(concat)
        z = self.sampling([z_mean, z_log_var])
        
        return z_mean, z_log_var, z
    
    def decode(self, inputs):
        z, condition = inputs
        z = self.latent_inputs(z)
        condition = self.condition_inputs(condition)
        concat = self.decoder_concat_layer([z, condition])
        
        for layer in self.decoder_layers:
            concat = layer(concat)
        
        reconstruction = self.decoder_outputs(concat)
        return reconstruction
    
    def call(self, inputs):
        x, condition = inputs
        z_mean, z_log_var, z = self.encode([x, condition])
        reconstruction = self.decode([z, condition])
        return reconstruction
    
    def train_step(self, data):
        x, condition = data
        
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
        x, condition = data
        
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
print("Đang xây dựng mô hình CVAE...")
cvae = CVAE(input_dim, condition_dim, latent_dim, hidden_units)

# Biên dịch mô hình
cvae.compile(optimizer=Adam(learning_rate=learning_rate))

# Tạo TensorBoard callback
log_dir = os.path.join(model_path, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Tạo ModelCheckpoint callback
checkpoint_path = os.path.join(model_path, "cvae_checkpoint.weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True
)

# Tạo EarlyStopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=True
)

# Huấn luyện mô hình
print("\nBắt đầu huấn luyện mô hình...")
start_time = time.time()

# Tạo dataset từ dữ liệu
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, condition_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, condition_test)).batch(batch_size)

history = cvae.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset,
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
