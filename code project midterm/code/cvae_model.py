import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import datetime

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

# Định nghĩa lớp Sampling để lấy mẫu từ không gian tiềm ẩn
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Xây dựng Encoder
def build_encoder(input_dim, condition_dim, latent_dim, hidden_units):
    # Đầu vào tín hiệu PPG
    encoder_inputs = layers.Input(shape=(input_dim,), name='encoder_input')
    
    # Đầu vào điều kiện (HR và RR)
    condition_inputs = layers.Input(shape=(condition_dim,), name='condition_input')
    
    # Kết hợp đầu vào tín hiệu và điều kiện
    x = layers.Concatenate()([encoder_inputs, condition_inputs])
    
    # Các lớp ẩn
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
    
    # Lớp đầu ra
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    # Lấy mẫu từ không gian tiềm ẩn
    z = Sampling()([z_mean, z_log_var])
    
    # Định nghĩa mô hình
    encoder = Model([encoder_inputs, condition_inputs], [z_mean, z_log_var, z], name='encoder')
    
    return encoder

# Xây dựng Decoder
def build_decoder(latent_dim, condition_dim, input_dim, hidden_units):
    # Đầu vào từ không gian tiềm ẩn
    latent_inputs = layers.Input(shape=(latent_dim,), name='latent_input')
    
    # Đầu vào điều kiện (HR và RR)
    condition_inputs = layers.Input(shape=(condition_dim,), name='condition_input')
    
    # Kết hợp đầu vào từ không gian tiềm ẩn và điều kiện
    x = layers.Concatenate()([latent_inputs, condition_inputs])
    
    # Các lớp ẩn
    for units in reversed(hidden_units):
        x = layers.Dense(units, activation='relu')(x)
    
    # Lớp đầu ra
    decoder_outputs = layers.Dense(input_dim, activation='tanh')(x)
    
    # Định nghĩa mô hình
    decoder = Model([latent_inputs, condition_inputs], decoder_outputs, name='decoder')
    
    return decoder

# Xây dựng mô hình CVAE
class CVAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
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
    
    def train_step(self, data):
        if isinstance(data, tuple):
            x_condition = data[0]
            if isinstance(x_condition, list) and len(x_condition) == 2:
                x, condition = x_condition
            else:
                raise ValueError("Input data format is incorrect. Expected a list with [x, condition]")
        else:
            raise ValueError("Input data format is incorrect. Expected a tuple with ([x, condition], None)")
        
        with tf.GradientTape() as tape:
            # Encoder
            z_mean, z_log_var, z = self.encoder([x, condition])
            
            # Decoder
            reconstruction = self.decoder([z, condition])
            
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
            x_condition = data[0]
            if isinstance(x_condition, list) and len(x_condition) == 2:
                x, condition = x_condition
            else:
                raise ValueError("Input data format is incorrect. Expected a list with [x, condition]")
        else:
            raise ValueError("Input data format is incorrect. Expected a tuple with ([x, condition], None)")
        
        # Encoder
        z_mean, z_log_var, z = self.encoder([x, condition])
        
        # Decoder
        reconstruction = self.decoder([z, condition])
        
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
    
    def call(self, inputs):
        x, condition = inputs
        z_mean, z_log_var, z = self.encoder([x, condition])
        reconstruction = self.decoder([z, condition])
        return reconstruction
    
    def generate(self, condition, z=None):
        if z is None:
            # Tạo vector ngẫu nhiên từ không gian tiềm ẩn
            z = tf.random.normal(shape=(condition.shape[0], latent_dim))
        
        # Tạo tín hiệu PPG từ vector z và điều kiện
        return self.decoder([z, condition])

# Xây dựng mô hình
print("Đang xây dựng mô hình CVAE...")
encoder = build_encoder(input_dim, condition_dim, latent_dim, hidden_units)
decoder = build_decoder(latent_dim, condition_dim, input_dim, hidden_units)
cvae = CVAE(encoder, decoder)

# Biên dịch mô hình
cvae.compile(optimizer=Adam(learning_rate=learning_rate))

# Tóm tắt mô hình
print("Tóm tắt mô hình Encoder:")
encoder.summary()
print("\nTóm tắt mô hình Decoder:")
decoder.summary()

# Chuẩn bị dữ liệu điều kiện
condition_train = np.column_stack((hr_train, rr_train))
condition_test = np.column_stack((hr_test, rr_test))

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

# Lưu thông tin mô hình
with open(os.path.join(model_path, 'model_info.txt'), 'w') as f:
    f.write("THÔNG TIN MÔ HÌNH CVAE\n")
    f.write("=====================\n\n")
    
    f.write("Tham số mô hình:\n")
    f.write(f"- Kích thước đầu vào: {input_dim}\n")
    f.write(f"- Kích thước điều kiện: {condition_dim}\n")
    f.write(f"- Kích thước không gian tiềm ẩn: {latent_dim}\n")
    f.write(f"- Số đơn vị ẩn trong các lớp: {hidden_units}\n")
    f.write(f"- Kích thước batch: {batch_size}\n")
    f.write(f"- Số epoch: {epochs}\n")
    f.write(f"- Tốc độ học: {learning_rate}\n\n")
    
    f.write("Kích thước dữ liệu:\n")
    f.write(f"- Tập huấn luyện: {X_train.shape[0]} mẫu\n")
    f.write(f"- Tập kiểm thử: {X_test.shape[0]} mẫu\n")

print("\nMô hình CVAE đã được xây dựng thành công.")
print("Thông tin mô hình đã được lưu vào file model_info.txt.")
print("Sẵn sàng để huấn luyện mô hình.")
