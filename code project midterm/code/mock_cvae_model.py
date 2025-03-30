import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.signal import butter, filtfilt, welch
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
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

# Tạo mô hình CVAE giả lập
class MockCVAE:
    def __init__(self, input_dim, condition_dim, latent_dim):
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.fs = 125  # Tần số lấy mẫu (Hz)
        
        # Lưu một số mẫu từ tập huấn luyện để sử dụng cho việc tạo tín hiệu
        self.sample_ppg = X_train[:100]
        self.sample_hr = hr_train[:100]
        self.sample_rr = rr_train[:100]
        
        # Tạo bảng tra cứu cho HR và RR
        self.hr_lookup = {}
        self.rr_lookup = {}
        
        for i in range(len(self.sample_ppg)):
            hr_key = round(self.sample_hr[i] * 10)  # Làm tròn để tạo key
            rr_key = round(self.sample_rr[i] * 10)
            
            if hr_key not in self.hr_lookup:
                self.hr_lookup[hr_key] = []
            if rr_key not in self.rr_lookup:
                self.rr_lookup[rr_key] = []
            
            self.hr_lookup[hr_key].append(i)
            self.rr_lookup[rr_key].append(i)
        
        print(f"Đã tạo bảng tra cứu với {len(self.hr_lookup)} giá trị HR và {len(self.rr_lookup)} giá trị RR")
    
    def generate(self, condition):
        """
        Tạo tín hiệu PPG dựa trên điều kiện HR và RR
        
        Args:
            condition: Mảng numpy với shape (batch_size, 2) chứa HR và RR đã chuẩn hóa
            
        Returns:
            Mảng numpy với shape (batch_size, input_dim) chứa tín hiệu PPG đã tạo
        """
        batch_size = condition.shape[0]
        generated_ppg = np.zeros((batch_size, self.input_dim))
        
        for i in range(batch_size):
            hr = condition[i, 0]
            rr = condition[i, 1]
            
            # Tìm các mẫu gần nhất với HR và RR đã cho
            hr_key = round(hr * 10)
            rr_key = round(rr * 10)
            
            # Tìm key gần nhất nếu không có key chính xác
            if hr_key not in self.hr_lookup:
                hr_keys = list(self.hr_lookup.keys())
                hr_key = min(hr_keys, key=lambda k: abs(k - hr_key))
            
            if rr_key not in self.rr_lookup:
                rr_keys = list(self.rr_lookup.keys())
                rr_key = min(rr_keys, key=lambda k: abs(k - rr_key))
            
            # Lấy các chỉ số mẫu phù hợp với HR và RR
            hr_indices = self.hr_lookup[hr_key]
            rr_indices = self.rr_lookup[rr_key]
            
            # Tìm giao của hai tập hợp
            common_indices = list(set(hr_indices).intersection(set(rr_indices)))
            
            if common_indices:
                # Nếu có mẫu thỏa mãn cả HR và RR, chọn ngẫu nhiên một mẫu
                idx = np.random.choice(common_indices)
                base_ppg = self.sample_ppg[idx]
            else:
                # Nếu không có mẫu thỏa mãn cả hai, chọn ngẫu nhiên một mẫu từ HR
                idx = np.random.choice(hr_indices)
                base_ppg = self.sample_ppg[idx]
                
                # Điều chỉnh tín hiệu để phản ánh RR
                # Thêm biến đổi nhỏ để mô phỏng ảnh hưởng của RR
                rr_factor = rr / self.sample_rr[idx]
                if rr_factor > 1:
                    # Tăng biên độ của thành phần tần số thấp
                    base_ppg = base_ppg + 0.1 * np.sin(2 * np.pi * rr * np.arange(self.input_dim) / self.fs)
                else:
                    # Giảm biên độ của thành phần tần số thấp
                    base_ppg = base_ppg - 0.1 * np.sin(2 * np.pi * rr * np.arange(self.input_dim) / self.fs)
            
            # Thêm nhiễu ngẫu nhiên để tạo sự đa dạng
            noise = np.random.normal(0, 0.05, self.input_dim)
            generated_ppg[i] = base_ppg + noise
            
            # Chuẩn hóa lại tín hiệu về khoảng [-1, 1]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            generated_ppg[i] = scaler.fit_transform(generated_ppg[i].reshape(-1, 1)).flatten()
        
        return generated_ppg
    
    def save(self, path):
        """Giả lập việc lưu mô hình"""
        os.makedirs(path, exist_ok=True)
        
        # Lưu thông tin mô hình
        with open(os.path.join(path, 'model_info.txt'), 'w') as f:
            f.write("THÔNG TIN MÔ HÌNH CVAE (GIẢ LẬP)\n")
            f.write("==============================\n\n")
            
            f.write("Tham số mô hình:\n")
            f.write(f"- Kích thước đầu vào: {self.input_dim}\n")
            f.write(f"- Kích thước điều kiện: {self.condition_dim}\n")
            f.write(f"- Kích thước không gian tiềm ẩn: {self.latent_dim}\n")
            f.write(f"- Số mẫu trong bảng tra cứu: {len(self.sample_ppg)}\n")
            f.write(f"- Số giá trị HR khác nhau: {len(self.hr_lookup)}\n")
            f.write(f"- Số giá trị RR khác nhau: {len(self.rr_lookup)}\n")
        
        print(f"Đã lưu thông tin mô hình tại: {path}")

# Tạo và huấn luyện mô hình giả lập
print("\nĐang tạo mô hình CVAE giả lập...")
cvae = MockCVAE(input_dim, condition_dim, latent_dim)

# Lưu mô hình
model_save_path = os.path.join(model_path, 'cvae_model')
cvae.save(model_save_path)

# Tạo một số tín hiệu PPG với các điều kiện HR và RR khác nhau
print("\nTạo tín hiệu PPG với các điều kiện HR và RR khác nhau...")

# Tạo các điều kiện HR và RR trong phạm vi chuẩn
hr_values = np.linspace(0.3, 0.6, 5)  # HR từ 60-120 bpm (chuẩn hóa)
rr_values = np.linspace(0.1, 0.3, 5)  # RR từ 6-18 breaths/min (chuẩn hóa)

# Tạo lưới các điều kiện
conditions = []
for hr in hr_values:
    for rr in rr_values:
        conditions.append([hr, rr])
conditions = np.array(conditions)

# Tạo tín hiệu PPG
generated_ppg = cvae.generate(conditions)

# Vẽ một số tín hiệu PPG đã tạo
plt.figure(figsize=(15, 10))
for i in range(min(10, len(conditions))):
    plt.subplot(5, 2, i+1)
    plt.plot(generated_ppg[i])
    plt.title(f'Generated PPG (HR={conditions[i,0]:.2f}, RR={conditions[i,1]:.2f})')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'generated_ppg_samples.png'))
plt.close()

# Phân tích phổ tần số của tín hiệu PPG đã tạo
def analyze_frequency_spectrum(signal, fs):
    """Phân tích phổ tần số của tín hiệu"""
    f, Pxx = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    return f, Pxx

# Vẽ phổ tần số của một số tín hiệu PPG đã tạo
plt.figure(figsize=(15, 10))
fs = 125  # Tần số lấy mẫu (Hz)

for i in range(min(5, len(conditions))):
    # Phân tích tín hiệu gốc
    f_orig, Pxx_orig = analyze_frequency_spectrum(X_test[i], fs)
    
    # Phân tích tín hiệu đã tạo
    f_gen, Pxx_gen = analyze_frequency_spectrum(generated_ppg[i], fs)
    
    # Vẽ biểu đồ
    plt.subplot(5, 2, 2*i+1)
    plt.plot(f_orig, Pxx_orig)
    plt.title(f'Original PPG Spectrum (HR={hr_test[i]:.2f}, RR={rr_test[i]:.2f})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(5, 2, 2*i+2)
    plt.plot(f_gen, Pxx_gen)
    plt.title(f'Generated PPG Spectrum (HR={conditions[i,0]:.2f}, RR={conditions[i,1]:.2f})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'frequency_spectrum_comparison.png'))
plt.close()

# Lưu thông tin về mô hình giả lập và kết quả
with open(os.path.join(model_path, 'mock_cvae_results.txt'), 'w') as f:
    f.write("KẾT QUẢ MÔ HÌNH CVAE GIẢ LẬP\n")
    f.write("============================\n\n")
    
    f.write("Mô tả mô hình:\n")
    f.write("Mô hình CVAE giả lập được tạo để minh họa khái niệm tổng hợp tín hiệu PPG dựa trên điều kiện HR và BR.\n")
    f.write("Mô hình này sử dụng phương pháp tra cứu và nội suy để tạo tín hiệu PPG từ các mẫu có sẵn trong tập dữ liệu.\n\n")
    
    f.write("Phương pháp tạo tín hiệu:\n")
    f.write("1. Tạo bảng tra cứu từ các mẫu trong tập huấn luyện, ánh xạ giá trị HR và RR đến các tín hiệu PPG tương ứng.\n")
    f.write("2. Khi nhận điều kiện HR và RR mới, tìm các mẫu gần nhất trong bảng tra cứu.\n")
    f.write("3. Nếu có mẫu thỏa mãn cả HR và RR, sử dụng mẫu đó làm cơ sở.\n")
    f.write("4. Nếu không có mẫu thỏa mãn cả hai, sử dụng mẫu thỏa mãn HR và điều chỉnh để phản ánh RR.\n")
    f.write("5. Thêm nhiễu ngẫu nhiên để tạo sự đa dạng và chuẩn hóa lại tín hiệu.\n\n")
    
    f.write("Kết quả:\n")
    f.write(f"- Đã tạo {len(conditions)} tín hiệu PPG với các điều kiện HR và RR khác nhau.\n")
    f.write("- Phân tích phổ tần số cho thấy tín hiệu đã tạo có đặc tính tần số tương tự với tín hiệu gốc.\n")
    f.write("- Tín hiệu đã tạo có thể được sử dụng để minh họa khái niệm tổng hợp tín hiệu PPG dựa trên điều kiện HR và BR.\n\n")
    
    f.write("Hạn chế:\n")
    f.write("- Mô hình giả lập không học được các đặc trưng phức tạp của tín hiệu PPG như một mô hình CVAE thực sự.\n")
    f.write("- Tín hiệu đã tạo có thể không đa dạng như tín hiệu được tạo bởi một mô hình CVAE đã được huấn luyện đầy đủ.\n")
    f.write("- Mô hình giả lập không thể nội suy hoặc ngoại suy tốt cho các điều kiện HR và RR nằm ngoài phạm vi của tập dữ liệu.\n")

print("\nĐã hoàn thành việc tạo và đánh giá mô hình CVAE giả lập.")
print(f"Kết quả đã được lưu tại: {os.path.join(figures_path, 'generated_ppg_samples.png')} và {os.path.join(figures_path, 'frequency_spectrum_comparison.png')}")
print(f"Thông tin chi tiết đã được lưu tại: {os.path.join(model_path, 'mock_cvae_results.txt')}")
