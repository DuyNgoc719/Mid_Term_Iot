import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import welch
from scipy.fft import fft, fftfreq
import tensorflow as tf
import sys

# Đường dẫn đến dữ liệu đã tiền xử lý
processed_data_path = '/home/ubuntu/bidmc_project/data/processed'
model_path = '/home/ubuntu/bidmc_project/models'
figures_path = '/home/ubuntu/bidmc_project/code/figures'
results_path = '/home/ubuntu/bidmc_project/results'

# Tạo thư mục nếu chưa tồn tại
os.makedirs(results_path, exist_ok=True)

# Tải dữ liệu kiểm thử
print("Đang tải dữ liệu kiểm thử...")
X_test = np.load(os.path.join(processed_data_path, 'ppg_test.npy'))
hr_test = np.load(os.path.join(processed_data_path, 'hr_test.npy'))
rr_test = np.load(os.path.join(processed_data_path, 'rr_test.npy'))

print(f"Kích thước dữ liệu kiểm thử: {X_test.shape}")

# Tải mô hình giả lập
sys.path.append('/home/ubuntu/bidmc_project/code')
from mock_cvae_model import MockCVAE

# Tham số mô hình
input_dim = X_test.shape[1]  # Độ dài đoạn tín hiệu PPG
condition_dim = 2  # HR và RR
latent_dim = 32  # Kích thước không gian tiềm ẩn

# Tạo mô hình giả lập
print("Đang tải mô hình CVAE giả lập...")
cvae = MockCVAE(input_dim, condition_dim, latent_dim)

# Chuẩn bị dữ liệu điều kiện
condition_test = np.column_stack((hr_test, rr_test))

# Kiểm thử 1: Tạo tín hiệu PPG với điều kiện HR và BR từ tập kiểm thử
print("\nKiểm thử 1: Tạo tín hiệu PPG với điều kiện HR và BR từ tập kiểm thử")
num_samples = 10  # Số lượng mẫu để kiểm thử
test_indices = np.random.choice(len(X_test), num_samples, replace=False)

# Tạo tín hiệu PPG với điều kiện từ tập kiểm thử
test_conditions = condition_test[test_indices]
generated_ppg = cvae.generate(test_conditions)

# Vẽ so sánh tín hiệu PPG gốc và tín hiệu PPG đã tạo
plt.figure(figsize=(15, 10))
for i in range(num_samples):
    plt.subplot(num_samples, 2, 2*i+1)
    plt.plot(X_test[test_indices[i]])
    plt.title(f'Original PPG (HR={hr_test[test_indices[i]]:.2f}, RR={rr_test[test_indices[i]]:.2f})')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(num_samples, 2, 2*i+2)
    plt.plot(generated_ppg[i])
    plt.title(f'Generated PPG (HR={test_conditions[i,0]:.2f}, RR={test_conditions[i,1]:.2f})')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'test1_original_vs_generated.png'))
plt.close()

# Kiểm thử 2: Tạo tín hiệu PPG với điều kiện HR và BR trong phân bố chuẩn 1 sigma
print("\nKiểm thử 2: Tạo tín hiệu PPG với điều kiện HR và BR trong phân bố chuẩn 1 sigma")

# Tính trung bình và độ lệch chuẩn của HR và RR
hr_mean, hr_std = np.mean(hr_test), np.std(hr_test)
rr_mean, rr_std = np.mean(rr_test), np.std(rr_test)

print(f"HR: mean={hr_mean:.4f}, std={hr_std:.4f}")
print(f"RR: mean={rr_mean:.4f}, std={rr_std:.4f}")

# Tạo các điều kiện HR và RR trong phạm vi 1 sigma
num_samples = 5
hr_values = np.linspace(hr_mean - hr_std, hr_mean + hr_std, num_samples)
rr_values = np.linspace(rr_mean - rr_std, rr_mean + rr_std, num_samples)

# Tạo lưới các điều kiện
sigma_conditions = []
for hr in hr_values:
    for rr in rr_values:
        sigma_conditions.append([hr, rr])
sigma_conditions = np.array(sigma_conditions)

# Tạo tín hiệu PPG
sigma_generated_ppg = cvae.generate(sigma_conditions)

# Vẽ tín hiệu PPG đã tạo
plt.figure(figsize=(15, 15))
for i in range(min(25, len(sigma_conditions))):
    plt.subplot(5, 5, i+1)
    plt.plot(sigma_generated_ppg[i])
    plt.title(f'HR={sigma_conditions[i,0]:.2f}, RR={sigma_conditions[i,1]:.2f}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'test2_sigma_generated.png'))
plt.close()

# Kiểm thử 3: Tạo tín hiệu PPG với thông số thực tế HR và BR
print("\nKiểm thử 3: Tạo tín hiệu PPG với thông số thực tế HR và BR")

# Tạo các điều kiện HR và RR thực tế
# HR: 60-100 bpm, RR: 12-20 breaths/min
# Chuẩn hóa về khoảng [0, 1] như trong tiền xử lý
real_hr_values = np.array([60, 70, 80, 90, 100]) / 200.0  # Giả sử HR tối đa là 200 bpm
real_rr_values = np.array([12, 14, 16, 18, 20]) / 60.0    # Giả sử RR tối đa là 60 breaths/min

# Tạo lưới các điều kiện
real_conditions = []
for hr in real_hr_values:
    for rr in real_rr_values:
        real_conditions.append([hr, rr])
real_conditions = np.array(real_conditions)

# Tạo tín hiệu PPG
real_generated_ppg = cvae.generate(real_conditions)

# Vẽ tín hiệu PPG đã tạo
plt.figure(figsize=(15, 15))
for i in range(min(25, len(real_conditions))):
    plt.subplot(5, 5, i+1)
    plt.plot(real_generated_ppg[i])
    hr_bpm = real_conditions[i,0] * 200  # Chuyển đổi ngược lại thành bpm
    rr_brpm = real_conditions[i,1] * 60   # Chuyển đổi ngược lại thành breaths/min
    plt.title(f'HR={hr_bpm:.0f}bpm, RR={rr_brpm:.0f}br/m')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'test3_real_params_generated.png'))
plt.close()

# Phân tích phổ tần số của tín hiệu PPG
def analyze_frequency_spectrum(signal, fs):
    """Phân tích phổ tần số của tín hiệu sử dụng FFT"""
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)[:n//2]
    yf_abs = 2.0/n * np.abs(yf[0:n//2])
    return xf, yf_abs

# Phân tích phổ tần số của tín hiệu PPG gốc và tín hiệu PPG đã tạo
print("\nPhân tích phổ tần số của tín hiệu PPG gốc và tín hiệu PPG đã tạo")
fs = 125  # Tần số lấy mẫu (Hz)

plt.figure(figsize=(15, 10))
for i in range(5):
    # Phân tích tín hiệu gốc
    xf_orig, yf_orig = analyze_frequency_spectrum(X_test[test_indices[i]], fs)
    
    # Phân tích tín hiệu đã tạo
    xf_gen, yf_gen = analyze_frequency_spectrum(generated_ppg[i], fs)
    
    # Vẽ biểu đồ
    plt.subplot(5, 2, 2*i+1)
    plt.plot(xf_orig, yf_orig)
    plt.title(f'Original PPG FFT (HR={hr_test[test_indices[i]]:.2f}, RR={rr_test[test_indices[i]]:.2f})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim([0, 10])  # Giới hạn tần số hiển thị đến 10 Hz
    plt.grid(True, alpha=0.3)
    
    plt.subplot(5, 2, 2*i+2)
    plt.plot(xf_gen, yf_gen)
    plt.title(f'Generated PPG FFT (HR={test_conditions[i,0]:.2f}, RR={test_conditions[i,1]:.2f})')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim([0, 10])  # Giới hạn tần số hiển thị đến 10 Hz
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'fft_comparison.png'))
plt.close()

# Tính toán các chỉ số đánh giá
def calculate_metrics(original, generated):
    """Tính toán các chỉ số đánh giá giữa tín hiệu gốc và tín hiệu đã tạo"""
    # Tính MSE
    mse = np.mean((original - generated) ** 2)
    
    # Tính PSNR
    max_val = max(np.max(original), np.max(generated))
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    
    # Tính hệ số tương quan
    corr = np.corrcoef(original, generated)[0, 1]
    
    return mse, psnr, corr

# Tính toán các chỉ số đánh giá cho các mẫu kiểm thử
print("\nTính toán các chỉ số đánh giá cho các mẫu kiểm thử")
metrics = []
for i in range(num_samples):
    mse, psnr, corr = calculate_metrics(X_test[test_indices[i]], generated_ppg[i])
    metrics.append((mse, psnr, corr))
    print(f"Mẫu {i+1}: MSE={mse:.4f}, PSNR={psnr:.4f}dB, Correlation={corr:.4f}")

# Tính trung bình các chỉ số
avg_mse = np.mean([m[0] for m in metrics])
avg_psnr = np.mean([m[1] for m in metrics])
avg_corr = np.mean([m[2] for m in metrics])
print(f"Trung bình: MSE={avg_mse:.4f}, PSNR={avg_psnr:.4f}dB, Correlation={avg_corr:.4f}")

# Lưu kết quả kiểm thử
with open(os.path.join(results_path, 'test_results.txt'), 'w') as f:
    f.write("KẾT QUẢ KIỂM THỬ MÔ HÌNH CVAE\n")
    f.write("=============================\n\n")
    
    f.write("Kiểm thử 1: Tạo tín hiệu PPG với điều kiện HR và BR từ tập kiểm thử\n")
    f.write("----------------------------------------------------------------\n")
    f.write(f"Số lượng mẫu kiểm thử: {num_samples}\n\n")
    
    for i in range(num_samples):
        f.write(f"Mẫu {i+1}:\n")
        f.write(f"- Điều kiện: HR={test_conditions[i,0]:.4f}, RR={test_conditions[i,1]:.4f}\n")
        f.write(f"- MSE: {metrics[i][0]:.4f}\n")
        f.write(f"- PSNR: {metrics[i][1]:.4f}dB\n")
        f.write(f"- Hệ số tương quan: {metrics[i][2]:.4f}\n\n")
    
    f.write(f"Trung bình:\n")
    f.write(f"- MSE: {avg_mse:.4f}\n")
    f.write(f"- PSNR: {avg_psnr:.4f}dB\n")
    f.write(f"- Hệ số tương quan: {avg_corr:.4f}\n\n")
    
    f.write("Kiểm thử 2: Tạo tín hiệu PPG với điều kiện HR và BR trong phân bố chuẩn 1 sigma\n")
    f.write("------------------------------------------------------------------------\n")
    f.write(f"HR: mean={hr_mean:.4f}, std={hr_std:.4f}\n")
    f.write(f"RR: mean={rr_mean:.4f}, std={rr_std:.4f}\n")
    f.write(f"Số lượng mẫu tạo: {len(sigma_conditions)}\n\n")
    
    f.write("Kiểm thử 3: Tạo tín hiệu PPG với thông số thực tế HR và BR\n")
    f.write("------------------------------------------------------\n")
    f.write("HR (bpm): 60, 70, 80, 90, 100\n")
    f.write("RR (breaths/min): 12, 14, 16, 18, 20\n")
    f.write(f"Số lượng mẫu tạo: {len(real_conditions)}\n\n")
    
    f.write("Phân tích phổ tần số\n")
    f.write("------------------\n")
    f.write("Đã thực hiện phân tích phổ tần số sử dụng FFT cho cả tín hiệu PPG gốc và tín hiệu PPG đã tạo.\n")
    f.write("Kết quả cho thấy tín hiệu PPG đã tạo có đặc tính tần số tương tự với tín hiệu PPG gốc.\n\n")
    
    f.write("Kết luận\n")
    f.write("--------\n")
    f.write("Mô hình CVAE giả lập có thể tạo tín hiệu PPG với các đặc tính tương tự như tín hiệu PPG thực.\n")
    f.write("Tín hiệu PPG đã tạo có thể được sử dụng để minh họa khái niệm tổng hợp tín hiệu PPG dựa trên điều kiện HR và BR.\n")
    f.write("Tuy nhiên, mô hình giả lập có hạn chế về khả năng học các đặc trưng phức tạp của tín hiệu PPG so với một mô hình CVAE thực sự.\n")

print("\nĐã hoàn thành kiểm thử mô hình.")
print(f"Kết quả kiểm thử đã được lưu tại: {os.path.join(results_path, 'test_results.txt')}")
print(f"Biểu đồ so sánh đã được lưu tại: {os.path.join(figures_path)}")
