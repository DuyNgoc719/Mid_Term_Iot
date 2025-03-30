import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from scipy.fft import fft, fftfreq
import pandas as pd

# Đường dẫn đến file dữ liệu
data_path = '/home/ubuntu/bidmc_project/data/bidmc_data.mat'
figures_path = '/home/ubuntu/bidmc_project/code/figures'
os.makedirs(figures_path, exist_ok=True)

# Tải dữ liệu
print("Đang tải dữ liệu từ file .mat...")
mat_data = sio.loadmat(data_path)
data = mat_data['data'][0]  # Lấy mảng chính chứa 53 bản ghi

# Khám phá cấu trúc dữ liệu
print(f"Số lượng bản ghi: {len(data)}")

# Khám phá cấu trúc chi tiết của bản ghi đầu tiên
first_record = data[0]
print("\nKhám phá cấu trúc chi tiết của bản ghi đầu tiên:")

# Kiểm tra cấu trúc của trường ppg
ppg_field = first_record['ppg'][0, 0]
print(f"Cấu trúc của trường ppg: {type(ppg_field)}")
if hasattr(ppg_field, 'dtype') and hasattr(ppg_field.dtype, 'names'):
    print(f"Các trường con của ppg: {ppg_field.dtype.names}")

# Kiểm tra cấu trúc của trường ref
ref_field = first_record['ref'][0, 0]
print(f"Cấu trúc của trường ref: {type(ref_field)}")
if hasattr(ref_field, 'dtype') and hasattr(ref_field.dtype, 'names'):
    print(f"Các trường con của ref: {ref_field.dtype.names}")
    
    # Kiểm tra cấu trúc của trường params trong ref
    if 'params' in ref_field.dtype.names:
        params_field = ref_field['params'][0, 0]
        print(f"Cấu trúc của trường params: {type(params_field)}")
        if hasattr(params_field, 'dtype') and hasattr(params_field.dtype, 'names'):
            print(f"Các trường con của params: {params_field.dtype.names}")

# Kiểm tra chi tiết hơn về cấu trúc dữ liệu
print("\nKiểm tra chi tiết hơn về cấu trúc dữ liệu:")
print(f"Kiểu dữ liệu của ppg.v: {type(first_record['ppg'][0, 0]['v'])}")
print(f"Kích thước của ppg.v: {first_record['ppg'][0, 0]['v'].shape}")

# Kiểm tra cấu trúc của HR và RR
print("\nKiểm tra cấu trúc của HR và RR:")
hr_field = first_record['ref'][0, 0]['params'][0, 0]['hr'][0]
print(f"Kiểu dữ liệu của hr: {type(hr_field)}")
print(f"Kích thước của hr: {hr_field.shape}")
print(f"Giá trị đầu tiên của hr: {hr_field[0] if hr_field.size > 0 else 'Không có dữ liệu'}")

rr_field = first_record['ref'][0, 0]['params'][0, 0]['rr'][0]
print(f"Kiểu dữ liệu của rr: {type(rr_field)}")
print(f"Kích thước của rr: {rr_field.shape}")
print(f"Giá trị đầu tiên của rr: {rr_field[0] if rr_field.size > 0 else 'Không có dữ liệu'}")

# Trích xuất và vẽ tín hiệu PPG từ bản ghi đầu tiên
try:
    # Truy cập trực tiếp vào dữ liệu PPG
    ppg_data = first_record['ppg'][0, 0]['v']
    if isinstance(ppg_data, np.ndarray):
        # Nếu là mảng numpy, lấy dữ liệu trực tiếp
        sample_ppg = ppg_data.flatten()
    else:
        # Nếu không phải mảng numpy, chuyển đổi thành mảng
        sample_ppg = np.array(ppg_data, dtype=float).flatten()
    
    # Lấy tần số lấy mẫu
    sample_fs_ppg = float(first_record['ppg'][0, 0]['fs'][0, 0])
    
    # Tương tự cho ECG và Resp
    ecg_data = first_record['ekg'][0, 0]['v']
    if isinstance(ecg_data, np.ndarray):
        sample_ecg = ecg_data.flatten()
    else:
        sample_ecg = np.array(ecg_data, dtype=float).flatten()
    
    sample_fs_ecg = float(first_record['ekg'][0, 0]['fs'][0, 0])
    
    resp_data = first_record['ref'][0, 0]['resp_sig'][0, 0]['imp'][0, 0]['v']
    if isinstance(resp_data, np.ndarray):
        sample_resp = resp_data.flatten()
    else:
        sample_resp = np.array(resp_data, dtype=float).flatten()
    
    sample_fs_resp = float(first_record['ref'][0, 0]['resp_sig'][0, 0]['imp'][0, 0]['fs'][0, 0])

    print(f"\nTần số lấy mẫu PPG: {sample_fs_ppg} Hz")
    print(f"Tần số lấy mẫu ECG: {sample_fs_ecg} Hz")
    print(f"Tần số lấy mẫu Resp: {sample_fs_resp} Hz")

    print(f"Độ dài tín hiệu PPG: {len(sample_ppg)} mẫu")
    print(f"Độ dài tín hiệu ECG: {len(sample_ecg)} mẫu")
    print(f"Độ dài tín hiệu Resp: {len(sample_resp)} mẫu")

    # Tính thời gian cho trục x
    time_ppg = np.arange(len(sample_ppg)) / sample_fs_ppg
    time_ecg = np.arange(len(sample_ecg)) / sample_fs_ecg
    time_resp = np.arange(len(sample_resp)) / sample_fs_resp

    # Vẽ tín hiệu mẫu
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(time_ecg[:1000], sample_ecg[:1000])
    plt.title('ECG Signal (First Record - First 1000 samples)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(time_ppg[:1000], sample_ppg[:1000])
    plt.title('PPG Signal (First Record - First 1000 samples)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.plot(time_resp[:1000], sample_resp[:1000])
    plt.title('Respiratory Signal (First Record - First 1000 samples)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'sample_signals.png'))
    plt.close()
    
    # Phân tích phổ tần số của tín hiệu PPG
    def plot_fft(signal, fs, title, filename):
        N = len(signal)
        T = 1.0 / fs
        yf = fft(signal)
        xf = fftfreq(N, T)[:N//2]
        
        plt.figure(figsize=(10, 6))
        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.grid(True, alpha=0.3)
        plt.title(f'FFT of {title}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, 5)  # Giới hạn tần số hiển thị đến 5Hz
        plt.savefig(os.path.join(figures_path, filename))
        plt.close()

    # Phân tích phổ tần số của tín hiệu PPG, ECG và Resp
    plot_fft(sample_ppg, sample_fs_ppg, 'PPG Signal', 'ppg_fft.png')
    plot_fft(sample_ecg, sample_fs_ecg, 'ECG Signal', 'ecg_fft.png')
    plot_fft(sample_resp, sample_fs_resp, 'Respiratory Signal', 'resp_fft.png')
    
    # Tạo báo cáo tóm tắt
    with open(os.path.join(figures_path, 'data_exploration_summary.txt'), 'w') as f:
        f.write("BÁO CÁO KHÁM PHÁ DỮ LIỆU BIDMC PPG AND RESPIRATION DATASET\n")
        f.write("==========================================================\n\n")
        
        f.write(f"Số lượng bản ghi: {len(data)}\n\n")
        
        f.write("Cấu trúc dữ liệu:\n")
        f.write("- Mỗi bản ghi chứa các trường: ppg, ekg, ref, fix\n")
        f.write("- Tín hiệu PPG và ECG được lưu trữ với giá trị (v) và tần số lấy mẫu (fs)\n")
        f.write("- Tín hiệu hô hấp được lưu trữ trong trường ref.resp_sig.imp\n")
        f.write("- Các thông số sinh lý (HR, RR, PR, SpO2) được lưu trữ trong trường ref.params\n\n")
        
        f.write(f"Tần số lấy mẫu PPG: {sample_fs_ppg} Hz\n")
        f.write(f"Tần số lấy mẫu ECG: {sample_fs_ecg} Hz\n")
        f.write(f"Tần số lấy mẫu Resp: {sample_fs_resp} Hz\n\n")
        
        f.write("Thách thức trong việc truy cập dữ liệu:\n")
        f.write("- Cấu trúc dữ liệu phức tạp với nhiều lớp lồng nhau\n")
        f.write("- Khó khăn trong việc chuyển đổi dữ liệu HR và RR sang định dạng float\n")
        f.write("- Cần phương pháp tiếp cận cẩn thận để trích xuất và xử lý dữ liệu\n\n")
        
        f.write("Các file đã tạo:\n")
        f.write("1. sample_signals.png - Biểu đồ mẫu của tín hiệu ECG, PPG và Respiratory\n")
        f.write("2. ppg_fft.png, ecg_fft.png, resp_fft.png - Phân tích phổ tần số của các tín hiệu\n")
        
    print("\nPhân tích dữ liệu hoàn tất. Các biểu đồ và báo cáo đã được lưu vào thư mục figures.")
    
except Exception as e:
    print(f"Lỗi khi vẽ tín hiệu mẫu: {e}")

# Kiểm tra trực tiếp một số bản ghi để tìm hiểu cấu trúc HR và RR
print("\nKiểm tra trực tiếp HR và RR từ một số bản ghi:")
for i in range(min(5, len(data))):
    try:
        record = data[i]
        params = record['ref'][0, 0]['params'][0, 0]
        
        hr_data = params['hr'][0]
        rr_data = params['rr'][0]
        
        print(f"\nBản ghi {i}:")
        print(f"Kiểu dữ liệu HR: {type(hr_data)}")
        if hasattr(hr_data, 'dtype'):
            print(f"Dtype của HR: {hr_data.dtype}")
        if hasattr(hr_data, 'shape'):
            print(f"Kích thước của HR: {hr_data.shape}")
        
        print(f"Kiểu dữ liệu RR: {type(rr_data)}")
        if hasattr(rr_data, 'dtype'):
            print(f"Dtype của RR: {rr_data.dtype}")
        if hasattr(rr_data, 'shape'):
            print(f"Kích thước của RR: {rr_data.shape}")
        
        # Thử in ra một số giá trị đầu tiên
        if hasattr(hr_data, 'size') and hr_data.size > 0:
            if hasattr(hr_data, 'dtype') and hr_data.dtype.names is not None and 'v' in hr_data.dtype.names:
                print(f"Giá trị HR đầu tiên (trường v): {hr_data['v'][0] if hr_data['v'].size > 0 else 'Không có dữ liệu'}")
            else:
                print(f"Giá trị HR đầu tiên: {hr_data[0]}")
        
        if hasattr(rr_data, 'size') and rr_data.size > 0:
            if hasattr(rr_data, 'dtype') and rr_data.dtype.names is not None and 'v' in rr_data.dtype.names:
                print(f"Giá trị RR đầu tiên (trường v): {rr_data['v'][0] if rr_data['v'].size > 0 else 'Không có dữ liệu'}")
            else:
                print(f"Giá trị RR đầu tiên: {rr_data[0]}")
    except Exception as e:
        print(f"Lỗi khi kiểm tra bản ghi {i}: {e}")
