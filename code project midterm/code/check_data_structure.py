import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from scipy.fft import fft, fftfreq
import pandas as pd

# Đường dẫn đến file dữ liệu
data_path = '/home/ubuntu/bidmc_project/data/bidmc_data.mat'
figures_path = '/home/ubuntu/bidmc_project/code/figures'

# Tải dữ liệu
print("Đang tải dữ liệu từ file .mat...")
mat_data = sio.loadmat(data_path)

# Kiểm tra cấu trúc dữ liệu
print("Các khóa trong file .mat:")
for key in mat_data.keys():
    print(f"- {key}")

# Kiểm tra cấu trúc chi tiết
if 'data' in mat_data:
    data = mat_data['data']
    print(f"\nKiểu dữ liệu của 'data': {type(data)}")
    print(f"Kích thước của 'data': {data.shape}")
    
    # Kiểm tra cấu trúc của phần tử đầu tiên nếu data là mảng
    if isinstance(data, np.ndarray) and data.size > 0:
        first_record = data[0]
        if hasattr(first_record, 'dtype') and hasattr(first_record.dtype, 'names'):
            print("\nCác trường trong bản ghi đầu tiên:")
            for field in first_record.dtype.names:
                print(f"- {field}")
                
                # Kiểm tra cấu trúc chi tiết của từng trường
                field_data = first_record[field]
                print(f"  Kiểu: {type(field_data)}, Kích thước: {field_data.shape}")
                
                # Nếu là mảng có cấu trúc, hiển thị thêm thông tin
                if hasattr(field_data, 'dtype') and hasattr(field_data.dtype, 'names'):
                    print(f"  Các trường con: {field_data.dtype.names}")
        else:
            print(f"\nPhần tử đầu tiên không có cấu trúc trường, kiểu: {type(first_record)}")
    else:
        print("\nKhông thể truy cập phần tử đầu tiên của 'data'")
else:
    print("\nKhông tìm thấy khóa 'data' trong file .mat")
    print("Đang kiểm tra cấu trúc dữ liệu thay thế...")
    
    # Tìm khóa có thể chứa dữ liệu chính
    potential_data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    for key in potential_data_keys:
        print(f"\nKiểm tra khóa: {key}")
        data_item = mat_data[key]
        print(f"Kiểu: {type(data_item)}, ", end="")
        
        if isinstance(data_item, np.ndarray):
            print(f"Kích thước: {data_item.shape}")
            
            # Kiểm tra nếu là mảng có cấu trúc
            if data_item.dtype.names is not None:
                print(f"Các trường: {data_item.dtype.names}")
            
            # Kiểm tra phần tử đầu tiên nếu là mảng nhiều chiều
            if data_item.size > 0:
                first_item = data_item[0]
                print(f"Kiểu phần tử đầu tiên: {type(first_item)}")
                
                # Nếu phần tử đầu tiên là mảng có cấu trúc
                if hasattr(first_item, 'dtype') and hasattr(first_item.dtype, 'names'):
                    print(f"Các trường trong phần tử đầu tiên: {first_item.dtype.names}")
        else:
            print(f"Không phải mảng numpy")

print("\nĐang lưu thông tin cấu trúc dữ liệu vào file...")
with open(os.path.join(figures_path, 'data_structure.txt'), 'w') as f:
    f.write("CẤU TRÚC DỮ LIỆU BIDMC PPG AND RESPIRATION DATASET\n")
    f.write("=================================================\n\n")
    
    f.write("Các khóa trong file .mat:\n")
    for key in mat_data.keys():
        f.write(f"- {key}\n")
    
    f.write("\nThông tin chi tiết về cấu trúc dữ liệu được lưu trong file này.")

print("\nĐã lưu thông tin cấu trúc dữ liệu. Vui lòng kiểm tra file data_structure.txt")
