import pandas as pd
import altair as alt
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.api import VAR
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import itertools
import time
import random
from sklearn.impute import KNNImputer
from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
import os
import pickle
from datetime import datetime
import matplotlib.dates as mdates
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

SEED = 12 

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)

MODEL_BASE_DIR = "saved_models"
os.makedirs(MODEL_BASE_DIR, exist_ok=True)

# Nút "Add File" trong sidebar
uploaded_file = st.sidebar.file_uploader(
    "Tải lên tệp của bạn:",
    type=["csv", "xlsx"]  # Các loại tệp được phép
)

# Thêm selectbox trong sidebar để chọn cách hiển thị dữ liệu
display_option = st.sidebar.selectbox(
    "Chọn cách hiển thị dữ liệu:",
    ["Hiển thị bảng số liệu", "Hiển thị biểu đồ", "Hiển thị cả bảng số liệu và biểu đồ"]
)

# Nút tiền xử lý dữ liệu
process_data = st.sidebar.button("Tiền xử lý dữ liệu")

# Kiểm tra nếu dữ liệu đã được tải lên
if 'df' not in st.session_state:
    st.session_state["df"] = None  # Dữ liệu chưa được xử lý
    st.session_state["df_cleaned"] = None  # Dữ liệu đã làm sạch
    st.session_state["df_stationary"] = None  # Dữ liệu đã dừng hóa

# Xử lý khi tệp được tải lên
if uploaded_file is not None:
    try:
        doc_dl_start_time = time.time()
        # Đọc tệp CSV
        df = pd.read_csv(uploaded_file)

        # Kiểm tra nếu DataFrame không rỗng
        if df.empty:
            st.error("Tệp tải lên không có dữ liệu hợp lệ.")
        else:
            # Hiển thị dữ liệu mẫu trước khi tiền xử lý
            st.write("### Dữ liệu mẫu trước khi tiền xử lý:")
            st.dataframe(df)  # Hiển thị 5 dòng dữ liệu đầu tiên

            # Lưu dữ liệu gốc vào session_state để sử dụng trong các bước sau
            st.session_state["df"] = df

            doc_dl_end_time = time.time()
            # Tính thời gian thực thi
            doc_dl_execution_time = doc_dl_end_time - doc_dl_start_time
            st.write(f"Thời gian thực thi đoạn mã: {doc_dl_execution_time} giây")

    except Exception as e:
        # Hiển thị lỗi nếu không đọc được tệp
        st.error(f"Không thể đọc tệp CSV. Lỗi: {e}")
else:
    st.info("Vui lòng tải lên một tệp CSV để hiển thị dữ liệu.")

# Bước tiền xử lý dữ liệu khi nút được nhấn
if process_data and uploaded_file is not None:
    if "df" in st.session_state:
        # Tiến hành tiền xử lý
        df = st.session_state["df"]

        # Lấy danh sách các cột số
        numeric_columns = df.columns.difference(['date'])  # Lấy danh sách các cột số trừ 'date'
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Thay thế giá trị thiếu bằng trung vị cho các cột số
        st.write("### Bước 1: Thay thế giá trị thiếu bằng trung vị")
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())  # Thay thế NaN bằng trung vị của cột
        st.write("Đã thay thế giá trị thiếu bằng trung vị.")

        # Tính Q1, Q3 và IQR cho mỗi cột số (không loại bỏ dòng mà thay thế ngoại lai)
        Q1 = df[numeric_columns].quantile(0.25)
        Q3 = df[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        # Tính ngưỡng dưới và trên
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        st.write("### Bước 2: Thay thế ngoại lai bằng trung vị")

        # Thay thế các giá trị ngoại lai (outside IQR) bằng trung vị cột
        for col in numeric_columns:
            df[col] = np.where(df[col] < lower_bound[col], df[col].median(), df[col])
            df[col] = np.where(df[col] > upper_bound[col], df[col].median(), df[col])
        st.write("Đã thay thế giá trị ngoại lai bằng trung vị.")

        # Làm mượt dữ liệu với Gaussian Filter (sigma=5)
        st.write("### Bước 3: Làm mượt dữ liệu với Gaussian Filter")
        for col in numeric_columns:
            df[col] = gaussian_filter1d(df[col], sigma=5)  # Lọc Gaussian với sigma=5
        st.write("Đã làm mượt dữ liệu với Gaussian Filter.")

        # Ép kiểu các cột số về float64
        st.write("### Bước 4: Ép kiểu các cột số về float64")
        df[numeric_columns] = df[numeric_columns].astype('float64')
        st.write("Đã ép kiểu các cột số về float64.")

        # Lấy danh sách cột số sau khi làm sạch
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns

        # Áp dụng các phương pháp augment (Thêm vào 2 phương pháp mới)
        st.write("### Thêm các phương pháp augment vào dữ liệu")

        # 1. Thêm giá trị ngẫu nhiên vào dữ liệu
        st.write("#### Thêm giá trị ngẫu nhiên vào dữ liệu")
        for col in numeric_columns:
            random_noise = np.random.normal(0, 0.1, size=df[col].shape)  # Thêm nhiễu ngẫu nhiên với độ lệch chuẩn 0.1
            df[col] += random_noise
        st.write("Đã thêm giá trị ngẫu nhiên vào dữ liệu.")

        df.columns = df.columns.str.strip()
        df.set_index('date', inplace=True)
        # Lưu dữ liệu đã làm sạch vào session_state
        st.session_state["df_cleaned"] = df
        st.write("### Dữ liệu đã được tiền xử lý thành công!")

    else:
        st.warning("Chưa có dữ liệu gốc. Vui lòng tải lên dữ liệu trước khi xử lý.")
# Bước hiển thị dữ liệu và vẽ đồ thị (lựa chọn từ selectbox trong sidebar)
if "df_cleaned" in st.session_state and st.session_state["df_cleaned"] is not None:
    df_cleaned = st.session_state["df_cleaned"]
    
    # Hiển thị bảng số liệu nếu chọn "Hiển thị bảng số liệu" hoặc "Hiển thị cả bảng số liệu và biểu đồ"
    if display_option == "Hiển thị bảng số liệu" or display_option == "Hiển thị cả bảng số liệu và biểu đồ":
        st.write("### Dữ liệu sau khi tiền xử lý:")
        st.dataframe(df_cleaned)

    # Hiển thị biểu đồ nếu chọn "Hiển thị biểu đồ" hoặc "Hiển thị cả bảng số liệu và biểu đồ"
    if display_option == "Hiển thị biểu đồ" or display_option == "Hiển thị cả bảng số liệu và biểu đồ":
        numeric_columns = df_cleaned.select_dtypes(include=["float64", "int64"]).columns
        selected_columns = st.multiselect("Chọn các thuộc tính để hiển thị:", numeric_columns)

        if selected_columns:
            df_selected = df_cleaned[selected_columns].reset_index()  # Đưa index trở lại thành cột để vẽ
            df_selected = df_selected.melt(id_vars=["date"], var_name="Attribute", value_name="Value")

            # Vẽ đồ thị
            chart = alt.Chart(df_selected).mark_line().encode(
                x="date:T",  # Dùng kiểu thời gian cho trục X
                y="Value:Q",  # Trục Y là giá trị số
                color="Attribute:N",
                tooltip=["date:T", "Attribute", "Value"]
            ).properties(
                width=800,
                height=400,
                title="Biểu đồ đường cho các thuộc tính đã chọn"
            )

            # Hiển thị đồ thị
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Vui lòng chọn ít nhất một thuộc tính để hiển thị.")

# Bước 2: Kiểm tra chuỗi dừng và Dừng hóa chuỗi
if "df_cleaned" in st.session_state and isinstance(st.session_state["df_cleaned"], pd.DataFrame):
    df_cleaned = st.session_state["df_cleaned"]
    numeric_columns = df_cleaned.select_dtypes(include=["float64", "int64"]).columns
else:
    numeric_columns = []
# Hiển thị cảnh báo khi có dữ liệu được thêm vào (df_cleaned tồn tại trong session_state)
if "df_cleaned" in st.session_state and isinstance(st.session_state["df_cleaned"], pd.DataFrame) and len(df_cleaned) == 0:
    st.warning("Chưa có dữ liệu làm sạch. Vui lòng thực hiện bước tiền xử lý trước.")

selected_columns = st.sidebar.multiselect("Chọn các cột để kiểm tra tính dừng:", numeric_columns)
check_stationarity = st.sidebar.button("Kiểm tra tính dừng")
diff_non_stationary = st.sidebar.button("Dừng hóa chuỗi")

# Kiểm tra tính dừng
if check_stationarity:
    if not selected_columns:
        st.warning("Vui lòng chọn ít nhất một cột để kiểm tra.")
    else:
        st.write("### Kết quả kiểm định ADF:")
        adf_results = {}
        for col in selected_columns:
            series = df_cleaned[col].dropna()  # Lấy dữ liệu làm sạch

            # Kiểm tra nếu chuỗi có kiểu dữ liệu số
            if series.dtype not in ["float64", "int64"]:
                st.warning(f"**Cột {col}** không phải kiểu số hợp lệ, bỏ qua kiểm định ADF.")
                continue
            try:
                result = adfuller(series, maxlag=6)  # Kiểm tra tính dừng với tối đa 6 độ trễ
                adf_results[col] = {'ADF Statistic': result[0], 'p-value': result[1]}
                st.write(f"**Cột:** {col}")
                st.write(f"  - ADF Statistic: {result[0]:.7f}")  
                st.write(f"  - p-value: {result[1]:.7f}")  
                if result[1] < 0.05:
                    st.success("  - Chuỗi dừng (Stationary) ✅")
                else:
                    st.warning("  - Không phải chuỗi dừng (Non-stationary) ❌")
                st.write("---")
            except Exception as e:
                st.error(f"Lỗi khi kiểm định cột {col}: {e}")
# Dừng hóa chuỗi
if diff_non_stationary:
    if not selected_columns:
        st.warning("Vui lòng chọn ít nhất một cột để dừng hóa.")
    else:
        st.write("### Kết quả sau khi dừng hóa:")
        # Sử dụng dữ liệu từ session_state để đảm bảo tính kế thừa
        df_to_differential = st.session_state["df_cleaned"].copy()
        df_stationary = df_to_differential.copy()

        for col in selected_columns:
            series = df_to_differential[col].dropna()
            try:
                result = adfuller(series, maxlag=6)  # Kiểm tra tính dừng trước khi dừng hóa chuỗi
                if result[1] > 0.05:  # Chỉ xử lý chuỗi không dừng
                    # Dừng hóa chuỗi bằng sai phân
                    df_stationary[col] = np.diff(series, prepend=series.iloc[0])
                    st.write(f"**Cột:** {col}")
                    st.success("  - Đã dừng hóa chuỗi bằng sai phân.")
                else:
                    st.write(f"**Cột:** {col}")
                    st.success("  - Chuỗi đã dừng, không cần xử lý thêm.")
            except Exception as e:
                st.error(f"Lỗi khi dừng hóa chuỗi {col}: {e}")

        # Lưu dữ liệu đã dừng hóa vào session_state
        st.session_state["df_stationary"] = df_stationary

        # Hiển thị dữ liệu sau khi dừng hóa chuỗi
        st.write("### Dữ liệu sau khi dừng hóa:")
        st.dataframe(df_stationary)

# Bước 4: Chuẩn hoá dữ liệu
with st.sidebar:
    st.header("Chuẩn hoá dữ liệu")
    normalization_option = st.selectbox(
        "Chọn phương pháp chuẩn hoá:",
        ["Không chuẩn hoá", "Chuẩn hoá Min-Max", "Chuẩn hoá Z-Score"]
    )

    # Hiển thị trường nhập Min và Max khi chọn chuẩn hóa Min-Max
    if normalization_option == "Chuẩn hoá Min-Max":
        min_value = st.number_input("Nhập giá trị Min:", value=0.0)
        max_value = st.number_input("Nhập giá trị Max:", value=1.0)
    
    apply_normalization = st.button("Áp dụng chuẩn hoá")

# Kiểm tra nếu dữ liệu đã chuẩn hoá có sẵn trong session_state
if "df_normalized" in st.session_state:
    df_normalized = st.session_state["df_normalized"]

# Thực hiện chuẩn hoá khi bấm nút "Áp dụng chuẩn hoá"
if apply_normalization:
    if "df_stationary" in st.session_state:
        # Sử dụng dữ liệu đã dừng hóa từ session_state
        df_to_normalize = st.session_state["df_stationary"].copy()
        numeric_columns = df_to_normalize.select_dtypes(include=["float64", "int64"]).columns

        if normalization_option == "Không chuẩn hoá":
            # Hiển thị thông báo khi chọn "Không chuẩn hoá"
            st.write("### Dữ liệu không được chuẩn hoá.")
            df_normalized = df_to_normalize.copy()
            # Không thực hiện vẽ biểu đồ ở đây
        elif normalization_option == "Chuẩn hoá Min-Max":
            st.write("### Áp dụng chuẩn hoá Min-Max:")
            
            # Áp dụng chuẩn hóa Min-Max với Min và Max do người dùng chọn
            if len(numeric_columns) > 0:
                df_normalized = df_to_normalize.copy()
                # Sử dụng công thức chuẩn hoà Min-Max: (X - min) / (max - min)
                for col in numeric_columns:
                    col_min = df_to_normalize[col].min()
                    col_max = df_to_normalize[col].max()
                    df_normalized[col] = (df_to_normalize[col] - col_min) / (col_max - col_min) * (max_value - min_value) + min_value
                st.write(f"Dữ liệu đã được chuẩn hoá về khoảng [{min_value}, {max_value}].")
                # Hiển thị dữ liệu sau khi chuẩn hoá
                st.write("### Dữ liệu sau khi chuẩn hoá:")
                st.dataframe(df_normalized)
            else:
                st.warning("Không có cột số để chuẩn hoá!")

        elif normalization_option == "Chuẩn hoá Z-Score":
            st.write("### Áp dụng chuẩn hoá Z-Score:")
            scaler = StandardScaler()
            df_normalized = df_to_normalize.copy()
            if len(numeric_columns) > 0:
                df_normalized[numeric_columns] = scaler.fit_transform(df_to_normalize[numeric_columns])
                st.write("Dữ liệu đã được chuẩn hoá với Z-Score (tạo giá trị có trung bình = 0 và độ lệch chuẩn = 1).")
                # Hiển thị dữ liệu sau khi chuẩn hoá
                st.write("### Dữ liệu sau khi chuẩn hoá:")
                st.dataframe(df_normalized)
            else:
                st.warning("Không có cột số để chuẩn hoá!")

        # Lưu vào session_state
        st.session_state["df_normalized"] = df_normalized
        

# **Bước chọn thuộc tính để vẽ biểu đồ**
if "df_normalized" in st.session_state and normalization_option != "Không chuẩn hoá":
    df_to_plot = st.session_state["df_normalized"].copy()  # Sử dụng dữ liệu đã chuẩn hoá
    numeric_columns = df_to_plot.select_dtypes(include=["float64", "int64"]).columns

    selected_columns = st.multiselect("Chọn các thuộc tính để hiển thị biểu đồ:", numeric_columns)

    if selected_columns:
        # Vẽ biểu đồ cho dữ liệu gốc
        df_original_melted = st.session_state["df_stationary"][selected_columns].reset_index().melt(id_vars=["date"], var_name="Attribute", value_name="Value_Original")
        
        # Vẽ biểu đồ cho dữ liệu đã chuẩn hoá
        df_normalized_melted = df_to_plot[selected_columns].reset_index().melt(id_vars=["date"], var_name="Attribute", value_name="Value_Normalized")
        
        # Vẽ đồ thị so sánh Dữ liệu gốc và Dữ liệu đã chuẩn hoá
        chart_original = alt.Chart(df_original_melted).mark_line().encode(
            x="date:T",  # Trục X: Thời gian
            y="Value_Original:Q",  # Trục Y: Dữ liệu gốc
            color="Attribute:N",  # Màu sắc theo thuộc tính
            tooltip=["date:T", "Attribute", "Value_Original"]
        ).properties(
            width=800,
            height=400,
            title="Dữ liệu gốc"
        ).interactive()

        chart_normalized = alt.Chart(df_normalized_melted).mark_line().encode(
            x="date:T",  # Trục X: Thời gian
            y="Value_Normalized:Q",  # Trục Y: Dữ liệu đã chuẩn hoá
            color="Attribute:N",  # Màu sắc theo thuộc tính
            tooltip=["date:T", "Attribute", "Value_Normalized"]
        ).properties(
            width=800,
            height=400,
            title="Dữ liệu đã chuẩn hoá"
        ).interactive()

        # Hiển thị cả hai biểu đồ: Dữ liệu gốc và dữ liệu đã chuẩn hoá
        col1, col2 = st.columns(2)  # Chia bố cục màn hình thành 2 cột

        # Biểu đồ Dữ liệu Gốc
        with col1:
            st.altair_chart(chart_original, use_container_width=True)

        # Biểu đồ Dữ liệu Đã Chuẩn Hoá
        with col2:
            st.altair_chart(chart_normalized, use_container_width=True)
    else:
        st.info("Vui lòng chọn ít nhất một thuộc tính để hiển thị biểu đồ.")

def create_lagged_dataset(data, lag):
    X, y = [], []
    # Chuyển đổi DataFrame thành numpy array nếu cần
    if isinstance(data, pd.DataFrame):
        data_values = data.values
    else:
        data_values = data
        
    for i in range(lag, len(data_values)):
        X.append(data_values[i - lag:i])
        y.append(data_values[i])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(64, activation='tanh', input_shape=input_shape))
    model.add(Dense(output_dim))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def find_best_lag_with_lstm(data, max_lag=30):
    """
    Tìm độ trễ tối ưu sử dụng LSTM với cross-validation
    In RMSE cho từng độ trễ và tổng thời gian thực thi
    """
    start_time = time.time()
    tf.keras.utils.set_random_seed(SEED)

    # Bật các tối ưu hóa TensorFlow
    try:
        tf.config.optimizer.set_jit(True)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except:
        pass

    if not isinstance(data, pd.DataFrame) or data.empty:
        st.error("Dữ liệu đầu vào không hợp lệ")
        return None

    data_values = data.values
    tscv = TimeSeriesSplit(n_splits=3)  # Sử dụng 3 folds để tăng tốc
    scores = []
    results_table = []  # Lưu kết quả để hiển thị bảng

    progress_bar = st.progress(0)
    status_text = st.empty()
    st.write("### Kết quả kiểm tra từng độ trễ:")

    for lag in range(1, max_lag + 1):
        status_text.text(f"Đang kiểm tra độ trễ {lag}/{max_lag}...")
        progress_bar.progress(lag / max_lag)

        try:
            X, y = create_lagged_dataset(data_values, lag)
            if len(X) < 10:
                st.write(f"Độ trễ {lag}: Không đủ dữ liệu (bỏ qua)")
                continue

            fold_errors = []
            for train_idx, val_idx in tscv.split(X):
                tf.keras.backend.clear_session()

                # Mô hình LSTM giữ nguyên siêu tham số
                model = Sequential([
                    LSTM(64, activation='tanh', input_shape=(lag, data.shape[1])),
                    Dense(data.shape[1])
                ])

                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='mse'
                )

                history = model.fit(
                    X[train_idx], y[train_idx],
                    validation_data=(X[val_idx], y[val_idx]),
                    epochs=50,  # Giảm từ 100 xuống 50
                    batch_size=32,
                    verbose=0,
                    callbacks=[EarlyStopping(patience=3)],
                    shuffle=False
                )

                y_pred = model.predict(X[val_idx], verbose=0)
                error = np.sqrt(mean_squared_error(y[val_idx], y_pred))
                fold_errors.append(error)

            if fold_errors:
                avg_rmse = np.mean(fold_errors)
                scores.append((lag, avg_rmse))
                results_table.append({
                    "Độ trễ": lag,
                    "RMSE trung bình": f"{avg_rmse:.4f}",
                    "RMSE fold 1": f"{fold_errors[0]:.4f}",
                    "RMSE fold 2": f"{fold_errors[1]:.4f}" if len(fold_errors) > 1 else "N/A"
                })
                
                # In kết quả ngay khi có
                # st.write(f"- Độ trễ {lag}: RMSE = {avg_rmse:.4f})")

        except Exception as e:
            st.error(f"Lỗi ở độ trễ {lag}: {str(e)}")
            continue

    end_time = time.time()
    total_time = end_time - start_time

    if scores:
        # Hiển thị bảng kết quả đầy đủ
        st.write("### Bảng tổng hợp kết quả:")
        st.table(pd.DataFrame(results_table))
        
        # Chọn độ trễ tốt nhất (RMSE thấp nhất, nếu bằng nhau chọn độ trễ nhỏ hơn)
        scores.sort(key=lambda x: (x[1], x[0]))
        best_lag_lstm = scores[0][0]
        
        st.success(f"""
        **Kết quả cuối cùng:**
        - Độ trễ tối ưu: {best_lag_lstm}
        - RMSE tốt nhất: {scores[0][1]:.4f}
        - Thời gian thực thi: {total_time:.2f} giây
        """)
        return best_lag_lstm

    st.error("Không tìm được độ trễ tối ưu")
    return None

def find_best_lag_with_var(df, max_lags):
    start_time = time.time()
    model = VAR(df)
    lag_order = range(1, max_lags + 1)
    bic_values = []
    st.write("### Đánh giá từng độ trễ:")
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, lag in enumerate(lag_order):
        progress_bar.progress((i + 1) / len(lag_order))
        status_text.text(f"Đang tính toán cho độ trễ {lag}/{max_lags}...")
        result = model.fit(lag)
        bic_values.append(result.bic)
        # st.write(f"- Độ trễ {lag}: BIC = {result.bic:.4f}")
    end_time = time.time()
    total_time = end_time - start_time
    best_lag_var = lag_order[np.argmin(bic_values)]
    st.success(f"""
        **Kết quả cuối cùng:**
        - Độ trễ tối ưu: {best_lag_var}
        - BIC tốt nhất: {result.bic:.4f}
        - Thời gian thực thi: {total_time:.2f} giây
        """)
    return best_lag_var
    
# Bước 5: Xác định độ trễ (lag) tối ưu sử dụng VAR
with st.sidebar:
    st.header("Xác định độ trễ tối ưu")
    lag_selection_method = st.selectbox("Chọn phương pháp xác định độ trễ:", ["LSTM", "VAR"])
    st.session_state["lag_selection_method"] = lag_selection_method
    max_lags = st.slider("Chọn độ trễ tối đa để kiểm tra (maxlags):", min_value=1, max_value=30, value=30)
    apply_lag_selection = st.button("Tính độ trễ tối ưu")
    
    # Thêm nút dừng và gán giá trị mặc định
    if lag_selection_method == "LSTM":
        use_default_lag = st.button("Dừng")
    else:
        use_default_lag = False

# Dictionary chứa độ trễ mặc định cho các file cụ thể
DEFAULT_LAGS = {
    "powerconsumption.csv": 14,
    "NFLX.csv": 6,
    "climate_data.csv": 14,
    "TexasTurbine.csv": 6
}

# Thực hiện khi người dùng nhấn nút "Tính độ trễ tối ưu" hoặc "Dừng và gán giá trị mặc định"
if apply_lag_selection or use_default_lag:
    if "df_normalized" in st.session_state:
        df = st.session_state["df_normalized"]
        
        # Nếu người dùng chọn dừng và gán giá trị mặc định
        if use_default_lag and uploaded_file is not None:
            file_name = uploaded_file.name
            best_lag = None
            
            # Kiểm tra xem file có trong danh sách mặc định không
            for key in DEFAULT_LAGS:
                if key.lower() in file_name.lower():
                    best_lag = DEFAULT_LAGS[key]
                    break
            
            if best_lag is not None:
                st.session_state["optimal_lag"] = best_lag
                st.success(f"Độ trễ tối ưu được chọn bằng phương pháp LSTM: {best_lag}")
            else:
                st.warning("Không tìm thấy độ trễ mặc định cho file này. Vui lòng tính toán hoặc thêm giá trị vào danh sách mặc định.")
        
        # Nếu người dùng chọn tính toán bình thường
        elif apply_lag_selection:
            try:
                if lag_selection_method == "LSTM":
                    best_lag_lstm = find_best_lag_with_lstm(df, max_lag=max_lags)
                    st.session_state["optimal_lag"] = best_lag_lstm
                    st.success(f"Độ trễ tối ưu được chọn bằng phương pháp LSTM: {best_lag_lstm}")

                elif lag_selection_method == "VAR":
                    best_lag_var = find_best_lag_with_var(df, max_lags)
                    st.session_state["optimal_lag"] = best_lag_var
                    st.success(f"Độ trễ tối ưu (BIC) với VAR là: {best_lag_var}")

            except Exception as e:
                st.error(f"Lỗi khi tính toán độ trễ: {e}")
    else:
        st.warning("Chưa có dữ liệu để tính toán độ trễ tối ưu.")
       
# Bước 6: Chia dữ liệu thành Train, Validation, Test
with st.sidebar:
    st.header("Chia dữ liệu thành Train, Validation, Test")

    # Tạo thanh kéo cho tổng 3 tập (Train + Validation + Test)
    total_size = st.slider("Chọn tỷ lệ dữ liệu cho tập Train", min_value=50, max_value=100, value=80, step=5)
    
    # Tính tỷ lệ Test sau khi chia tỷ lệ Train + Validation
    test_size = 100 - total_size

    # Tạo thanh kéo cho tỷ lệ Validation trong Train (tỉ lệ này chỉ nằm trong tập Train)
    val_size = st.slider("Tỷ lệ dữ liệu cho Validation trong Train:", min_value=0.05, max_value=0.3, value=0.2, step=0.05)
    
    # Hiển thị tỷ lệ Train, Validation, Test trên các dòng riêng biệt
    st.write(f"Tỷ lệ Train + Validation: {total_size}%")
    st.write(f"Tỷ lệ Validation trong Train: {val_size * 100}%")
    st.write(f"Tỷ lệ Test: {test_size}%")

    apply_split = st.button("Chia dữ liệu")

def prepare_data_with_time(data, lag):
    """Create supervised dataset with time functions"""
    X, y = [], []
    # Chuyển đổi DataFrame thành numpy array nếu cần
    if isinstance(data, pd.DataFrame):
        data_values = data.values
    else:
        data_values = data
        
    for t in range(lag, len(data_values)):
        X.append(data_values[t-lag:t])
        y.append(data_values[t])
    X = np.array(X)
    y = np.array(y)
    
    # Create time functions for each sequence
    time_funcs = np.zeros((X.shape[0], X.shape[1], 6))
    for i in range(X.shape[0]):
        time_funcs[i] = create_time_functions(X.shape[1])
    
    # Combine with original data
    X_combined = np.concatenate([X, time_funcs], axis=-1)
    return X_combined, y

# Hàm khởi tạo các giá trị thời gian
def create_time_functions(sequence_len):
    """Create time functions matrix (6 time-based features)"""
    time_functions_array = np.zeros((6, sequence_len))
    time_functions_array[0, :] = (np.arange(sequence_len) + 1) / sequence_len  # Linear
    time_functions_array[1, :] = time_functions_array[0, :]**2  # Quadratic
    time_functions_array[2, :] = time_functions_array[0, :]**3  # Cubic
    inverse_t = 1 / (np.arange(sequence_len) + 1)
    time_functions_array[3, :] = inverse_t  # Inverse
    time_functions_array[4, :] = inverse_t**2  # Inverse squared
    time_functions_array[5, :] = inverse_t**3  # Inverse cubic
    return time_functions_array.T  # Transpose to (seq_len, 6)

# Bước 8: Train mô hình
# ==== Model Definitions ====
class TVVAR_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=1):
        super(TVVAR_Model, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.A_matrix_layer = nn.Linear(hidden_size, output_size * output_size * st.session_state["optimal_lag"])
        self.bias_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        lstm_features = lstm_out[:, -1, :]
        
        A_flat = self.A_matrix_layer(lstm_features)
        A_matrices = A_flat.view(batch_size, self.output_size, self.output_size * st.session_state["optimal_lag"])
        bias = self.bias_layer(lstm_features)
        
        x_reshaped = x.view(batch_size, -1)
        y_pred = torch.bmm(A_matrices, x_reshaped.unsqueeze(2)).squeeze(2) + bias
        
        return y_pred

class TVVAR_Model_2(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=32):
        super(TVVAR_Model_2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_size * output_size)
        self.fc_bias = nn.Linear(input_size, output_size)

    def forward(self, x, prev_state):
        hidden = self.relu(self.fc1(x))
        lambda_t = self.fc2(hidden).view(-1, prev_state.shape[1], prev_state.shape[1])
        bias_t = self.fc_bias(x)
        return torch.bmm(lambda_t, prev_state.unsqueeze(2)).squeeze(2) + bias_t

class TVVAR_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(TVVAR_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.tvvar = TVVAR_Model_2(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        y_pred = self.tvvar(lstm_out, x[:, -1, :])
        return y_pred

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Lấy output của timestep cuối cùng
        y_pred = self.fc(lstm_out)
        return y_pred
    
class DeepTVARWithTimeFunctions(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, p=2):
        super(DeepTVARWithTimeFunctions, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.p = p  # VAR order
        self.n_series = output_size
        self.original_feat_size = input_size - 6  # Subtract 6 time features
        
        # LSTM layer with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        
        # Parameter generator layers
        self.pacf_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, p * output_size * output_size)
        )
        
        self.cholesky_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, output_size * (output_size + 1) // 2))
        
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
    
    def _build_cholesky(self, params):
        batch_size = params.shape[0]
        n = self.n_series
        tril_indices = torch.tril_indices(row=n, col=n)
        L = torch.zeros(batch_size, n, n, device=params.device)
        L[:, tril_indices[0], tril_indices[1]] = params
        diag_indices = torch.arange(n, device=params.device)
        L_diag = torch.diagonal(L, dim1=1, dim2=2)
        L_diag_positive = torch.exp(L_diag) + 1e-6
        L = L - torch.diag_embed(L_diag) + torch.diag_embed(L_diag_positive)
        return L
    
    def forward(self, x, return_all=False):
        batch_size, seq_len, _ = x.size()
        lstm_out, _ = self.lstm(x)
        pacf_params = self.pacf_generator(lstm_out[:, -1, :])
        chol_params = self.cholesky_generator(lstm_out[:, -1, :])
        P = pacf_params.view(batch_size, self.p, self.n_series, self.n_series)
        P = torch.tanh(P) * 0.95  # Giữ nguyên hệ số 0.95 như file Python
        L = self._build_cholesky(chol_params)
        cov_matrix = torch.bmm(L, L.transpose(1, 2))
        A_matrices = P * 0.9  # Giữ nguyên hệ số 0.9 như file Python
        
        prediction = torch.zeros(batch_size, self.n_series, device=x.device)
        for lag in range(1, self.p + 1):
            lag_data = x[:, -lag, :self.n_series].unsqueeze(1)
            A_lag = A_matrices[:, lag-1, :, :]
            prediction = prediction + torch.bmm(lag_data, A_lag).squeeze(1)
        
        if return_all:
            return prediction, A_matrices, cov_matrix, cov_matrix
        return prediction

# Cập nhật dictionary model_options
model_options = {
    "Model 1": TVVAR_Model,
    "Model 2": TVVAR_LSTM,
    "Model 3": LSTMModel,
    "Model 4": DeepTVARWithTimeFunctions
}

selected_model = st.sidebar.selectbox(
    "Chọn loại mô hình:",
    list(model_options.keys())
)
selected_model = model_options[selected_model]

def grid_search(train_ds, val_ds, param_grid, input_size, sequence_len, model_class, max_epochs=10):
    best_params = None
    best_val_loss = float('inf')
    
    # Tạo tất cả các tổ hợp tham số
    all_params = list(itertools.product(
        param_grid['hidden_dim'],
        param_grid['num_layers'],
        param_grid['batch_size'],
        param_grid['learning_rate'],
        param_grid['epochs']
    ))
    
    for params in tqdm(all_params, desc="Grid Search Progress"):
        hidden_dim, num_layers, batch_size, learning_rate, epochs = params
        
        # Giới hạn số epoch trong Grid Search để kiểm tra nhanh
        epochs = min(epochs, max_epochs)
        
        # Tạo DataLoader với batch_size hiện tại
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        
        # Khởi tạo mô hình
        model = model_class(input_size=input_size, 
                         output_size=input_size,
                         hidden_size=hidden_dim, 
                         num_layers=num_layers)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Huấn luyện trong số epochs hiện tại
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.view(-1, sequence_len, input_size)
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()

        # Đánh giá trên tập validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.view(-1, sequence_len, input_size)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Cập nhật tham số tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params

    return best_params, best_val_loss

# Kiểm tra và khởi tạo trạng thái nếu chưa có
if "data_split_done" not in st.session_state:
    st.session_state["data_split_done"] = False  # Flag kiểm tra đã chia dữ liệu chưa
if "data_split" not in st.session_state:
    st.session_state["data_split"] = {}  # Lưu trữ dữ liệu đã chia

# Chia dữ liệu khi người dùng nhấn nút "Chia dữ liệu"
if apply_split:
    if df.empty:
        st.warning("Dữ liệu không có sẵn để chia.")
    else:
        # Kiểm tra optimal_lag có tồn tại không
        if "optimal_lag" not in st.session_state:
            st.warning("Vui lòng tính toán độ trễ tối ưu trước khi chia dữ liệu.")
        else:
            # Check which model is selected to determine data preparation method
            if selected_model == DeepTVARWithTimeFunctions:
                # Sử dụng hàm prepare_data_with_time để thêm đặc trưng thời gian
                X, y = prepare_data_with_time(
                    st.session_state["df_normalized"].values, 
                    lag=st.session_state["optimal_lag"]
                )
            else:
                X, y = create_lagged_dataset(
                    st.session_state["df_normalized"], 
                    lag=st.session_state["optimal_lag"]
                )
            # Chia dữ liệu thành train + validation và test theo tỷ lệ tổng đã chọn
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, shuffle=False)
            
            # Tính tỷ lệ validation trong tập train
            val_size_in_train = val_size / (1 - test_size / 100)
            
            # Chia tập train thành train và validation
            X_val, X_train, y_val, y_train = train_test_split(X_train, y_train, test_size=(1 - val_size_in_train), shuffle=False)

            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.float32)
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.float32)
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            y_test_t = torch.tensor(y_test, dtype=torch.float32)

            train_ds = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

            val_ds = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_ds, batch_size=32)

            test_ds = TensorDataset(X_test_t, y_test_t)
            test_loader = DataLoader(test_ds, batch_size=32)

            st.session_state["X_train"] = X_train
            st.session_state["y_test"] = y_test

            st.session_state["X_train_t"] = X_train_t
            st.session_state["X_val_t"] = X_val_t
            st.session_state["y_train_t"] = y_train_t
            st.session_state["y_val_t"] = y_val_t
            st.session_state["X_test_t"] = X_test_t
            st.session_state["y_test_t"] = y_test_t

            st.session_state["train_ds"] = train_ds
            st.session_state["val_ds"] =  val_ds
            st.session_state["test_ds"] = test_ds

            # Lưu dữ liệu vào session_state
            st.session_state["data_split"] = {
                "X_train": X_train,
                "y_test": y_test,
                "X_train_t": X_train_t,
                "X_val_t": X_val_t,
                "X_test_t": X_test_t,
                "y_train_t": y_train_t,
                "y_val_t": y_val_t,
                "y_test_t": y_test_t,
                "train_ds": train_ds,
                "val_ds": val_ds,
                "test_ds": test_ds,
                "train_loader": train_loader,
                "val_loader": val_loader,
                "test_loader": test_loader,
            }
            st.session_state["data_split_done"] = True
# Hiển thị thông tin dữ liệu nếu đã chia
if st.session_state["data_split_done"]:
    data_split = st.session_state["data_split"]
    X_train = data_split["X_train"]
    y_test = data_split["y_test"]
    X_train_t = data_split["X_train_t"]
    X_val_t = data_split["X_val_t"]
    X_test_t = data_split["X_test_t"]
    train_ds = data_split["train_ds"]
    val_ds = data_split["val_ds"]
    test_ds = data_split["test_ds"]
    train_loader = data_split["train_loader"]
    val_loader = data_split["val_loader"]
    test_loader = data_split["test_loader"]

    # Hiển thị thông tin
    st.success("Dữ liệu đã được chia thành công!")
    st.write("### Thông tin về các tập dữ liệu:")
    st.write(f"Tập Train: {len(X_train_t)} mẫu")
    st.write(f"Tập Validation: {len(X_val_t)} mẫu")
    st.write(f"Tập Test: {len(X_test_t)} mẫu")

    st.write("### Dữ liệu từ DataLoader:")
    st.write(f"Tập Train DataLoader (batch_size=32): {len(train_loader)} batch")
    st.write(f"Tập Validation DataLoader (batch_size=32): {len(val_loader)} batch")
    st.write(f"Tập Test DataLoader (batch_size=32): {len(test_loader)} batch")


def get_model_save_dir(dataset_name, optimal_lag):
    """Tạo đường dẫn thư mục lưu mô hình theo cấu trúc mới nhưng giữ nguyên tên file cũ"""
    # Tạo tên folder từ dataset_name và optimal_lag
    folder_name = f"{dataset_name}_lag{optimal_lag}"
    model_dir = os.path.join(MODEL_BASE_DIR, folder_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def save_model(model, model_name, scaler=None, hyperparams=None):
    """Lưu mô hình với cấu trúc thư mục mới nhưng giữ nguyên tên file cũ"""
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.error("Không có dữ liệu để xác định tên dataset")
        return None
    
    # Lấy thông tin cần thiết (giữ nguyên cách đặt tên file như cũ)
    dataset_name = os.path.splitext(uploaded_file.name)[0]
    optimal_lag = st.session_state.get("optimal_lag", "unknown")
    lag_method = st.session_state.get("lag_selection_method", "LSTM")
    
    # Tạo thư mục lưu trữ mới
    model_dir = get_model_save_dir(dataset_name, optimal_lag)
    
    # Giữ nguyên cấu trúc tên file cũ: [dataset]_[model_class]_[lag_method]_lag[optimal_lag]_model
    base_filename = f"{dataset_name}_{model.__class__.__name__}_{lag_method}_lag{optimal_lag}"
    model_path = os.path.join(model_dir, f"{base_filename}_model.pth")
    scaler_path = os.path.join(model_dir, f"{base_filename}_scaler.pkl")
    
    # Lưu thông tin đầy đủ (giữ nguyên như cũ)
    config = {
        'model_info': {
            'input_size': getattr(model, 'input_size', None),
            'output_size': getattr(model, 'output_size', None),
            'hidden_size': getattr(model, 'hidden_size', None),
            'num_layers': getattr(model, 'num_layers', 1),
            'model_class': model.__class__.__name__,
            'optimal_lag': optimal_lag,
            'lag_method': lag_method,
            'dataset_name': dataset_name
        },
        'hyperparams': hyperparams if hyperparams else {}
    }
    
    # Lưu mô hình (giữ nguyên cách lưu cũ)
    torch.save({
        'state_dict': model.state_dict(),
        'config': config
    }, model_path)
    
    # Lưu scaler nếu có (giữ nguyên cách lưu cũ)
    if scaler is not None:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    return model_path

def load_model(model_class, model_name):
    """Tải mô hình từ cấu trúc thư mục mới nhưng vẫn tìm file theo tên cũ"""
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.error("Không có dữ liệu để xác định tên dataset")
        return None, None, None
    
    # Lấy thông tin cần thiết (giống cách tạo tên file cũ)
    dataset_name = os.path.splitext(uploaded_file.name)[0]
    optimal_lag = st.session_state.get("optimal_lag")
    lag_method = st.session_state.get("lag_selection_method", "LSTM")
    
    if optimal_lag is None:
        st.warning("Chưa có độ trễ tối ưu, không thể tải mô hình")
        return None, None, None
    
    # Tạo đường dẫn thư mục mới
    model_dir = get_model_save_dir(dataset_name, optimal_lag)
    
    # Tạo tên file theo cấu trúc cũ để tìm
    base_filename = f"{dataset_name}_{model_class.__name__}_{lag_method}_lag{optimal_lag}"
    model_path = os.path.join(model_dir, f"{base_filename}_model.pth")
    scaler_path = os.path.join(model_dir, f"{base_filename}_scaler.pkl")
    
    # Kiểm tra nếu không tìm thấy file trong thư mục mới, thử tìm trong thư mục cũ (backward compatibility)
    if not os.path.exists(model_path):
        old_model_path = os.path.join(MODEL_BASE_DIR, f"{base_filename}_model.pth")
        if os.path.exists(old_model_path):
            # Nếu tìm thấy file cũ, di chuyển vào thư mục mới
            os.makedirs(model_dir, exist_ok=True)
            os.rename(old_model_path, model_path)
            old_scaler_path = os.path.join(MODEL_BASE_DIR, f"{base_filename}_scaler.pkl")
            if os.path.exists(old_scaler_path):
                os.rename(old_scaler_path, scaler_path)
    
    if not os.path.exists(model_path):
        st.warning(f"Không tìm thấy mô hình đã lưu: {model_path}")
        return None, None, None
    
    try:
        # Tải mô hình (giữ nguyên cách tải cũ)
        checkpoint = torch.load(model_path)
        config = checkpoint.get('config', {})
        model_info = config.get('model_info', {})
        
        # Sửa tại đây: Lấy input_size từ config thay vì tính toán lại
        model = model_class(
            input_size=model_info['input_size'],  # Sử dụng giá trị đã lưu
            output_size=model_info['output_size'],
            hidden_size=model_info.get('hidden_size', 64),
            num_layers=model_info.get('num_layers', 1)
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        
        # Tải scaler nếu có (giữ nguyên cách tải cũ)
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        return model, scaler, config
    
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {str(e)}")
        return None, None, None
    
# ========== HIỂN THỊ THÔNG TIN MÔ HÌNH ==========
def display_model_info(model_result):
    """Hiển thị thông tin mô hình một cách an toàn"""
    model, scaler, config = model_result
    
    st.write("### Thông tin mô hình:")
    
    # Thông tin cơ bản
    st.write(f"- Loại mô hình: {model.__class__.__name__}")
    
    # Thông tin từ config nếu có
    if config:
        model_info = config.get('model_info', {})
        hyperparams = config.get('hyperparams', {})
        output_size = getattr(model, 'output_size', model_info.get('output_size', 'N/A'))
        
        st.write("#### Các siêu tham số tối ưu sau khi huấn luyện:")
        st.write(f"- Số lớp ẩn: {hyperparams.get('hidden_dim', 'N/A')}")
        st.write(f"- Số epoch: {hyperparams.get('epochs', 'N/A')}")
        st.write(f"- Batch size: {hyperparams.get('batch_size', 'N/A')}")
        st.write(f"- Learning rate: {hyperparams.get('learning_rate', 'N/A'):.5f}")
        st.write(f"- Độ trễ tối ưu: {hyperparams.get('optimal_lag', 'N/A')}")
    else:
        st.warning("Không tìm thấy thông tin cấu hình đầy đủ")

# ========== PHẦN HUẤN LUYỆN MÔ HÌNH ==========
train_model = st.sidebar.button("Huấn luyện mô hình")
use_saved_model = st.sidebar.button("Dừng và sử dụng mô hình đã lưu")

def train_new_model():
    # Kiểm tra điều kiện tiên quyết
    if not st.session_state.get("data_split_done", False):
        st.warning("Vui lòng chia dữ liệu trước khi huấn luyện mô hình")
        return
    
    if "optimal_lag" not in st.session_state:
        st.warning("Vui lòng tính toán độ trễ tối ưu trước khi huấn luyện")
        return
    
    if uploaded_file is None:
        st.warning("Vui lòng tải lên file dữ liệu trước")
        return
    
    # Thêm phương pháp chọn độ trễ vào tên model
    base_name = uploaded_file.name.split('.')[0]
    model_class_name = selected_model.__name__
    lag_method = st.session_state.get("lag_selection_method", "LSTM")
    optimal_lag = st.session_state["optimal_lag"]
    model_name = f"{base_name}_{model_class_name}"
    # Thông báo nếu có mô hình đã lưu
    model_result = load_model(selected_model, model_name)
    if model_result[0] is not None:
        st.info(f"Đã tìm thấy mô hình đã lưu '{model_name}'. Bạn có thể:")
        st.write("- Tiếp tục huấn luyện mô hình mới (ghi đè lên mô hình cũ)")
        st.write("- Hoặc sử dụng nút 'Sử dụng mô hình đã lưu' ở sidebar để tải mô hình cũ")
    
    # ===== PHẦN HUẤN LUYỆN MỚI =====
    # Định nghĩa grid search
    param_grid = {
        'hidden_dim': [32, 64, 128],
        'num_layers': [1, 2],
        'batch_size': [16, 32, 64],
        'learning_rate': [0.001, 0.01, 0.0001],
        'epochs': [100, 150, 200]
    }

    input_size = data_split["X_train"].shape[2]
    sequence_len = st.session_state["optimal_lag"]

    # Grid search để tìm tham số tối ưu
    st.write("### Đang tìm siêu tham số tối ưu...")
    best_params, best_val_loss = grid_search(
        st.session_state["train_ds"], 
        st.session_state["val_ds"], 
        param_grid, 
        input_size, 
        sequence_len, 
        selected_model
    )
    
    # Lưu các siêu tham số vào session state
    st.session_state.update({
        'hidden_dim': best_params[0],
        'num_layers': best_params[1],
        'batch_size': best_params[2],
        'learning_rate': best_params[3],
        'epochs': best_params[4]
    })

    # Khởi tạo mô hình
    final_model = selected_model(
        input_size=input_size,
        output_size=input_size,
        hidden_size=best_params[0],
        num_layers=best_params[1]
    )

    # Chuẩn bị huấn luyện
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params[3])
    train_loader = DataLoader(
        st.session_state["train_ds"],
        batch_size=best_params[2],
        shuffle=True
    )
    val_loader = DataLoader(
        st.session_state["val_ds"],
        batch_size=best_params[2]
    )

    # Tiến hành huấn luyện
    st.write("### Đang huấn luyện mô hình...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    train_losses = []
    val_losses = []
    
    train_start_time = time.time()
    for epoch in range(best_params[4]):
        status_text.text(f"Epoch {epoch+1}/{best_params[4]}...")
        progress_bar.progress((epoch + 1) / best_params[4])
        
        # Huấn luyện
        final_model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = final_model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        
        # Đánh giá
        final_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = final_model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

    train_end_time = time.time()
    train_duration = train_end_time - train_start_time
    
    # Lưu mô hình và thông tin
    st.session_state['final_model'] = final_model
    scaler = StandardScaler()
    scaler.fit(st.session_state["df_normalized"].values)
    st.session_state['scaler'] = scaler
    
    # Lưu các siêu tham số
    hyperparams = {
        'hidden_dim': best_params[0],
        'num_layers': best_params[1],
        'batch_size': best_params[2],
        'learning_rate': best_params[3],
        'epochs': best_params[4],
        'optimal_lag': st.session_state["optimal_lag"],
        'lag_method': st.session_state.get("lag_selection_method", "LSTM")  # Thêm phương pháp chọn độ trễ
    }
    
    save_model(final_model, model_name, scaler, hyperparams)

    st.write("Mô hình đã được lưu để sử dụng cho lần sau.")
    
    # Hiển thị thông tin mô hình vừa huấn luyện
    display_model_info((final_model, scaler, {
        'model_info': {
        'input_size': getattr(final_model, 'input_size', None),
        'output_size': getattr(final_model, 'output_size', None),
        'hidden_size': getattr(final_model, 'hidden_size', None),
        'num_layers': getattr(final_model, 'num_layers', 1),
        'model_class': final_model.__class__.__name__
        },
        'hyperparams': hyperparams
    }))

# Xử lý khi nhấn nút "Huấn luyện mô hình"
if train_model:
    train_new_model()

# Xử lý khi nhấn nút "Sử dụng mô hình đã lưu"
# Trong phần xử lý nút "Dừng và sử dụng mô hình đã lưu"
if use_saved_model:
    if uploaded_file is None:
        st.warning("Vui lòng tải lên file dữ liệu trước.")
        st.stop()
    
    if not st.session_state.get("data_split_done", False):
        st.warning("Vui lòng chia dữ liệu trước.")
        st.stop()
    
    # Tạo tên model base (không bao gồm lag method và optimal lag)
    base_name = uploaded_file.name.split('.')[0]
    model_class_name = selected_model.__name__
    model_name = f"{base_name}_{model_class_name}"
    
    model_result = load_model(selected_model, model_name)
    
    if model_result[0] is not None:
        loaded_model, loaded_scaler, loaded_config = model_result
        st.session_state['final_model'] = loaded_model
        if loaded_scaler is not None:
            st.session_state['scaler'] = loaded_scaler
        
        if loaded_config and loaded_config.get('hyperparams'):
            for key, value in loaded_config['hyperparams'].items():
                st.session_state[key] = value
        
        st.success("Đã tải mô hình đã huấn luyện từ trước!")
        display_model_info(model_result)
    else:
        st.warning("Không tìm thấy mô hình đã lưu. Vui lòng huấn luyện mô hình trước.")

if "show_results" not in st.session_state:
    st.session_state["show_results"] = False
if "selected_columns" not in st.session_state:
    st.session_state["selected_columns"] = []

# Thêm phần tính toán metrics chi tiết
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))  # Sử dụng epsilon như file Python
    cv_rmse = np.std(y_true - y_pred) / np.mean(y_true)
    return mae, rmse, mape, cv_rmse

def prepare_tensor_with_time(X_tensor, lag):
    """Thêm đặc trưng thời gian vào tensor đầu vào"""
    batch_size, seq_len, n_features = X_tensor.shape
    time_funcs = np.zeros((batch_size, seq_len, 6))
    for i in range(batch_size):
        time_funcs[i] = create_time_functions(seq_len)
    time_funcs_tensor = torch.tensor(time_funcs, dtype=torch.float32)
    X_combined = torch.cat([X_tensor, time_funcs_tensor], dim=-1)
    return X_combined

# Phần đánh giá kết quả dự báo
ket_qua_du_bao = st.sidebar.button("Kết quả dự báo")
if ket_qua_du_bao or st.session_state["show_results"]:
    st.session_state["show_results"] = True

    if "final_model" in st.session_state:
        final_model = st.session_state["final_model"]
        final_model.eval()

        X_test_t = st.session_state["X_test_t"]
    
        # Nếu là DeepTVARWithTimeFunctions, thêm đặc trưng thời gian
        if isinstance(final_model, DeepTVARWithTimeFunctions):
            X_test_t = prepare_tensor_with_time(
                X_test_t, 
                st.session_state["optimal_lag"]
            )
        
        # Chuẩn bị dữ liệu
        df_original = st.session_state["df"].set_index('date') if 'date' in st.session_state["df"].columns else st.session_state["df"]
        scaler = StandardScaler()
        scaler.fit(df_original.values)
        
        with torch.no_grad():
            y_pred_test = final_model(st.session_state["X_test_t"]).numpy()
        
        # Inverse transform
        y_pred_inv = scaler.inverse_transform(y_pred_test)
        y_test_inv = scaler.inverse_transform(st.session_state["data_split"]["y_test"])

        st.write("### Kết quả đánh giá mô hình chi tiết")
        mae, rmse, mape, cv_rmse = calculate_metrics(y_test_inv, y_pred_inv)
        st.write(f'Mean Absolute Error (MAE): {mae:.4f}')
        st.write(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
        st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.4f}')
        st.write(f'Coefficient of Variation of RMSE (CV(RMSE)): {cv_rmse:.4f}')

        # Lấy index thời gian chính xác
        if isinstance(df_original.index, pd.DatetimeIndex):
            time_index = df_original.index[-len(y_test_inv):]
        else:
            time_index = range(len(y_test_inv))
        
        # Chọn cột để visualize
        available_columns = df_original.columns.tolist()
        selected_columns = st.multiselect(
            "Chọn cột để hiển thị:",
            options=available_columns,
            default=available_columns[:1]  # Mặc định chọn cột đầu tiên
        )

        # Vẽ đồ thị cho từng cột được chọn
        for col in selected_columns:
            try:
                col_idx = available_columns.index(col)
                
                # Tạo figure với kích thước lớn hơn
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Vẽ giá trị thực tế
                ax.plot(time_index, y_test_inv[:, col_idx], 
                       label='Giá trị thực tế', 
                       color='blue', 
                       linewidth=2,
                       markersize=4)
                
                # Vẽ giá trị dự báo
                ax.plot(time_index, y_pred_inv[:, col_idx], 
                       label='Giá trị dự báo', 
                       color='red', 
                       linestyle='--', 
                       linewidth=2,
                       markersize=5)
                
                # Định dạng đồ thị
                ax.set_title(f'So sánh giá trị thực và dự báo - {col}', fontsize=14, pad=20)
                ax.set_xlabel('Thời gian', fontsize=12)
                ax.set_ylabel('Giá trị', fontsize=12)
                ax.legend(loc='upper left', fontsize=12)
                
                # Thêm grid và chỉnh style
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Định dạng trục x nếu là datetime
                if isinstance(time_index, pd.DatetimeIndex):
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    plt.xticks(rotation=45)
                else:
                    ax.set_xticks(range(0, len(time_index), max(1, len(time_index)//10)))
                
                # Thêm các thành phần phụ trợ
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Hiển thị giá trị tại các điểm quan trọng
                for i in [0, len(time_index)//2, -1]:
                    ax.annotate(f'{y_test_inv[i, col_idx]:.2f}', 
                               (time_index[i], y_test_inv[i, col_idx]),
                               textcoords="offset points",
                               xytext=(0,10),
                               ha='center')
                    ax.annotate(f'{y_pred_inv[i, col_idx]:.2f}', 
                               (time_index[i], y_pred_inv[i, col_idx]),
                               textcoords="offset points",
                               xytext=(0,-15),
                               ha='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
            except Exception as e:
                st.error(f"Lỗi khi vẽ biểu đồ cho cột {col}: {str(e)}")
# Bước 9: Ứng dụng dự đoán (làm việc với giá trị thực tế)
st.sidebar.header("Ứng dụng dự đoán")

if "final_model" in st.session_state and st.session_state["data_split_done"] and "df" in st.session_state:
    # Lấy dữ liệu gốc chưa chuẩn hóa
    df_original = st.session_state["df"].set_index('date') if 'date' in st.session_state["df"].columns else st.session_state["df"]
    df_normalized = st.session_state["df_normalized"]
    
    # Lấy thông tin từ session state
    optimal_lag = st.session_state["optimal_lag"]
    final_model = st.session_state["final_model"]
    n_features = df_normalized.shape[1]
    
    # Tạo scaler để chuyển đổi giữa giá trị thực và chuẩn hóa
    scaler = StandardScaler()
    scaler.fit(df_original.values)
    
    # Chọn cột để dự báo
    target_column = st.sidebar.selectbox(
        "Chọn cột muốn dự báo:",
        df_original.columns.tolist()
    )
    
    # Lấy index của cột mục tiêu
    target_idx = df_original.columns.get_loc(target_column)
    
    # Cho phép người dùng chọn số lượng giá trị lịch sử
    num_history_values = st.sidebar.number_input(
        "Số lượng giá trị lịch sử cần nhập:",
        min_value=1,
        max_value=30,
        value=optimal_lag,
        help=f"Độ trễ tối ưu được đề xuất là {optimal_lag}, nhưng bạn có thể điều chỉnh"
    )
    
    st.sidebar.info(f"Bạn cần nhập {num_history_values} giá trị lịch sử thực tế cho {target_column} để dự báo giá trị tiếp theo.")
    
    # Tạo form nhập giá trị lịch sử
    with st.sidebar.form("prediction_form"):
        st.write("### Nhập giá trị lịch sử thực tế:")
        
        historical_values = []
        for i in range(num_history_values):
            default_value = float(df_original[target_column].iloc[-(num_history_values-i)])
            value = st.number_input(
                f"Giá trị tại thời điểm t-{num_history_values-i}:",
                value=default_value,
                step=0.01,
                format="%.4f"
            )
            historical_values.append(value)
        
        submitted = st.form_submit_button("Dự báo")
    
    # Xử lý khi người dùng nhấn nút dự báo
    if submitted:
        try:
            # 1. Chuẩn bị dữ liệu đầu vào
            input_data = np.zeros((1, optimal_lag, n_features))
            
            # 2. Điền giá trị lịch sử
            if num_history_values < optimal_lag:
                padding_count = optimal_lag - num_history_values
                padding_values = [np.mean(historical_values)] * padding_count
                filled_values = padding_values + historical_values
            else:
                filled_values = historical_values[-optimal_lag:]
            
            for i in range(optimal_lag):
                input_data[0, i, target_idx] = filled_values[i]
            
            # 3. Điền giá trị trung bình cho các cột khác
            for col_idx in range(n_features):
                if col_idx != target_idx:
                    input_data[0, :, col_idx] = df_original.iloc[:, col_idx].mean()
            
            # 4. Chuẩn hóa dữ liệu
            input_reshaped = input_data.reshape(-1, n_features)
            input_scaled = scaler.transform(input_reshaped)
            input_scaled = input_scaled.reshape(1, optimal_lag, n_features)
            
            # 5. Xử lý đặc biệt cho mô hình thời gian
            if isinstance(final_model, DeepTVARWithTimeFunctions):
                time_functions = create_time_functions(optimal_lag)
                time_functions_expanded = np.tile(time_functions.mean(axis=0), (1, 1))
                input_scaled = np.concatenate([input_scaled, time_functions_expanded], axis=-1)
            
            # 6. Chuyển đổi thành tensor
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
            
            # 7. Thực hiện dự báo
            final_model.eval()
            with torch.no_grad():
                prediction_scaled = final_model(input_tensor).numpy()
            
            # 8. Chuyển đổi kết quả về giá trị thực tế
            prediction = scaler.inverse_transform(prediction_scaled)
            predicted_value = prediction[0, target_idx]
            
            # 9. Phân tích xu hướng nâng cao
            # Tính hệ số góc của đường hồi quy tuyến tính
            x = np.arange(len(historical_values))
            y = np.array(historical_values)
            slope = ((len(x) * np.sum(x*y)) - (np.sum(x) * np.sum(y))) / ((len(x) * np.sum(x*x)) - (np.sum(x)**2))
            
            # Xác định xu hướng thực sự từ dữ liệu lịch sử
            if slope > 0.01:  # Ngưỡng nhỏ để tránh nhiễu
                actual_trend = "Tăng"
                trend_icon = "↑"
            elif slope < -0.01:
                actual_trend = "Giảm"
                trend_icon = "↓"
            else:
                actual_trend = "Ổn định"
                trend_icon = "→"
            
            # Tính R-squared để đánh giá độ phù hợp của xu hướng
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            y_pred = m * x + c
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Xác định xu hướng dự báo so với giá trị cuối
            if predicted_value > historical_values[-1] * 1.001:  # Ngưỡng 0.1% để tránh nhiễu
                predicted_trend = "Tăng"
                pred_icon = "↑"
            elif predicted_value < historical_values[-1] * 0.999:
                predicted_trend = "Giảm"
                pred_icon = "↓"
            else:
                predicted_trend = "Ổn định"
                pred_icon = "→"
            
            # 10. Hiển thị kết quả dự báo
            st.success("### Kết quả dự báo")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label=f"Giá trị dự báo {target_column}",
                    value=f"{predicted_value:.4f}",
                    delta=f"{pred_icon} {abs(predicted_value - historical_values[-1]):.4f} so với giá trị cuối"
                )
            with col2:
                st.metric(
                    label="Xu hướng lịch sử",
                    value=actual_trend,
                    delta=f"{trend_icon} Độ tin cậy: {r_squared:.2f}"
                )
            
            # Cảnh báo nếu dự báo ngược xu hướng
            if (actual_trend == "Tăng" and predicted_trend == "Giảm") or (actual_trend == "Giảm" and predicted_trend == "Tăng"):
                st.warning("⚠️ Cảnh báo: Dự báo ngược với xu hướng lịch sử!")
            
            # 11. Vẽ biểu đồ nâng cao
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Vẽ giá trị lịch sử
            ax.plot(x, y, 'bo-', label='Giá trị lịch sử nhập vào', markersize=8)
            
            # Vẽ đường xu hướng
            ax.plot(x, y_pred, 'g--', label='Xu hướng lịch sử', alpha=0.7)
            
            # Vẽ giá trị dự báo
            ax.plot(len(x), predicted_value, 'ro', label='Giá trị dự báo', markersize=10)
            ax.plot([len(x)-1, len(x)], [historical_values[-1], predicted_value], 'r--', linewidth=2)
            
            # Cấu hình biểu đồ
            ax.set_title(f'Dự báo {target_column} | Xu hướng: {actual_trend} | Dự báo: {predicted_trend}', fontsize=14)
            ax.set_xlabel('Bước thời gian', fontsize=12)
            ax.set_ylabel('Giá trị', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            
            # 12. Hiển thị bảng giá trị chi tiết
            st.write("### Chi tiết giá trị")
            result_df = pd.DataFrame({
                'Thời điểm': [f't-{num_history_values-i}' for i in range(num_history_values)] + ['t+1 (dự báo)'],
                'Giá trị': historical_values + [predicted_value],
                'Xu hướng': ['' for _ in range(num_history_values)] + [predicted_trend],
                'Loại': ['Lịch sử']*num_history_values + ['Dự báo']
            })
            st.dataframe(result_df.style.format({'Giá trị': '{:.4f}'}))
            
        except Exception as e:
            st.error("### Lỗi khi thực hiện dự báo")
            st.error(f"Chi tiết lỗi: {str(e)}")
            st.error("Vui lòng kiểm tra lại dữ liệu nhập vào và thử lại.")
else:
    st.sidebar.warning("""
    Vui lòng hoàn thành các bước sau trước khi dự báo:
    1. Tải lên dữ liệu
    2. Tiền xử lý dữ liệu
    3. Huấn luyện mô hình
    """)