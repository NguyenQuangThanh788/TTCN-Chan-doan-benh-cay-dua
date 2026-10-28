import os
import sys
import json
import argparse
from pathlib import Path

# Đảm bảo mã hóa UTF-8 cho console trên Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import tensorflow as tf
from tensorflow import keras


# Định nghĩa các đuôi file ảnh hợp lệ
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_image_paths(dataset_dir: str):
    """
    1. Đọc và thu thập danh sách đường dẫn tất cả các file ảnh từ thư mục dataset/.
    Hỗ trợ cả trường hợp ảnh nằm trực tiếp trong dataset/ hoặc phân cấp theo thư mục lớp.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"[CẢNH BÁO] Thư mục '{dataset_dir}' không tồn tại.")
        return []

    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in VALID_IMAGE_EXTENSIONS:
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

    # Sắp xếp đường dẫn để đảm bảo thứ tự nhất quán
    image_paths.sort()
    return image_paths


def load_teacher_model(model_path: str):
    """
    2. Load mô hình Keras (teacher model).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy file mô hình tại: '{model_path}'. "
            f"Vui lòng kiểm tra lại đường dẫn mô hình hoặc chỉ định qua đối số --model-path."
        )

    print(f"[THÔNG TIN] Đang load mô hình từ '{model_path}'...")
    model = keras.models.load_model(model_path)
    print("[THÔNG TIN] Load mô hình thành công.")
    return model


def determine_target_size(model, user_img_size=None):
    """
    Xác định kích thước ảnh đầu vào (width, height) từ cấu hình mô hình hoặc người dùng cung cấp.
    """
    if user_img_size is not None and len(user_img_size) == 2:
        return (user_img_size[0], user_img_size[1])

    # Thử lấy input_shape từ mô hình nếu có
    try:
        input_shape = model.input_shape
        # Input shape thường có dạng (None, H, W, C) hoặc [ (None, H, W, C) ]
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if input_shape and len(input_shape) >= 3:
            h, w = input_shape[1], input_shape[2]
            if h is not None and w is not None:
                return (int(h), int(w))
    except Exception:
        pass

    # Kích thước mặc định phổ biến
    default_size = (224, 224)
    print(f"[THÔNG TIN] Không thể xác định kích thước từ model, sử dụng kích thước mặc định: {default_size}")
    return default_size


def preprocess_image(image_path: str, target_size: tuple):
    """
    Đọc và tiền xử lý một ảnh:
    - Resize theo target_size
    - Chuyển thành numpy array
    - Chuẩn hóa giá trị pixel về khoảng [0, 1]
    """
    img = keras.utils.load_img(image_path, target_size=target_size)
    img_array = keras.utils.img_to_array(img)
    img_array = img_array / 255.0  # Chuẩn hóa về [0, 1]
    return img_array


def predict_soft_labels(
    model,
    image_paths: list,
    dataset_dir: str,
    target_size: tuple,
    batch_size: int = 32,
    key_mode: str = "auto"
):
    """
    3. Tiền xử lý theo lô (batch) và chạy dự đoán (predict) để lấy vector xác suất (soft labels).
    Trả về dictionary:
    {
        "tên_file_hoặc_đường_dẫn": [p1, p2, ..., pk],
        ...
    }
    """
    results = {}
    total_images = len(image_paths)
    dataset_path = Path(dataset_dir)

    # Xác định cách đặt key trong dictionary
    basenames = [os.path.basename(p) for p in image_paths]
    is_unique = (len(basenames) == len(set(basenames)))

    if key_mode == "auto":
        use_basename = is_unique
    elif key_mode == "filename":
        use_basename = True
    else:  # relpath
        use_basename = False

    print(f"[THÔNG TIN] Bắt đầu xử lý {total_images} ảnh với kích thước {target_size}, batch size: {batch_size}...")
    if use_basename:
        print("[THÔNG TIN] Định dạng key trong kết quả: Tên file ảnh (filename).")
    else:
        print("[THÔNG TIN] Định dạng key trong kết quả: Đường dẫn tương đối (relative path).")

    for start_idx in range(0, total_images, batch_size):
        end_idx = min(start_idx + batch_size, total_images)
        batch_paths = image_paths[start_idx:end_idx]

        # Chuẩn bị mảng ảnh cho batch hiện tại
        batch_images = []
        valid_keys = []

        for p in batch_paths:
            try:
                processed_img = preprocess_image(p, target_size)
                batch_images.append(processed_img)

                if use_basename:
                    key = os.path.basename(p)
                else:
                    key = os.path.relpath(p, dataset_path).replace("\\", "/")
                valid_keys.append(key)
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể xử lý ảnh '{p}': {e}")

        if not batch_images:
            continue

        batch_array = np.array(batch_images, dtype=np.float32)


        # Chạy dự đoán
        preds = model.predict(batch_array, verbose=0)

        # Nếu model trả về logits (chưa qua Softmax), tự động áp dụng Softmax
        # Kiểm tra xem tổng các phần tử có xấp xỉ 1 và các giá trị >= 0 không
        if preds.ndim == 2:
            row_sums = np.sum(preds, axis=1)
            is_probabilities = np.all(preds >= 0) and np.allclose(row_sums, 1.0, atol=1e-2)
            if not is_probabilities:
                preds = tf.nn.softmax(preds, axis=-1).numpy()

        # Lưu kết quả vector xác suất (soft labels) vào dictionary
        for key, pred_vector in zip(valid_keys, preds):
            # Chuyển đổi sang kiểu float thuần Python để có thể serialize sang JSON
            results[key] = [round(float(val), 6) for val in pred_vector.tolist()]

        print(f" -> Đã xử lý {end_idx}/{total_images} ảnh...")

    return results


def save_results_to_json(results: dict, output_path: str):
    """
    4. Lưu toàn bộ kết quả vào file JSON dưới dạng dictionary.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"[THÀNH CÔNG] Đã lưu kết quả ({len(results)} mẫu) vào: '{output_path}'")


def create_sample_teacher_model(save_path: str, num_classes: int = 5, input_shape: tuple = (224, 224, 3)):
    """
    Hàm tiện ích tạo một mô hình Keras mẫu (Teacher Model) dùng cho mục đích kiểm thử.
    """
    print(f"[THÔNG TIN] Đang tạo mô hình teacher mẫu với {num_classes} lớp tại '{save_path}'...")
    base_model = keras.applications.MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=input_shape
    )
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs, name="sample_teacher_model")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"[THÀNH CÔNG] Đã tạo và lưu mô hình mẫu tại '{save_path}'")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Đọc ảnh từ dataset/, chạy dự đoán qua mô hình Keras Teacher và lưu soft labels vào file JSON."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Đường dẫn đến thư mục chứa ảnh (mặc định: 'dataset')."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="teacher_model.h5",
        help="Đường dẫn đến file mô hình Keras .h5 hoặc .keras (mặc định: 'teacher_model.h5')."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="training-results.json",
        help="Đường dẫn file JSON đầu ra để lưu nhãn mềm (mặc định: 'training-results.json')."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Kích thước batch khi chạy dự đoán (mặc định: 32)."
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Kích thước ảnh đầu vào (vd: --img-size 224 224). Nếu bỏ trống sẽ tự suy luận từ mô hình."
    )
    parser.add_argument(
        "--create-sample-model",
        action="store_true",
        help="Tạo một mô hình mẫu teacher_model.h5 nếu file mô hình chưa tồn tại để kiểm thử."
    )
    parser.add_argument(
        "--key-mode",
        type=str,
        choices=["auto", "filename", "relpath"],
        default="auto",
        help="Định dạng key trong dictionary kết quả: 'filename' (tên file), 'relpath' (đường dẫn tương đối), hoặc 'auto' (tự động chọn)."
    )

    args = parser.parse_args()

    # 1. Đọc tất cả các tên file ảnh từ thư mục dataset/
    print("=" * 60)
    print(f"Bước 1: Quét thư mục ảnh: '{args.dataset_dir}'")
    image_paths = get_image_paths(args.dataset_dir)
    print(f"Tìm thấy tổng cộng: {len(image_paths)} file ảnh.")

    # 2. Load mô hình Keras mẫu
    print("=" * 60)
    print(f"Bước 2: Chuẩn bị mô hình Teacher: '{args.model_path}'")
    if not os.path.exists(args.model_path):
        if args.create_sample_model:
            print(f"[THÔNG TIN] Không tìm thấy '{args.model_path}', đang tự động khởi tạo mô hình mẫu...")
            create_sample_teacher_model(args.model_path)
        else:
            print(f"[LỖI] Không tìm thấy mô hình tại '{args.model_path}'.")
            print("Bạn có thể thêm tham số --create-sample-model để tự tạo mô hình mẫu kiểm thử:")
            print(f"  python script/generate_labels.py --create-sample-model")
            sys.exit(1)

    model = load_teacher_model(args.model_path)
    target_size = determine_target_size(model, args.img_size)

    # 3. Xử lý ảnh và chạy dự đoán lấy vector xác suất (soft labels)
    print("=" * 60)
    print(f"Bước 3: Dự đoán và trích xuất vector xác suất (soft labels)...")
    if not image_paths:
        print("[CẢNH BÁO] Không có ảnh nào để dự đoán. Kết quả trả về sẽ là dictionary rỗng.")
        results = {}
    else:
        results = predict_soft_labels(
            model=model,
            image_paths=image_paths,
            dataset_dir=args.dataset_dir,
            target_size=target_size,
            batch_size=args.batch_size,
            key_mode=args.key_mode
        )


    # 4. Lưu toàn bộ kết quả vào file training-results.json
    print("=" * 60)
    print(f"Bước 4: Lưu kết quả vào file JSON: '{args.output_path}'")
    save_results_to_json(results, args.output_path)
    print("=" * 60)
    print("[HOÀN TẤT] Quá trình trích xuất nhãn mềm (soft labels) đã kết thúc thành công!")


if __name__ == "__main__":
    main()
