"""
AI Worker Module - Xử lý inference model trên luồng riêng biệt.

Module này sử dụng QThread để chạy model AI prediction
mà không block main UI thread.
"""

# QUAN TRỌNG: Set environment variables TRƯỚC KHI import bất kỳ thứ gì từ TensorFlow
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal

# Lazy import tensorflow để tránh slow startup
_model_cache: dict = {}


class AIWorker(QThread):
    """
    Worker thread để chạy AI inference.
    
    Signals:
        prediction_ready: Emit khi có kết quả (label: str, confidence: float)
        error_occurred: Emit khi có lỗi (error_message: str)
        progress_updated: Emit cập nhật tiến trình (message: str)
    
    Example:
        worker = AIWorker(image_path="/path/to/image.jpg", model_path="/path/to/model.keras")
        worker.prediction_ready.connect(on_result)
        worker.error_occurred.connect(on_error)
        worker.start()
    """
    
    # Signals
    prediction_ready = Signal(str, float)  # (label, confidence_percentage)
    error_occurred = Signal(str)           # error_message
    progress_updated = Signal(str)         # status_message
    
    # Class labels cho lung cancer classification
    CLASS_LABELS = [
        "Lung Adenocarcinoma",      # Ung thư biểu mô tuyến phổi
        "Lung Normal",               # Phổi bình thường
        "Lung Squamous Cell Carcinoma"  # Ung thư biểu mô vảy phổi
    ]
    
    # Image preprocessing constants
    TARGET_SIZE = (224, 224)
    
    def __init__(
        self,
        image_path: str,
        model_path: str = "lung_cancer_model_ver2.keras",
        parent=None
    ):
        """
        Khởi tạo AIWorker.
        
        Args:
            image_path: Đường dẫn đến file ảnh cần phân tích
            model_path: Đường dẫn đến file model (.keras)
            parent: Parent QObject (optional)
        """
        super().__init__(parent)
        self.image_path = Path(image_path)
        self.model_path = Path(model_path)
        self._is_cancelled = False
    
    def cancel(self) -> None:
        """Hủy worker nếu đang chạy."""
        self._is_cancelled = True
    
    def run(self) -> None:
        """
        Entry point của thread - thực hiện inference.
        
        Flow:
        1. Validate input
        2. Load và preprocess ảnh
        3. Load model (có cache)
        4. Predict
        5. Emit kết quả
        """
        try:
            # Step 1: Validate inputs
            self.progress_updated.emit("Đang kiểm tra file...")
            self._validate_inputs()
            
            if self._is_cancelled:
                return
            
            # Step 2: Load và preprocess image
            self.progress_updated.emit("Đang xử lý ảnh...")
            image_array = self._load_and_preprocess_image()
            
            if self._is_cancelled:
                return
            
            # Step 3: Load model
            self.progress_updated.emit("Đang tải model AI...")
            model = self._load_model()
            
            if self._is_cancelled:
                return
            
            # Step 4: Predict
            self.progress_updated.emit("Đang phân tích...")
            label, confidence = self._predict(model, image_array)
            
            if self._is_cancelled:
                return
            
            # Step 5: Emit result
            self.progress_updated.emit("Hoàn thành!")
            self.prediction_ready.emit(label, confidence)
            
        except FileNotFoundError as e:
            self.error_occurred.emit(f"Không tìm thấy file: {e}")
        except ValueError as e:
            self.error_occurred.emit(f"Lỗi dữ liệu: {e}")
        except MemoryError:
            self.error_occurred.emit("Không đủ bộ nhớ để xử lý. Vui lòng đóng bớt ứng dụng khác.")
        except Exception as e:
            self.error_occurred.emit(f"Lỗi không xác định: {type(e).__name__}: {e}")
    
    def _validate_inputs(self) -> None:
        """
        Validate input files.
        
        Raises:
            FileNotFoundError: Nếu file không tồn tại
            ValueError: Nếu định dạng file không hợp lệ
        """
        # Check image file
        if not self.image_path.exists():
            raise FileNotFoundError(f"Ảnh '{self.image_path}' không tồn tại")
        
        # Check image extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        if self.image_path.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Định dạng ảnh không được hỗ trợ: {self.image_path.suffix}. "
                f"Các định dạng hỗ trợ: {', '.join(valid_extensions)}"
            )
        
        # Check model file
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model '{self.model_path}' không tồn tại")
        
        if self.model_path.suffix != '.keras':
            raise ValueError(f"File model phải có định dạng .keras, nhận được: {self.model_path.suffix}")
    
    def _load_and_preprocess_image(self) -> np.ndarray:
        """
        Load và preprocess ảnh cho model.
        
        Returns:
            np.ndarray: Ảnh đã được preprocess với shape (1, 224, 224, 3)
        
        Raises:
            ValueError: Nếu không thể đọc được ảnh
        """
        try:
            # Load image using PIL
            with Image.open(self.image_path) as img:
                # Convert grayscale to RGB if needed
                if img.mode == 'L':
                    # Grayscale image -> convert to RGB
                    img = img.convert('RGB')
                elif img.mode == 'RGBA':
                    # RGBA -> RGB (remove alpha channel)
                    img = img.convert('RGB')
                elif img.mode != 'RGB':
                    # Other modes -> convert to RGB
                    img = img.convert('RGB')
                
                # Resize to target size
                img_resized = img.resize(self.TARGET_SIZE, Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                image_array = np.array(img_resized, dtype=np.float32)
        
        except Exception as e:
            raise ValueError(f"Không thể đọc file ảnh: {e}")
        
        # Validate shape
        if image_array.shape != (224, 224, 3):
            raise ValueError(
                f"Shape ảnh không đúng sau khi xử lý. "
                f"Expected: (224, 224, 3), Got: {image_array.shape}"
            )
        
        # Expand dimensions for batch: (224, 224, 3) -> (1, 224, 224, 3)
        # Không chia 255 vì model đã có lớp Rescaling
        image_batch = np.expand_dims(image_array, axis=0)
        
        return image_batch
    
    def _load_model(self):
        """
        Load Keras model với caching.
        
        Returns:
            Loaded Keras model
        
        Note:
            Model được cache để tránh load lại nhiều lần.
            Sử dụng Keras 3 thuần (không dùng tf_keras).
        """
        global _model_cache
        
        model_key = str(self.model_path.absolute())
        
        if model_key not in _model_cache:
            # Suppress warnings
            import logging
            logging.getLogger('tensorflow').setLevel(logging.ERROR)
            
            # Load model với Keras 3
            import keras
            _model_cache[model_key] = keras.models.load_model(
                str(self.model_path),
                compile=False,
                safe_mode=False
            )
        
        return _model_cache[model_key]
    
    def _predict(self, model, image_array: np.ndarray) -> tuple[str, float]:
        """
        Thực hiện prediction.
        
        Args:
            model: Loaded Keras model
            image_array: Preprocessed image array
        
        Returns:
            tuple: (label: str, confidence_percentage: float)
        """
        # Run inference
        predictions = model.predict(image_array, verbose=0)
        
        # Get predicted class and confidence
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx]) * 100  # Convert to percentage
        
        # Get label
        if 0 <= predicted_idx < len(self.CLASS_LABELS):
            label = self.CLASS_LABELS[predicted_idx]
        else:
            label = f"Unknown Class ({predicted_idx})"
        
        return label, confidence


def clear_model_cache() -> None:
    """
    Xóa cache model để giải phóng bộ nhớ.
    
    Gọi function này khi cần giải phóng RAM,
    ví dụ khi đóng ứng dụng.
    """
    global _model_cache
    _model_cache.clear()


# Convenience function để chạy prediction đồng bộ (cho testing)
def predict_sync(
    image_path: str,
    model_path: str = "lung_cancer_model_ver2.keras"
) -> tuple[str, float]:
    """
    Chạy prediction đồng bộ (blocking).
    
    Chỉ nên dùng cho testing hoặc CLI tools.
    Với GUI, hãy dùng AIWorker class.
    
    Args:
        image_path: Đường dẫn ảnh
        model_path: Đường dẫn model
    
    Returns:
        tuple: (label, confidence_percentage)
    """
    worker = AIWorker(image_path, model_path)
    worker._validate_inputs()
    image_array = worker._load_and_preprocess_image()
    model = worker._load_model()
    return worker._predict(model, image_array)
