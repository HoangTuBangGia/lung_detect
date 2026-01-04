"""
Lung Cancer Detection Application - Entry Point

Ứng dụng phát hiện ung thư phổi sử dụng AI.
Sử dụng PySide6 cho GUI và TensorFlow cho AI inference.
"""

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

# Add src to path for imports
SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

# Constants
APP_NAME = "Lung Cancer Detection"
APP_VERSION = "0.1.0"
ORG_NAME = "LungCancerAI"
MODEL_FILENAME = "lung_cancer_model_ver2.keras"
STYLES_FILENAME = "assets/styles.qss"


def find_model_path() -> Path:
    """
    Tìm đường dẫn đến file model.
    
    Tìm kiếm theo thứ tự:
    1. Thư mục gốc dự án (ROOT_DIR)
    2. Thư mục hiện tại (cwd)
    3. Thư mục chứa script
    
    Returns:
        Path: Đường dẫn đến file model
    
    Raises:
        FileNotFoundError: Nếu không tìm thấy model
    """
    search_paths = [
        ROOT_DIR / MODEL_FILENAME,
        Path.cwd() / MODEL_FILENAME,
        SRC_DIR / MODEL_FILENAME,
    ]
    
    for path in search_paths:
        if path.exists():
            return path.resolve()
    
    # Không tìm thấy
    searched = "\n  - ".join(str(p) for p in search_paths)
    raise FileNotFoundError(
        f"Không tìm thấy file model '{MODEL_FILENAME}'.\n"
        f"Đã tìm kiếm tại:\n  - {searched}"
    )


def setup_application() -> QApplication:
    """
    Khởi tạo và cấu hình QApplication.
    
    Returns:
        QApplication: Instance của application
    """
    # Enable high DPI scaling (PHẢI gọi TRƯỚC khi tạo QApplication)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Tạo application
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(ORG_NAME)
    
    # Load stylesheet
    load_stylesheet(app)
    
    return app


def load_stylesheet(app: QApplication) -> None:
    """
    Load và áp dụng QSS stylesheet.
    
    Args:
        app: QApplication instance
    """
    # Tìm file stylesheet
    style_paths = [
        ROOT_DIR / STYLES_FILENAME,
        Path.cwd() / STYLES_FILENAME,
    ]
    
    for style_path in style_paths:
        if style_path.exists():
            try:
                with open(style_path, "r", encoding="utf-8") as f:
                    stylesheet = f.read()
                app.setStyleSheet(stylesheet)
                print(f"🎨 Đã load stylesheet: {style_path}")
                return
            except Exception as e:
                print(f"⚠️ Không thể load stylesheet: {e}")
                return
    
    print(f"ℹ️ Không tìm thấy file stylesheet, sử dụng style mặc định.")


def show_error_and_exit(title: str, message: str) -> int:
    """
    Hiển thị dialog lỗi và thoát ứng dụng.
    
    Args:
        title: Tiêu đề dialog
        message: Nội dung lỗi
    
    Returns:
        int: Exit code (1 = error)
    """
    # Cần QApplication để hiển thị dialog
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    QMessageBox.critical(None, title, message)
    return 1


def main() -> int:
    """
    Main entry point của ứng dụng.
    
    Returns:
        int: Exit code (0 = success, non-zero = error)
    """
    # 1. Khởi tạo QApplication
    app = setup_application()
    
    # 2. Tìm và validate model path
    try:
        model_path = find_model_path()
        print(f"✅ Đã tìm thấy model: {model_path}")
    except FileNotFoundError as e:
        return show_error_and_exit("Lỗi khởi động", str(e))
    
    # 3. Import và khởi tạo MainWindow
    try:
        from ui.main_window import MainWindow
        
        window = MainWindow(model_path=str(model_path))
        window.show()
        
        print(f"🚀 {APP_NAME} v{APP_VERSION} đã khởi động!")
        print(f"   Model: {model_path.name}")
        
    except ImportError as e:
        return show_error_and_exit(
            "Lỗi Import",
            f"Không thể import module cần thiết:\n{e}\n\n"
            "Hãy đảm bảo đã cài đặt đầy đủ dependencies:\n"
            "  uv sync"
        )
    except Exception as e:
        return show_error_and_exit(
            "Lỗi khởi động",
            f"Không thể khởi động ứng dụng:\n{type(e).__name__}: {e}"
        )
    
    # 4. Chạy event loop
    exit_code = app.exec()
    
    # 5. Cleanup
    print("👋 Ứng dụng đã đóng.")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
