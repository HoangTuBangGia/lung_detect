"""
Main Window UI - Giao diện chính của ứng dụng Lung Cancer Detection.

Thiết kế Clean & Modern với PySide6.
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCursor, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    """
    Cửa sổ chính của ứng dụng Lung Cancer Detection.
    
    Layout:
    ┌──────────────┬────────────────────────────────┐
    │   SIDEBAR    │           CONTENT              │
    │              │                                │
    │  [Logo]      │     ┌──────────────────┐       │
    │              │     │                  │       │
    │  [Chọn ảnh]  │     │   Preview Area   │       │
    │              │     │                  │       │
    │              │     └──────────────────┘       │
    │              │                                │
    │              │     ┌──────────────────┐       │
    │              │     │  Result Area     │       │
    │  [Thoát]     │     └──────────────────┘       │
    └──────────────┴────────────────────────────────┘
    """
    
    # Constants
    SIDEBAR_WIDTH = 280
    WINDOW_MIN_WIDTH = 1000
    WINDOW_MIN_HEIGHT = 700
    PREVIEW_MIN_SIZE = 400
    
    # Supported image formats
    IMAGE_FILTERS = "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;All Files (*)"
    
    # Vietnamese translations for diagnosis labels
    DIAGNOSIS_TRANSLATIONS = {
        "Lung Adenocarcinoma": "Ung thư biểu mô tuyến phổi",
        "Lung Normal": "Phổi bình thường",
        "Lung Squamous Cell Carcinoma": "Ung thư biểu mô vảy phổi",
    }
    
    def __init__(self, model_path: str = "lung_cancer_model_ver2.keras", parent=None):
        """Khởi tạo MainWindow.
        
        Args:
            model_path: Đường dẫn đến file model AI (.keras)
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Instance variables
        self.model_path = Path(model_path)
        self.current_image_path: Optional[Path] = None
        self._ai_worker: Optional["AIWorker"] = None  # type: ignore
        
        self._setup_window()
        self._setup_ui()
        self._apply_styles()
        self._setup_connections()
    
    def _setup_window(self) -> None:
        """Cấu hình cửa sổ chính."""
        self.setWindowTitle("Lung Cancer Detection - AI Diagnostic Tool")
        self.setMinimumSize(self.WINDOW_MIN_WIDTH, self.WINDOW_MIN_HEIGHT)
        self.setObjectName("MainWindow")
    
    def _setup_ui(self) -> None:
        """Thiết lập toàn bộ UI."""
        # Central widget
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create sidebar and content
        sidebar = self._create_sidebar()
        content = self._create_content()
        
        # Add to main layout
        main_layout.addWidget(sidebar)
        main_layout.addWidget(content, 1)  # stretch factor = 1
    
    # =========================================================================
    # SIDEBAR
    # =========================================================================
    
    def _create_sidebar(self) -> QFrame:
        """
        Tạo sidebar chứa logo và các nút điều khiển.
        
        Returns:
            QFrame: Sidebar widget
        """
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(self.SIDEBAR_WIDTH)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 30, 20, 30)
        layout.setSpacing(15)
        
        # --- Logo / Title Section ---
        logo_section = self._create_logo_section()
        layout.addWidget(logo_section)
        
        # --- Separator ---
        separator = QFrame()
        separator.setObjectName("Separator")
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFixedHeight(2)
        layout.addWidget(separator)
        
        # --- Spacer ---
        layout.addSpacing(20)
        
        # --- Action Buttons ---
        self.btn_select_image = QPushButton("🖼️  Chọn ảnh")
        self.btn_select_image.setObjectName("BtnSelectImage")
        self.btn_select_image.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_select_image.setMinimumHeight(50)
        layout.addWidget(self.btn_select_image)
        
        # --- Info Label ---
        info_label = QLabel("Hỗ trợ: JPG, PNG, BMP\nKích thước khuyến nghị: 224x224")
        info_label.setObjectName("InfoLabel")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # --- Flexible Spacer (đẩy nút Thoát xuống dưới) ---
        layout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )
        
        # --- Status Indicator ---
        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # --- Exit Button ---
        self.btn_exit = QPushButton("Thoát")
        self.btn_exit.setObjectName("BtnExit")
        self.btn_exit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_exit.setMinimumHeight(40)
        layout.addWidget(self.btn_exit)
        
        return sidebar
    
    def _create_logo_section(self) -> QWidget:
        """
        Tạo section logo và title.
        
        Returns:
            QWidget: Logo section widget
        """
        container = QWidget()
        container.setObjectName("LogoSection")
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # App Icon/Logo
        logo_label = QLabel("🫁")
        logo_label.setObjectName("LogoIcon")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setFont(QFont("Segoe UI Emoji", 48))
        layout.addWidget(logo_label)
        
        # App Title
        title_label = QLabel("Lung Cancer\nDetection")
        title_label.setObjectName("AppTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("AI-Powered Diagnostic Tool")
        subtitle_label.setObjectName("AppSubtitle")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle_label)
        
        return container
    
    # =========================================================================
    # CONTENT AREA
    # =========================================================================
    
    def _create_content(self) -> QFrame:
        """
        Tạo content area chứa preview ảnh và kết quả.
        
        Returns:
            QFrame: Content widget
        """
        content = QFrame()
        content.setObjectName("ContentArea")
        
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # --- Preview Area ---
        preview_section = self._create_preview_section()
        layout.addWidget(preview_section, 3)  # stretch factor = 3
        
        # --- Result Area ---
        result_section = self._create_result_section()
        layout.addWidget(result_section, 1)  # stretch factor = 1
        
        return content
    
    def _create_preview_section(self) -> QFrame:
        """
        Tạo section hiển thị preview ảnh.
        
        Returns:
            QFrame: Preview section widget
        """
        container = QFrame()
        container.setObjectName("PreviewSection")
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # Section Title
        title = QLabel("📷 Preview Ảnh")
        title.setObjectName("SectionTitle")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Image Preview Label
        self.preview_label = QLabel()
        self.preview_label.setObjectName("PreviewLabel")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(self.PREVIEW_MIN_SIZE, self.PREVIEW_MIN_SIZE)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.preview_label.setScaledContents(False)
        
        # Placeholder text khi chưa có ảnh
        self._set_preview_placeholder()
        
        layout.addWidget(self.preview_label, 1)
        
        # Image info label
        self.image_info_label = QLabel("Chưa chọn ảnh")
        self.image_info_label.setObjectName("ImageInfoLabel")
        self.image_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_info_label)
        
        return container
    
    def _create_result_section(self) -> QFrame:
        """
        Tạo section hiển thị kết quả dự đoán.
        
        Returns:
            QFrame: Result section widget
        """
        container = QFrame()
        container.setObjectName("ResultSection")
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Section Title
        title = QLabel("🔬 Kết quả phân tích")
        title.setObjectName("SectionTitle")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Result Container (horizontal layout)
        result_container = QWidget()
        result_layout = QHBoxLayout(result_container)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(30)
        
        # --- Left: Diagnosis Result ---
        diagnosis_widget = self._create_diagnosis_widget()
        result_layout.addWidget(diagnosis_widget, 1)
        
        # --- Right: Confidence ---
        confidence_widget = self._create_confidence_widget()
        result_layout.addWidget(confidence_widget, 1)
        
        layout.addWidget(result_container)
        
        return container
    
    def _create_diagnosis_widget(self) -> QWidget:
        """
        Tạo widget hiển thị kết quả chẩn đoán.
        
        Returns:
            QWidget: Diagnosis widget
        """
        widget = QWidget()
        widget.setObjectName("DiagnosisWidget")
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Label "Chẩn đoán:"
        label = QLabel("Chẩn đoán:")
        label.setObjectName("ResultLabel")
        layout.addWidget(label)
        
        # Kết quả chẩn đoán
        self.diagnosis_result = QLabel("Chưa có kết quả")
        self.diagnosis_result.setObjectName("DiagnosisResult")
        self.diagnosis_result.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.diagnosis_result.setWordWrap(True)
        layout.addWidget(self.diagnosis_result)
        
        # Bản dịch tiếng Việt
        self.diagnosis_vietnamese = QLabel("")
        self.diagnosis_vietnamese.setObjectName("DiagnosisVietnamese")
        self.diagnosis_vietnamese.setFont(QFont("Segoe UI", 14))
        self.diagnosis_vietnamese.setWordWrap(True)
        self.diagnosis_vietnamese.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(self.diagnosis_vietnamese)
        
        return widget
    
    def _create_confidence_widget(self) -> QWidget:
        """
        Tạo widget hiển thị độ tin cậy với progress bar.
        
        Returns:
            QWidget: Confidence widget
        """
        widget = QWidget()
        widget.setObjectName("ConfidenceWidget")
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Label "Độ tin cậy:"
        label = QLabel("Độ tin cậy:")
        label.setObjectName("ResultLabel")
        layout.addWidget(label)
        
        # Confidence percentage text
        self.confidence_text = QLabel("-- %")
        self.confidence_text.setObjectName("ConfidenceText")
        self.confidence_text.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(self.confidence_text)
        
        # Progress Bar
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setObjectName("ConfidenceBar")
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(False)
        self.confidence_bar.setMinimumHeight(20)
        layout.addWidget(self.confidence_bar)
        
        return widget
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _set_preview_placeholder(self) -> None:
        """Đặt placeholder cho preview label khi chưa có ảnh."""
        self.preview_label.setText(
            "Kéo thả ảnh vào đây\nhoặc\nNhấn 'Chọn ảnh' để bắt đầu"
        )
        self.preview_label.setProperty("hasImage", False)
        # Trigger style refresh
        self.preview_label.style().unpolish(self.preview_label)
        self.preview_label.style().polish(self.preview_label)
    
    def set_preview_image(self, pixmap: QPixmap) -> None:
        """
        Hiển thị ảnh preview.
        
        Args:
            pixmap: QPixmap của ảnh cần hiển thị
        """
        if pixmap.isNull():
            self._set_preview_placeholder()
            return
        
        # Scale pixmap to fit label while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)
        self.preview_label.setProperty("hasImage", True)
        # Trigger style refresh
        self.preview_label.style().unpolish(self.preview_label)
        self.preview_label.style().polish(self.preview_label)
    
    def set_result(self, diagnosis: str, confidence: float) -> None:
        """
        Hiển thị kết quả dự đoán.
        
        Args:
            diagnosis: Tên bệnh/kết quả
            confidence: Độ tin cậy (0-100)
        """
        self.diagnosis_result.setText(diagnosis)
        
        # Hiển thị bản dịch tiếng Việt
        vietnamese = self.DIAGNOSIS_TRANSLATIONS.get(diagnosis, "")
        self.diagnosis_vietnamese.setText(vietnamese)
        
        self.confidence_text.setText(f"{confidence:.1f} %")
        self.confidence_bar.setValue(int(confidence))
        
        # Set confidence level property cho QSS styling
        if confidence >= 70:
            confidence_level = "high"
        elif confidence >= 40:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        self.confidence_bar.setProperty("confidence", confidence_level)
        # Trigger style refresh
        self.confidence_bar.style().unpolish(self.confidence_bar)
        self.confidence_bar.style().polish(self.confidence_bar)
    
    def reset_result(self) -> None:
        """Reset kết quả về trạng thái ban đầu."""
        self.diagnosis_result.setText("Chưa có kết quả")
        self.diagnosis_vietnamese.setText("")
        self.confidence_text.setText("-- %")
        self.confidence_bar.setValue(0)
    
    def set_status(self, message: str) -> None:
        """
        Cập nhật status label.
        
        Args:
            message: Thông báo trạng thái
        """
        self.status_label.setText(message)
    
    # =========================================================================
    # CONNECTIONS & EVENT HANDLERS
    # =========================================================================
    
    def _setup_connections(self) -> None:
        """Kết nối signals với slots."""
        self.btn_select_image.clicked.connect(self._on_select_image_clicked)
        self.btn_exit.clicked.connect(self._on_exit_clicked)
    
    @Slot()
    def _on_select_image_clicked(self) -> None:
        """Xử lý sự kiện click nút 'Chọn ảnh'."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh phổi để phân tích",
            "",
            self.IMAGE_FILTERS
        )
        
        if file_path:
            self._load_and_analyze_image(file_path)
    
    @Slot()
    def _on_exit_clicked(self) -> None:
        """Xử lý sự kiện click nút 'Thoát'."""
        self.close()
    
    def _load_and_analyze_image(self, image_path: str) -> None:
        """
        Load ảnh lên UI và bắt đầu phân tích AI.
        
        Args:
            image_path: Đường dẫn ảnh
        """
        self.current_image_path = Path(image_path)
        
        # 1. Hiển thị ảnh preview ngay lập tức
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(
                self,
                "Lỗi",
                f"Không thể đọc file ảnh:\n{image_path}"
            )
            return
        
        self.set_preview_image(pixmap)
        self.image_info_label.setText(
            f"{self.current_image_path.name} | "
            f"{pixmap.width()}x{pixmap.height()} px"
        )
        
        # 2. Reset kết quả cũ
        self.reset_result()
        
        # 3. Khởi chạy AIWorker
        self._start_ai_worker(image_path)
    
    def _start_ai_worker(self, image_path: str) -> None:
        """
        Khởi chạy AIWorker để phân tích ảnh.
        
        Args:
            image_path: Đường dẫn ảnh cần phân tích
        """
        # Import AIWorker ở đây để tránh circular import
        from core.ai_worker import AIWorker
        
        # Dừng worker cũ nếu đang chạy
        self._stop_ai_worker()
        
        # Tạo worker mới
        self._ai_worker = AIWorker(
            image_path=image_path,
            model_path=str(self.model_path),
            parent=self
        )
        
        # Connect signals
        self._ai_worker.prediction_ready.connect(self._on_prediction_ready)
        self._ai_worker.error_occurred.connect(self._on_prediction_error)
        self._ai_worker.progress_updated.connect(self._on_progress_updated)
        self._ai_worker.finished.connect(self._on_worker_finished)
        
        # Set UI state: Loading
        self._set_loading_state(True)
        
        # Start worker
        self._ai_worker.start()
    
    def _stop_ai_worker(self) -> None:
        """Dừng AIWorker nếu đang chạy."""
        if self._ai_worker is not None:
            if self._ai_worker.isRunning():
                self._ai_worker.cancel()
                self._ai_worker.wait(3000)  # Đợi tối đa 3 giây
            self._ai_worker.deleteLater()
            self._ai_worker = None
    
    def _set_loading_state(self, is_loading: bool) -> None:
        """
        Đặt trạng thái UI khi đang/không đang xử lý.
        
        Args:
            is_loading: True nếu đang xử lý AI
        """
        # Disable/Enable nút chọn ảnh
        self.btn_select_image.setEnabled(not is_loading)
        
        # Đổi cursor
        if is_loading:
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            self.btn_select_image.setText("⏳  Đang xử lý...")
        else:
            QApplication.restoreOverrideCursor()
            self.btn_select_image.setText("🖼️  Chọn ảnh")
    
    @Slot(str, float)
    def _on_prediction_ready(self, label: str, confidence: float) -> None:
        """
        Xử lý khi nhận được kết quả dự đoán.
        
        Args:
            label: Nhãn kết quả (tên bệnh)
            confidence: Độ tin cậy (0-100%)
        """
        self.set_result(label, confidence)
        self.set_status("✅ Hoàn thành")
        
        # Đặt property để style theo kết quả
        if "Normal" not in label:
            # Phát hiện bất thường - màu đỏ
            self.diagnosis_result.setProperty("result", "abnormal")
        else:
            # Bình thường - màu xanh
            self.diagnosis_result.setProperty("result", "normal")
        
        # Trigger style refresh
        self.diagnosis_result.style().unpolish(self.diagnosis_result)
        self.diagnosis_result.style().polish(self.diagnosis_result)
    
    @Slot(str)
    def _on_prediction_error(self, error_message: str) -> None:
        """
        Xử lý khi có lỗi từ AIWorker.
        
        Args:
            error_message: Thông báo lỗi
        """
        self.set_status("❌ Lỗi")
        self.diagnosis_result.setText("Lỗi xử lý")
        self.diagnosis_vietnamese.setText("")
        self.diagnosis_result.setProperty("result", "error")
        self.diagnosis_result.style().unpolish(self.diagnosis_result)
        self.diagnosis_result.style().polish(self.diagnosis_result)
        
        QMessageBox.critical(
            self,
            "Lỗi phân tích",
            f"Không thể phân tích ảnh:\n\n{error_message}"
        )
    
    @Slot(str)
    def _on_progress_updated(self, message: str) -> None:
        """
        Cập nhật tiến trình xử lý.
        
        Args:
            message: Thông báo tiến trình
        """
        self.set_status(f"⏳ {message}")
    
    @Slot()
    def _on_worker_finished(self) -> None:
        """Xử lý khi AIWorker hoàn thành (dù thành công hay thất bại)."""
        self._set_loading_state(False)
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    def closeEvent(self, event) -> None:
        """
        Xử lý sự kiện đóng cửa sổ.
        
        Đảm bảo dừng worker thread trước khi đóng.
        """
        self._stop_ai_worker()
        
        # Clear model cache để giải phóng bộ nhớ
        try:
            from core.ai_worker import clear_model_cache
            clear_model_cache()
        except ImportError:
            pass
        
        event.accept()
    
    # =========================================================================
    # STYLES
    # =========================================================================
    
    def _apply_styles(self) -> None:
        """Áp dụng QSS styles cho toàn bộ UI."""
        self.setStyleSheet(self._get_stylesheet())
    
    def _get_stylesheet(self) -> str:
        """
        Tạo QSS stylesheet.
        
        Returns:
            str: QSS stylesheet string
        """
        return """
            /* ===== GLOBAL ===== */
            QMainWindow#MainWindow {
                background-color: #f5f7fa;
            }
            
            /* ===== SIDEBAR ===== */
            QFrame#Sidebar {
                background-color: #1e293b;
                border: none;
            }
            
            QFrame#Sidebar QLabel {
                color: #e2e8f0;
            }
            
            QLabel#AppTitle {
                color: #ffffff;
                font-size: 18px;
            }
            
            QLabel#AppSubtitle {
                color: #94a3b8;
                font-size: 11px;
            }
            
            QLabel#InfoLabel {
                color: #64748b;
                font-size: 10px;
                padding: 10px;
            }
            
            QLabel#StatusLabel {
                color: #22c55e;
                font-size: 12px;
                padding: 8px;
                background-color: #1e3a2f;
                border-radius: 6px;
            }
            
            QFrame#Separator {
                background-color: #334155;
            }
            
            /* ===== BUTTONS ===== */
            QPushButton#BtnSelectImage {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 12px 20px;
            }
            
            QPushButton#BtnSelectImage:hover {
                background-color: #2563eb;
            }
            
            QPushButton#BtnSelectImage:pressed {
                background-color: #1d4ed8;
            }
            
            QPushButton#BtnExit {
                background-color: transparent;
                color: #94a3b8;
                border: 1px solid #475569;
                border-radius: 6px;
                font-size: 13px;
                padding: 8px 16px;
            }
            
            QPushButton#BtnExit:hover {
                background-color: #334155;
                color: #e2e8f0;
            }
            
            /* ===== CONTENT AREA ===== */
            QFrame#ContentArea {
                background-color: #f5f7fa;
                border: none;
            }
            
            QFrame#PreviewSection, QFrame#ResultSection {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }
            
            QLabel#SectionTitle {
                color: #334155;
                padding-bottom: 5px;
            }
            
            /* ===== PREVIEW LABEL ===== */
            QLabel#PreviewLabel {
                background-color: #f8fafc;
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                color: #94a3b8;
                font-size: 14px;
            }
            
            QLabel#PreviewLabel[hasImage="true"] {
                border: 2px solid #3b82f6;
                background-color: #ffffff;
            }
            
            QLabel#ImageInfoLabel {
                color: #64748b;
                font-size: 11px;
            }
            
            /* ===== RESULT SECTION ===== */
            QLabel#ResultLabel {
                color: #64748b;
                font-size: 12px;
            }
            
            QLabel#DiagnosisResult {
                color: #1e293b;
            }
            
            QLabel#ConfidenceText {
                color: #3b82f6;
            }
            
            /* ===== PROGRESS BAR ===== */
            QProgressBar#ConfidenceBar {
                background-color: #e2e8f0;
                border: none;
                border-radius: 10px;
            }
            
            QProgressBar#ConfidenceBar::chunk {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #3b82f6,
                    stop: 1 #8b5cf6
                );
                border-radius: 10px;
            }
        """


# Chạy test UI nếu chạy trực tiếp file này
if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
