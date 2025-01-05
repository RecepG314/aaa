from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class AIModelNode(Node):
    def __init__(self):
        super().__init__('ai_model_node')

        self.get_logger().info("AI Model Node başlatıldı.")
        
        # Model ve veri yükleme
        self.svm_model = joblib.load("/path/to/svm_model_resnet50.joblib")

        # Subscriber (fotoğraf alımı)
        self.image_subscriber = self.create_subscription(
            Image,
            '/simple_drone/bottom/image_raw',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()

    def image_callback(self, msg):
        """Gelen görüntüyü sınıflandır"""
        self.get_logger().info('Görüntü alındı, işleniyor...')
        
        # Görüntüyü OpenCV formatına çevir
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            self.get_logger().error(f"Görüntü dönüştürme hatası: {e}")
            return

        # Görüntüden öznitelik çıkarma (örnek için basit bir kullanım)
        feature_vector = self.extract_features(cv_image)

        # Sınıflandırma
        prediction = self.svm_model.predict(feature_vector.reshape(1, -1))
        self.get_logger().info(f"Tahmin edilen sınıf: {prediction[0]}")

    def extract_features(self, image):
        """Görüntüden basit öznitelik çıkarımı"""
        # Örneğin, bir ResNet modeli burada kullanılabilir
        return np.mean(image, axis=(0, 1))  # RGB ortalamaları (örnek basit öznitelik)
