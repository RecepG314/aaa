import rclpy
from rclpy.node import Node
import numpy as np
import joblib
import cv2

class AIModelNode(Node):
    def __init__(self):
        super().__init__('ai_model_node')
        self.get_logger().info("AI Model Node başlatıldı.")
        
        # Model ve veri yükleme
        self.get_logger().info("Model ve veri yükleniyor...")
        self.svm_model = joblib.load("/path/to/svm_model_resnet50.joblib")
        self.data = np.load("/path/to/features_data.npz")
        self.test_features = self.data["test_features"]
        self.test_labels = self.data["test_labels"]
        self.get_logger().info("Model ve veri yüklendi.")

        # Örnek sınıflandırma
        self.classify_sample(0)

    def classify_sample(self, index):
        test_feature = self.test_features[index]
        predicted_label = self.svm_model.predict(test_feature.reshape(1, -1))
        self.get_logger().info(f"Tahmin edilen sınıf: {predicted_label[0]}")
        self.get_logger().info(f"Gerçek sınıf: {self.test_labels[index]}")

def main(args=None):
    rclpy.init(args=args)
    node = AIModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
