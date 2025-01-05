#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import joblib
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class AIModelNode(Node):
    def __init__(self):
        super().__init__('ai_model_node')

        self.get_logger().info("AI Model Node başlatıldı.")
        
        # ResNet modelini yükle
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_model.eval()  # Modeli değerlendirme moduna al
        self.get_logger().info("ResNet modeli yüklendi.")

        # SVM modelini yükle
        self.svm_model = joblib.load("/path/to/svm_model_resnet50.joblib")
        self.get_logger().info("SVM modeli yüklendi.")

        # Görüntü işleme için dönüşümler
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # ResNet için giriş boyutu
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # CvBridge ile ROS görüntülerini işleyin
        self.bridge = CvBridge()

        # Publisher (Model sonucu yayınlama)
        self.result_publisher = self.create_publisher(String, '/ai_model/result', 10)

        # Subscriber (Fotoğraf alımı)
        self.image_subscriber = self.create_subscription(
            Image,
            '/simple_drone/bottom/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        """Gelen görüntüyü sınıflandır"""
        self.get_logger().info('Görüntü alındı, işleniyor...')

        try:
            # ROS Image mesajını OpenCV görüntüsüne dönüştür
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

            # Görüntüden öznitelik çıkar
            feature_vector = self.extract_features(cv_image)

            # SVM modelini kullanarak sınıflandır
            prediction = self.svm_model.predict(feature_vector.reshape(1, -1))
            predicted_label = prediction[0]

            # Sonucu yayınla
            result_msg = String()
            result_msg.data = f"Tahmin edilen sınıf: {predicted_label}"
            self.result_publisher.publish(result_msg)
            self.get_logger().info(f"Tahmin sonucu yayınlandı: {predicted_label}")

        except Exception as e:
            self.get_logger().error(f"Görüntü işleme hatası: {e}")

    def extract_features(self, image):
        """ResNet'i kullanarak görüntüden öznitelik çıkarır"""
        # Görüntüyü işleme tabi tut
        processed_image = self.transform(image).unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

        # ResNet'ten öznitelik çıkar
        with torch.no_grad():
            features = self.resnet_model(processed_image)

        # Öznitelikleri numpy array'e dönüştür
        return features.numpy().squeeze()

def main(args=None):
    rclpy.init(args=args)

    ai_model_node = AIModelNode()

    try:
        rclpy.spin(ai_model_node)
    except KeyboardInterrupt:
        pass
    finally:
        ai_model_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()