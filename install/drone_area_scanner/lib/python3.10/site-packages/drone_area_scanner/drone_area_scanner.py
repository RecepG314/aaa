#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

class DroneAreaScanner(Node):
    def __init__(self):
        super().__init__('drone_area_scanner')
        
        # Waypoint ve fotoğraf publisher'ları
        self.waypoint_publisher = self.create_publisher(PoseStamped, '/drone/goal_pose', 10)
        self.photo_trigger_publisher = self.create_publisher(Image, '/drone/camera/image_raw', 10)
        
        # Parametre tanımları
        self.declare_parameter('scan_step', 5.0)  # Her adımda 5 metre
        self.declare_parameter('flight_altitude', 10.0)  # 10 metre yükseklik
        
        self.scan_step = self.get_parameter('scan_step').value
        self.flight_altitude = self.get_parameter('flight_altitude').value
        
    def generate_coverage_path(self, coordinates):
        """
        Verilen koordinatları kullanarak alan tarama rotası oluştur
        
        :param coordinates: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] formatında koordinatlar
        :return: Tarama için waypoint listesi
        """
        # Koordinatları sınırlayan dikdörtgeni hesapla
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Serpantin tarama algoritması
        waypoints = []
        x = x_min
        direction = 1  # 1: yukarı, -1: aşağı
        
        while x <= x_max:
            # Y ekseninde yukarı aşağı hareket et
            if direction == 1:
                waypoints.extend([
                    [x, y_min, self.flight_altitude] 
                    for y in np.arange(y_min, y_max + self.scan_step, self.scan_step)
                ])
            else:
                waypoints.extend([
                    [x, y_max, self.flight_altitude] 
                    for y in np.arange(y_max, y_min - self.scan_step, -self.scan_step)
                ])
            
            # Bir sonraki x pozisyonuna geç
            x += self.scan_step
            direction *= -1
        
        return waypoints
    
    def create_pose_stamped(self, x, y, z):
        """Waypoint için PoseStamped mesajı oluştur"""
        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        
        # Varsayılan oryantasyon (gerekirse değiştirilebilir)
        pose.pose.orientation.w = 1.0
        
        return pose
    
    def take_photo(self):
        """Fotoğraf çekme fonksiyonu"""
        photo_msg = Image()
        photo_msg.header = Header()
        photo_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Gerçek uygulamada kamera parametreleri eklenecek
        # Şimdilik boş bir görüntü mesajı
        self.photo_trigger_publisher.publish(photo_msg)
        self.get_logger().info('Fotoğraf çekildi!')
    
    def start_scanning(self, coordinates):
        """Alan taramasını başlat"""
        waypoints = self.generate_coverage_path(coordinates)
        
        for waypoint in waypoints:
            # Her waypoint için pose mesajı oluştur
            pose = self.create_pose_stamped(*waypoint)
            
            # Waypoint'e git
            self.waypoint_publisher.publish(pose)
            self.get_logger().info(f'Hedefe gidiliyor: {waypoint}')
            
            # Biraz bekle (gerçek zamanlı sistemde synchronization gerekli)
            rclpy.spin_once(self, timeout_sec=5)
            
            # Fotoğraf çek
            self.take_photo()
    
def main(args=None):
    rclpy.init(args=args)
    
    drone_scanner = DroneAreaScanner()
    
    # Örnek koordinatlar (gerçek koordinatlarınızla değiştirebilirsiniz)
    example_coords = [
        [0, 0],   # Sol alt köşe
        [0, 100], # Sol üst köşe
        [100, 100], # Sağ üst köşe
        [100, 0]   # Sağ alt köşe
    ]
    
    try:
        drone_scanner.start_scanning(example_coords)
    except Exception as e:
        drone_scanner.get_logger().error(f'Hata oluştu: {str(e)}')
    finally:
        drone_scanner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()