# src/hardware/sensors/lidar.py

import numpy as np
import open3d as o3d

class Lidar:
    def __init__(self, lidar_id, frame_rate=10):
        """Initialize the LiDAR sensor.

        Args:
            lidar_id (str): Identifier for the LiDAR sensor.
            frame_rate (int): Frame rate for data capture.
        """
        self.lidar_id = lidar_id
        self.frame_rate = frame_rate
        self.point_cloud_data = None

    def start_capture(self):
        """Start capturing point cloud data from the LiDAR."""
        print(f"Starting capture from LiDAR {self.lidar_id} at {self.frame_rate} FPS...")
        # Placeholder for actual capture logic
        # In a real implementation, this would interface with the LiDAR hardware
        self.point_cloud_data = self._simulate_point_cloud()

    def stop_capture(self):
        """Stop capturing point cloud data."""
        print(f"Stopping capture from LiDAR {self.lidar_id}.")

    def get_point_cloud(self):
        """Get the latest point cloud data.

        Returns:
            np.array: Point cloud data (Nx3 array of XYZ coordinates).
        """
        if self.point_cloud_data is None:
            raise Exception("No point cloud data available. Please start capture.")
        return self.point_cloud_data

    def _simulate_point_cloud(self):
        """Simulate point cloud data for demonstration purposes.

        Returns:
            np.array: Simulated point cloud data.
        """
        # Simulate a random point cloud
        num_points = 1000
        return np.random.rand(num_points, 3) * 10  # Random points in a 10x10x10 cube

    def visualize_point_cloud(self):
        """Visualize the current point cloud data using Open3D."""
        if self.point_cloud_data is None:
            raise Exception("No point cloud data available. Please start capture.")

        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud_data)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd], window_name=f'LiDAR {self.lidar_id} Point Cloud')

    def get_lidar_properties(self):
        """Get the current properties of the LiDAR sensor.

        Returns:
            dict: Dictionary containing LiDAR properties.
        """
        properties = {
            'LiDAR ID': self.lidar_id,
            'Frame Rate': self.frame_rate,
            'Point Cloud Data Available': self.point_cloud_data is not None
        }
        return properties

# Example usage
if __name__ == "__main__":
    lidar = Lidar(lidar_id="LIDAR_1", frame_rate=10)
    try:
        lidar.start_capture()
        point_cloud = lidar.get_point_cloud()
        print("Point cloud data retrieved:", point_cloud)
        lidar.visualize_point_cloud()
    except Exception as e:
        print(e)
    finally:
        lidar.stop_capture()
