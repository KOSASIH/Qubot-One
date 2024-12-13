# src/hardware/sensors/camera.py

import cv2
import numpy as np

class Camera:
    def __init__(self, camera_id, resolution=(640, 480), fps=30):
        """Initialize the camera.

        Args:
            camera_id (int): Identifier for the camera (e.g., camera index).
            resolution (tuple): Resolution of the camera (width, height).
            fps (int): Frames per second for video capture.
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.capture = None
        self.is_opened = False

    def open(self):
        """Open the camera for capturing."""
        self.capture = cv2.VideoCapture(self.camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        self.is_opened = self.capture.isOpened()
        if not self.is_opened:
            raise Exception(f"Camera {self.camera_id} could not be opened.")

    def close(self):
        """Release the camera resources."""
        if self.capture is not None:
            self.capture.release()
            self.is_opened = False
            print(f"Camera {self.camera_id} closed.")

    def capture_image(self):
        """Capture a single image from the camera.

        Returns:
            np.array: Captured image in BGR format.
        """
        if not self.is_opened:
            raise Exception("Camera is not opened.")
        
        ret, frame = self.capture.read()
        if not ret:
            raise Exception("Failed to capture image.")
        
        return frame

    def start_video_stream(self):
        """Start video streaming from the camera."""
        if not self.is_opened:
            raise Exception("Camera is not opened.")
        
        print(f"Starting video stream from camera {self.camera_id}...")
        while self.is_opened:
            frame = self.capture_image()
            cv2.imshow(f'Camera {self.camera_id}', frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.close()
        cv2.destroyAllWindows()

    def apply_filter(self, frame, filter_type='gray'):
        """Apply a filter to the captured frame.

        Args:
            frame (np.array): The image frame to process.
            filter_type (str): Type of filter to apply ('gray', 'blur', 'edge').

        Returns:
            np.array: Processed image frame.
        """
        if filter_type == 'gray':
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif filter_type == 'blur':
            return cv2.GaussianBlur(frame, (5, 5), 0)
        elif filter_type == 'edge':
            return cv2.Canny(frame, 100, 200)
        else:
            raise ValueError("Unsupported filter type. Use 'gray', 'blur', or 'edge'.")

    def save_image(self, frame, file_path):
        """Save the captured image to a file.

        Args:
            frame (np.array): The image frame to save.
            file_path (str): Path to save the image.
        """
        cv2.imwrite(file_path, frame)
        print(f"Image saved to {file_path}")

    def get_camera_properties(self):
        """Get the current properties of the camera.

        Returns:
            dict: Dictionary containing camera properties.
        """
        properties = {
            'Camera ID': self.camera_id,
            'Resolution': self.resolution,
            'FPS': self.fps,
            'Opened': self.is_opened
        }
        return properties

# Example usage
if __name__ == "__main__":
    camera = Camera(camera_id=0, resolution=(640, 480), fps=30)
    try:
        camera.open()
        camera.start_video_stream()
    except Exception as e:
        print(e)
    finally:
        camera.close()
