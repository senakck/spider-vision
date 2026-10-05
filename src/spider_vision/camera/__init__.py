"""Camera layer: opens the webcam and delivers frames. Knows nothing about vision."""

from spider_vision.camera.camera import Camera, CameraError, Frame

__all__ = ["Camera", "CameraError", "Frame"]
