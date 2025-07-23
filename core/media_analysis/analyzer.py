"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-MEDIA-ANALYZER-0001                 │
// │ 📁 domain       : Media Analysis, Image Processing          │
// │ 🧠 description  : Advanced media analysis engine for        │
// │                  processing diverse media data sources      │
// │ 🕸️ hash_type    : UUID → CUID-linked processor              │
// │ 🔄 parent_node  : NODE_PROCESSOR                           │
// │ 🧩 dependencies : opencv, PIL, numpy, requests             │
// │ 🔧 tool_usage   : Content Analysis, Intelligence Extraction │
// │ 📡 input_type   : Image, Video, Audio                      │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : media processing, feature extraction      │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

CTAS Media Analysis Engine
--------------------------
Core module for advanced media analysis capabilities including image,
video, and audio processing for intelligence extraction and analysis.
"""

import os
import io
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import requests
from PIL import Image
import cv2
import hashlib
from dataclasses import dataclass

# Function creates subject logger
# Method initializes predicate output
# Operation configures object format
logger = logging.getLogger("ctas_media_analyzer")
logger.setLevel(logging.INFO)

# Constants for media analysis
# Function defines subject constants
# Method sets predicate values
# Operation configures object parameters
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]


@dataclass
class MediaMetadata:
    """
    Data class for media file metadata

    # Class stores subject metadata
    # Structure maintains predicate information
    # Container holds object properties
    """

    file_hash: str
    file_type: str
    creation_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    size_bytes: int = 0
    dimensions: Optional[Tuple[int, int]] = None
    duration: Optional[float] = None
    location_data: Optional[Dict[str, float]] = None
    device_info: Optional[Dict[str, str]] = None
    extraction_date: datetime = datetime.now()


class MediaAnalyzer:
    """
    Comprehensive media analysis engine for extracting intelligence from
    various media file types including images, videos, and audio.

    # Class analyzes subject media
    # Engine processes predicate files
    # Component extracts object intelligence
    """

    def __init__(self, cache_dir: str = "cache/media_analysis"):
        """
        Initialize the media analyzer with cache directory

        # Function initializes subject analyzer
        # Method configures predicate settings
        # Operation sets object parameters

        Args:
            cache_dir: Directory to cache analysis results
        """
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        logger.info(
            f"MediaAnalyzer initialized with cache directory: {cache_dir}"
        )

    def get_media_type(self, file_path: str) -> str:
        """
        Determine the type of media from the file extension

        # Function determines subject type
        # Method identifies predicate category
        # Operation classifies object format

        Args:
            file_path: Path to the media file

        Returns:
            Media type ('image', 'video', 'audio', or 'unknown')
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext in SUPPORTED_IMAGE_FORMATS:
            return "image"
        elif file_ext in SUPPORTED_VIDEO_FORMATS:
            return "video"
        elif file_ext in SUPPORTED_AUDIO_FORMATS:
            return "audio"
        else:
            return "unknown"

    def compute_file_hash(self, file_path: str) -> str:
        """
        Compute SHA-256 hash of a file

        # Function computes subject hash
        # Method calculates predicate fingerprint
        # Operation generates object identifier

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash of the file
        """
        hash_sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    def extract_metadata(self, file_path: str) -> MediaMetadata:
        """
        Extract metadata from a media file

        # Function extracts subject metadata
        # Method retrieves predicate attributes
        # Operation collects object properties

        Args:
            file_path: Path to the media file

        Returns:
            MediaMetadata object containing extracted metadata
        """
        # Get basic file information
        file_type = self.get_media_type(file_path)
        file_hash = self.compute_file_hash(file_path)
        file_stats = os.stat(file_path)

        # Initialize metadata object
        metadata = MediaMetadata(
            file_hash=file_hash,
            file_type=file_type,
            size_bytes=file_stats.st_size,
            modified_date=datetime.fromtimestamp(file_stats.st_mtime),
        )

        # Extract media-specific metadata
        if file_type == "image":
            try:
                with Image.open(file_path) as img:
                    metadata.dimensions = img.size

                    # Extract EXIF data if available
                    exif_data = {}
                    if hasattr(img, "_getexif") and img._getexif():
                        exif = img._getexif()
                        if exif:
                            for tag_id, value in exif.items():
                                exif_data[tag_id] = value

                            # Extract GPS information if available
                            if 34853 in exif_data:  # GPSInfo tag
                                gps_info = exif_data[34853]
                                lat, lon = self._parse_gps_info(gps_info)
                                if lat and lon:
                                    metadata.location_data = {
                                        "latitude": lat,
                                        "longitude": lon,
                                    }

                            # Extract device information if available
                            device_info = {}
                            if 271 in exif_data:  # Make tag
                                device_info["make"] = exif_data[271]
                            if 272 in exif_data:  # Model tag
                                device_info["model"] = exif_data[272]
                            if 305 in exif_data:  # Software tag
                                device_info["software"] = exif_data[305]

                            if device_info:
                                metadata.device_info = device_info

            except Exception as e:
                logger.warning(f"Error extracting image metadata: {e}")

        elif file_type == "video":
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    # Get video properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    metadata.dimensions = (width, height)

                    # Calculate duration
                    if fps > 0 and frame_count > 0:
                        metadata.duration = frame_count / fps

                cap.release()

            except Exception as e:
                logger.warning(f"Error extracting video metadata: {e}")

        return metadata

    def _parse_gps_info(
        self, gps_info: Dict
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Parse GPS information from EXIF data

        # Function parses subject GPS
        # Method extracts predicate coordinates
        # Operation converts object format

        Args:
            gps_info: GPS information from EXIF data

        Returns:
            Tuple of (latitude, longitude) or (None, None) if parsing fails
        """
        try:
            lat_ref = gps_info.get(1, "N")
            lat = gps_info.get(2)
            lon_ref = gps_info.get(3, "E")
            lon = gps_info.get(4)

            if lat and lon:
                # Convert to decimal degrees
                lat_decimal = self._convert_to_decimal_degrees(lat)
                lon_decimal = self._convert_to_decimal_degrees(lon)

                # Apply hemisphere reference
                if lat_ref == "S":
                    lat_decimal = -lat_decimal
                if lon_ref == "W":
                    lon_decimal = -lon_decimal

                return lat_decimal, lon_decimal

        except Exception as e:
            logger.warning(f"Error parsing GPS info: {e}")

        return None, None

    def _convert_to_decimal_degrees(self, dms: Tuple) -> float:
        """
        Convert degrees-minutes-seconds format to decimal degrees

        # Function converts subject format
        # Method transforms predicate coordinates
        # Operation calculates object degrees

        Args:
            dms: Degrees-minutes-seconds tuple

        Returns:
            Decimal degrees value
        """
        degrees = dms[0][0] / dms[0][1]
        minutes = dms[1][0] / dms[1][1] / 60
        seconds = dms[2][0] / dms[2][1] / 3600

        return degrees + minutes + seconds

    def analyze_image(
        self,
        image_path: str,
        detect_objects: bool = True,
        detect_faces: bool = True,
        ocr_text: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze an image for objects, faces, and text

        # Function analyzes subject image
        # Method processes predicate content
        # Operation detects object features

        Args:
            image_path: Path to the image file
            detect_objects: Whether to perform object detection
            detect_faces: Whether to perform face detection
            ocr_text: Whether to perform OCR on text in the image

        Returns:
            Dictionary containing analysis results
        """
        # Extract metadata first
        metadata = self.extract_metadata(image_path)

        # Initialize results dictionary
        results = {
            "metadata": metadata,
            "objects": [],
            "faces": [],
            "text": "",
            "dominant_colors": [],
        }

        try:
            # Load image for analysis
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Find dominant colors
            results["dominant_colors"] = self._extract_dominant_colors(image)

            # Object detection
            if detect_objects:
                results["objects"] = self._detect_objects(image)

            # Face detection
            if detect_faces:
                results["faces"] = self._detect_faces(image)

            # OCR text recognition
            if ocr_text:
                results["text"] = self._extract_text(image)

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")

        return results

    def _extract_dominant_colors(
        self, image: np.ndarray, num_colors: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Extract dominant colors from an image

        # Function extracts subject colors
        # Method finds predicate palette
        # Operation identifies object hues

        Args:
            image: Image array
            num_colors: Number of dominant colors to extract

        Returns:
            List of dominant colors with hex values and percentages
        """
        # Reshape image for processing
        pixels = image.reshape(-1, 3)

        # Convert from BGR to RGB
        pixels = pixels[:, ::-1]

        # Sample pixels for faster processing
        max_pixels = 10000
        if len(pixels) > max_pixels:
            indices = np.random.choice(len(pixels), max_pixels, replace=False)
            pixels = pixels[indices]

        # Perform k-means clustering
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)

        # Get the colors and their proportions
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(labels)

        # Convert to hex and create results
        results = []
        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            r, g, b = color
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            results.append(
                {
                    "hex": hex_color,
                    "percentage": float(percentage),
                    "rgb": (int(r), int(g), int(b)),
                }
            )

        # Sort by percentage
        results.sort(key=lambda x: x["percentage"], reverse=True)

        return results

    def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using pre-trained models

        # Function detects subject objects
        # Method identifies predicate items
        # Operation locates object entities

        Args:
            image: Image array

        Returns:
            List of detected objects with class, confidence, and bounding box
        """
        # Placeholder for object detection implementation
        # This would typically use a model like YOLO, SSD, or Faster R-CNN
        # For now, return empty list to indicate no implementation
        logger.info("Object detection called but not fully implemented")
        return []

    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image

        # Function detects subject faces
        # Method identifies predicate individuals
        # Operation locates object persons

        Args:
            image: Image array

        Returns:
            List of detected faces with bounding box and confidence
        """
        face_results = []

        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Use OpenCV's Haar Cascade classifier for face detection
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for x, y, w, h in faces:
                face_results.append(
                    {
                        "confidence": 0.85,  # Hardcoded confidence for Haar Cascade
                        "bounding_box": (x, y, w, h),
                    }
                )

        except Exception as e:
            logger.error(f"Error in face detection: {e}")

        return face_results

    def _extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from an image using OCR

        # Function extracts subject text
        # Method identifies predicate words
        # Operation recognizes object content

        Args:
            image: Image array

        Returns:
            Extracted text as string
        """
        # Placeholder for OCR implementation
        # This would typically use a library like pytesseract
        logger.info("OCR text extraction called but not fully implemented")
        return ""

    def analyze_video(
        self, video_path: str, sample_rate: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze a video file, sampling frames at the specified rate

        # Function analyzes subject video
        # Method processes predicate frames
        # Operation extracts object intelligence

        Args:
            video_path: Path to the video file
            sample_rate: Number of frames to skip between samples

        Returns:
            Dictionary containing analysis results
        """
        # Extract metadata first
        metadata = self.extract_metadata(video_path)

        # Initialize results
        results = {
            "metadata": metadata,
            "keyframes": [],
            "detected_scenes": [],
            "dominant_colors": [],
            "motion_areas": [],
        }

        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {video_path}")

            frame_count = 0
            keyframe_count = 0

            # Process video frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process only every sample_rate frames
                if frame_count % sample_rate == 0:
                    # Analyze keyframe
                    keyframe_results = self.analyze_image(
                        frame,
                        detect_objects=True,
                        detect_faces=True,
                        ocr_text=False,
                    )

                    # Store keyframe timestamp
                    timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)
                    keyframe_results["timestamp"] = timestamp

                    # Add to results
                    results["keyframes"].append(keyframe_results)
                    keyframe_count += 1

                frame_count += 1

            cap.release()

            logger.info(
                f"Analyzed {frame_count} frames, extracted {keyframe_count} keyframes"
            )

        except Exception as e:
            logger.error(f"Error analyzing video: {e}")

        return results

    def analyze_url_media(
        self, url: str, media_type: str = None
    ) -> Dict[str, Any]:
        """
        Analyze media from a URL by downloading it first

        # Function analyzes subject URL
        # Method processes predicate remote media
        # Operation downloads object content

        Args:
            url: URL to the media file
            media_type: Type of media if known, otherwise determined from URL

        Returns:
            Analysis results for the media
        """
        try:
            # Create a temporary filename based on URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()
            temp_dir = os.path.join(self.cache_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            # Determine file extension from URL if not specified
            if media_type is None:
                url_path = url.split("?")[0]  # Remove query parameters
                ext = os.path.splitext(url_path)[1].lower()

                if ext in SUPPORTED_IMAGE_FORMATS:
                    media_type = "image"
                elif ext in SUPPORTED_VIDEO_FORMATS:
                    media_type = "video"
                elif ext in SUPPORTED_AUDIO_FORMATS:
                    media_type = "audio"
                else:
                    # Default to image if can't determine
                    media_type = "image"
                    ext = ".jpg"
            else:
                # Assign default extension based on media type
                if media_type == "image":
                    ext = ".jpg"
                elif media_type == "video":
                    ext = ".mp4"
                elif media_type == "audio":
                    ext = ".mp3"
                else:
                    ext = ""

            temp_file = os.path.join(temp_dir, f"{url_hash}{ext}")

            # Download the file
            logger.info(f"Downloading media from URL: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Analyze the downloaded file based on media type
            if media_type == "image":
                results = self.analyze_image(temp_file)
            elif media_type == "video":
                results = self.analyze_video(temp_file)
            else:
                # Not implemented for other types yet
                results = {"metadata": self.extract_metadata(temp_file)}

            # Add URL to results
            results["source_url"] = url

            # Clean up the temporary file
            os.remove(temp_file)

            return results

        except Exception as e:
            logger.error(f"Error analyzing media from URL: {e}")
            return {"error": str(e), "source_url": url}
