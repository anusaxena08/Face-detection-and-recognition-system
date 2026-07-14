"""Face detection package.

Keep imports lazy so runtime consumers that only need detection do not
eagerly import DeepFace/TensorFlow through the training module.
"""

__all__ = [
	"FaceDetector",
	"draw_faces",
	"FaceRecognizer",
	"draw_recognition",
	"train",
]


def __getattr__(name: str):
	if name in {"FaceDetector", "draw_faces"}:
		from backend.face_detection.detector import FaceDetector, draw_faces

		exports = {
			"FaceDetector": FaceDetector,
			"draw_faces": draw_faces,
		}
		return exports[name]

	if name in {"FaceRecognizer", "draw_recognition"}:
		from backend.face_detection.recognizer import FaceRecognizer, draw_recognition

		exports = {
			"FaceRecognizer": FaceRecognizer,
			"draw_recognition": draw_recognition,
		}
		return exports[name]

	if name == "train":
		from backend.face_detection.trainer import train

		return train

	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
