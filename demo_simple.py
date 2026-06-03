"""
Simple Fall Detection Demo - No MediaPipe Pose
Using only OpenCV + YOLOv8 for detection
"""
import cv2
import time
from datetime import datetime
from ultralytics import YOLO


class SimpleFallDetectionDemo:
    """
    Demo don gian - chi su dung YOLO + aspect ratio
    """

    def __init__(self):
        print("=" * 70)
        print("  SIMPLE FALL DETECTION DEMO")
        print("  Using: YOLOv8 Person Detection + Aspect Ratio")
        print("=" * 70)
        print()

        # Load YOLOv8
        print("[1] Loading YOLOv8...")
        self.yolo = YOLO('yolov8n.pt')
        print("    Done!")

        # State tracking
        self.fall_start_time = None
        self.is_falling = False
        self.person_positions = {}  # Track person positions

        print()
        print("Ready! Press 'q' to quit, 'r' to reset")
        print("-" * 70)

    def detect_fall(self, bbox_width, bbox_height):
        """
        Detect fall based on bounding box aspect ratio
        - Normal: height > width (vertical person)
        - Fall: width > height (horizontal person)
        """
        if bbox_height > 0:
            aspect_ratio = bbox_width / bbox_height
            # Fall if aspect ratio > 1.3 and person is on ground (low in frame)
            is_fall = aspect_ratio > 1.3
            return is_fall, aspect_ratio
        return False, 1.0

    def run(self):
        """Run demo"""
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect persons with YOLO
            results = self.yolo(frame, conf=0.5, classes=[0], verbose=False)

            fall_detected = False
            aspect_ratio = 1.0
            person_count = 0

            # Process each detected person
            for result in results:
                for box in result.boxes:
                    person_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()

                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)),
                                 (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}",
                               (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Detect fall
                    is_fall, aspect_ratio = self.detect_fall(bbox_width, bbox_height)

                    current_time = time.time()

                    if is_fall:
                        if not self.is_falling:
                            self.is_falling = True
                            self.fall_start_time = current_time

                        time_falling = current_time - self.fall_start_time

                        # Show fall detection box
                        cv2.rectangle(frame, (int(x1), int(y1)),
                                     (int(x2), int(y2)), (0, 0, 255), 3)

                        if time_falling > 0.5:
                            # CONFIRMED FALL
                            cv2.rectangle(frame, (5, 5),
                                       (frame.shape[1]-5, frame.shape[0]-5),
                                       (0, 0, 255), 5)
                            cv2.putText(frame,
                                       "!!! CANH BAO: NGA !!!",
                                       (frame.shape[1]//2 - 150, 50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                       (0, 0, 255), 3)
                            cv2.putText(frame,
                                       f"Time: {time_falling:.1f}s",
                                       (int(x1), int(y2) + 25),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                       (0, 0, 255), 2)

                            # Alert beep
                            try:
                                import winsound
                                winsound.Beep(1000, 200)
                            except:
                                pass

                        fall_detected = True
                    else:
                        self.is_falling = False
                        self.fall_start_time = None

                    # Show aspect ratio
                    cv2.putText(frame,
                               f"Ratio: {aspect_ratio:.2f}",
                               (int(x1), int(y2) + 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (255, 0, 0), 1)

            # Draw info panel
            info_y_start = 30
            cv2.putText(frame, f"Persons: {person_count}",
                       (10, info_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {'FALL DETECTED' if fall_detected else 'NORMAL'}",
                       (10, info_y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 0, 255) if fall_detected else (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}",
                       (10, info_y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2)

            cv2.imshow("Simple Fall Detection - YOLOv8", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('r'):
                self.is_falling = False
                self.fall_start_time = None
                print("Reset!")

        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo ended!")


if __name__ == "__main__":
    demo = SimpleFallDetectionDemo()
    demo.run()
