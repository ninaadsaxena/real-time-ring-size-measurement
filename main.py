import cv2
import mediapipe as mp
import numpy as np
import math
import statistics

class PreciseRingSizeDetector:
    def __init__(self):
        # MediaPipe Hand Tracking Setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        # Ring size conversion charts
        self.RING_SIZE_CONVERSIONS = {
            'US': [
                {'diameter': 14.0, 'size': 3},
                {'diameter': 14.4, 'size': 4},
                {'diameter': 14.8, 'size': 5},
                {'diameter': 15.2, 'size': 6},
                {'diameter': 15.6, 'size': 7},
                {'diameter': 16.0, 'size': 8},
                {'diameter': 16.5, 'size': 9},
                {'diameter': 16.9, 'size': 10},
                {'diameter': 17.3, 'size': 11},
                {'diameter': 17.7, 'size': 12},
            ],
            'UK/Australia': [
                {'diameter': 14.1, 'size': 'F'},
                {'diameter': 14.5, 'size': 'H'},
                {'diameter': 14.9, 'size': 'J'},
                {'diameter': 15.3, 'size': 'L'},
                {'diameter': 15.7, 'size': 'N'},
                {'diameter': 16.1, 'size': 'P'},
                {'diameter': 16.6, 'size': 'R'},
                {'diameter': 17.0, 'size': 'T'},
                {'diameter': 17.4, 'size': 'V'},
                {'diameter': 17.8, 'size': 'X'},
            ],
            'Europe': [
                {'diameter': 14.0, 'size': 44},
                {'diameter': 14.4, 'size': 46},
                {'diameter': 14.8, 'size': 48},
                {'diameter': 15.2, 'size': 50},
                {'diameter': 15.6, 'size': 52},
                {'diameter': 16.0, 'size': 54},
                {'diameter': 16.5, 'size': 56},
                {'diameter': 16.9, 'size': 58},
                {'diameter': 17.3, 'size': 60},
                {'diameter': 17.7, 'size': 62},
            ]
        }

        # Measurement tracking
        self.diameter_measurements = []
        self.angle_measurements = {}
        self.tracking_stages = [
            'Front View', 
            'Side View (Left)', 
            'Side View (Right)', 
            'Slight Angle View'
        ]
        self.current_stage_index = 0

    def calculate_distance_3d(self, point1, point2):
        """Calculate 3D Euclidean distance between two points."""
        return math.sqrt(
            (point1.x - point2.x)**2 + 
            (point1.y - point2.y)**2 + 
            (point1.z - point2.z)**2
        )

    def calculate_finger_width(self, hand_landmarks):
        """
        Calculate ring finger width using multiple landmark points.
        More robust than single-point measurement.
        """
        # Ring finger landmarks
        ring_finger_mcp = hand_landmarks.landmark[17]  # Metacarpophalangeal joint
        ring_finger_pip = hand_landmarks.landmark[18]  # Proximal interphalangeal joint
        ring_finger_dip = hand_landmarks.landmark[19]  # Distal interphalangeal joint
        ring_finger_tip = hand_landmarks.landmark[20]  # Fingertip

        # Calculate width at different points
        width_mcp_pip = self.calculate_distance_3d(
            hand_landmarks.landmark[5],  # Palm side point
            hand_landmarks.landmark[17]  # Ring finger MCP
        )
        width_pip_dip = self.calculate_distance_3d(
            hand_landmarks.landmark[18],  # Ring finger PIP
            hand_landmarks.landmark[19]   # Ring finger DIP
        )

        # Average the measurements for more accuracy
        avg_width = (width_mcp_pip + width_pip_dip) / 2
        return avg_width

    def find_ring_size(self, diameter_mm):
        """Find closest ring size across different regions."""
        ring_sizes = {}
        for region, sizes in self.RING_SIZE_CONVERSIONS.items():
            closest_size = min(sizes, key=lambda x: abs(x['diameter'] - diameter_mm))
            ring_sizes[region] = closest_size['size']
        return ring_sizes

    def detect_ring_finger(self, frame):
        """
        Detect ring finger with advanced measurement techniques.
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Calculate ring finger width
                finger_width = self.calculate_finger_width(hand_landmarks)
                
                # Store measurement
                self.diameter_measurements.append(finger_width)
                
                return frame, finger_width
        
        return frame, None

    def run_precise_detection(self):
        """
        Advanced multi-angle ring finger measurement process.
        """
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Current tracking stage
            current_stage = self.tracking_stages[self.current_stage_index]
            
            # Detect ring finger
            processed_frame, finger_width = self.detect_ring_finger(frame)
            
            # Guidance instructions
            cv2.putText(processed_frame, 
                        f"Current View: {current_stage}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
            
            cv2.putText(processed_frame, 
                        "Position ring finger clearly in frame", 
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 1)
            
            # If measurement detected
            if finger_width:
                # Store measurement for current stage
                self.angle_measurements[current_stage] = finger_width
                
                # Progress to next stage
                if len(self.angle_measurements) == len(self.tracking_stages):
                    # Calculate final size
                    final_diameter = statistics.median(list(self.angle_measurements.values()))
                    ring_sizes = self.find_ring_size(final_diameter)
                    
                    # Display final results
                    cv2.putText(processed_frame, 
                                f"Estimated Ring Size: {ring_sizes}", 
                                (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 0, 255), 2)
                    
                    # Optional: Reset for new measurement
                    self.angle_measurements.clear()
                    self.current_stage_index = 0
                else:
                    # Move to next stage
                    self.current_stage_index += 1
            
            cv2.imshow('Precise Ring Size Detector', processed_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = PreciseRingSizeDetector()
    detector.run_precise_detection()

if __name__ == "__main__":
    main()
