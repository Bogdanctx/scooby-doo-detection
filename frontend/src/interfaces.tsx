export interface Patch {
    name: string;
    image: string; // base64 string
    detection_score: number;
    recognition_score: number;
}

export interface DetectionResult {
    detections: string; // base64 string of annotated image
    patches: Patch[];
}