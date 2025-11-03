import { API_URL } from './config';
import { OnnxDetectionService } from './OnnxDetectionService';

export interface AnolePrediction {
  species: string;
  scientificName: string;
  confidence: number;
  count: number;
  boundingBox?: number[]; // [x1, y1, x2, y2]
  detectionConfidence?: number;
}

export interface AnoleDetectionResult {
  totalLizards: number;
  predictions: AnolePrediction[];
  processingTime?: number;
}

export type DetectionMode = 'backend' | 'backend-pytorch' | 'onnx-frontend';

export class AnoleDetectionService {
  private static isOnnxInitialized = false;

  /**
   * Initialize ONNX models (only needed for frontend ONNX mode)
   */
  static async initializeOnnx(
    yoloModelUrl?: string,
    swinModelUrl?: string
  ): Promise<void> {
    if (!this.isOnnxInitialized) {
      await OnnxDetectionService.initialize(yoloModelUrl, swinModelUrl);
      this.isOnnxInitialized = true;
    }
  }

  /**
   * Detect and classify anoles using backend API
   * @param imageFile - Image to process
   * @param usePyTorch - If true, use PyTorch models instead of ONNX
   */
  static async detectBackend(imageFile: File, usePyTorch: boolean = false): Promise<AnoleDetectionResult> {
    try {
      const formData = new FormData();
      formData.append('file', imageFile);

      // Add use_onnx parameter (false for PyTorch, true for ONNX)
      const url = `${API_URL}/api/predict?use_onnx=${!usePyTorch}`;

      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Backend detection failed: ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Backend detection error:', error);
      throw new Error(`Failed to detect anoles via backend: ${error}`);
    }
  }

  /**
   * Detect and classify anoles using frontend ONNX models
   */
  static async detectFrontendOnnx(imageFile: File): Promise<AnoleDetectionResult> {
    console.log('🚨 detectFrontendOnnx called - CODE VERSION 2024-11-02-00:30');
    try {
      // Ensure ONNX is initialized
      if (!this.isOnnxInitialized) {
        console.log('⚠️ ONNX not initialized, initializing now...');
        await this.initializeOnnx();
      }

      // Load image
      console.log('📷 Loading image...');
      const img = await this.loadImage(imageFile);
      console.log(`📷 Image loaded: ${img.width}x${img.height}`);
      
      const startTime = performance.now();

      // Step 1: Detect anoles with YOLO
      // Note: Using high threshold (0.65) for single-class models to get top detections only
      // IOU threshold (0.45) for aggressive NMS to remove overlapping boxes
      console.log('📸 Starting YOLO detection...');
      let detections = await OnnxDetectionService.detectAnoles(img, 0.65, 0.45);
      
      // Limit to top 5 detections by score to avoid processing too many
      if (detections.length > 5) {
        console.log(`⚠️ Too many detections (${detections.length}), keeping top 5 by confidence`);
        detections = detections
          .sort((a, b) => b.score - a.score)
          .slice(0, 5);
      }
      
      console.log(`✅ YOLO returned ${detections.length} detections`);

      if (detections.length === 0) {
        console.warn('⚠️ No detections found by YOLO');
        return {
          totalLizards: 0,
          predictions: [],
          processingTime: (performance.now() - startTime) / 1000
        };
      }

      // Step 2: Classify each detected anole with Swin
      console.log(`🔬 Classifying ${detections.length} detected regions...`);
      const predictions: AnolePrediction[] = [];
      
      for (let i = 0; i < detections.length; i++) {
        const detection = detections[i];
        console.log(`🔬 Classifying detection ${i + 1}/${detections.length}...`);
        
        // Crop the detected region
        const [x, y, w, h] = detection.bbox;
        const croppedImg = await this.cropImage(img, x, y, w, h);
        
        // Classify the crop
        const classification = await OnnxDetectionService.classifyAnole(croppedImg);
        console.log(`✓ Detection ${i + 1}: ${classification.class} (${classification.score.toFixed(3)})`);

        predictions.push({
          species: classification.class,
          scientificName: this.getScientificName(classification.classId),
          confidence: classification.score,
          count: 1,
          boundingBox: [x, y, x + w, y + h],
          detectionConfidence: detection.score
        });
      }
      
      console.log(`✅ Classification complete! ${predictions.length} predictions`);

      const processingTime = (performance.now() - startTime) / 1000;

      return {
        totalLizards: predictions.length,
        predictions,
        processingTime
      };
    } catch (error) {
      console.error('Frontend ONNX detection error:', error);
      throw new Error(`Failed to detect anoles via frontend ONNX: ${error}`);
    }
  }

  /**
   * Detect anoles with configurable inference location
   * 
   * @param imageFile - Image file to process
   * @param mode - Where to run inference:
   *               - 'backend': Server-side PyTorch CPU (default, uses best.pt models)
   *               - 'backend-pytorch': Server-side PyTorch (same as backend)
   *               - 'onnx-frontend': Client-side ONNX (for privacy/offline)
   */
  static async detect(
    imageFile: File,
    mode: DetectionMode = 'backend'  // Default: backend PyTorch-CPU
  ): Promise<AnoleDetectionResult> {
    if (mode === 'backend') {
      return this.detectBackend(imageFile, true);   // PyTorch (default)
    } else if (mode === 'backend-pytorch') {
      return this.detectBackend(imageFile, true);   // PyTorch
    } else {
      try {
        return await this.detectFrontendOnnx(imageFile);
      } catch (error) {
        console.warn('Frontend ONNX failed, falling back to backend:', error);
        return this.detectBackend(imageFile, false);
      }
    }
  }

  /**
   * Load an image file and create an HTMLImageElement
   */
  private static async loadImage(file: File): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = URL.createObjectURL(file);
    });
  }

  /**
   * Crop an image to a specific region
   */
  private static async cropImage(
    img: HTMLImageElement,
    x: number,
    y: number,
    width: number,
    height: number
  ): Promise<HTMLCanvasElement> {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;
    
    // Ensure coordinates are within image bounds
    const clampedX = Math.max(0, Math.min(x, img.width));
    const clampedY = Math.max(0, Math.min(y, img.height));
    const clampedWidth = Math.min(width, img.width - clampedX);
    const clampedHeight = Math.min(height, img.height - clampedY);
    
    ctx.drawImage(
      img,
      clampedX,
      clampedY,
      clampedWidth,
      clampedHeight,
      0,
      0,
      width,
      height
    );
    
    return canvas;
  }

  /**
   * Get scientific name for a class ID
   */
  private static getScientificName(classId: number): string {
    const scientificNames: Record<number, string> = {
      0: 'Anolis distichus',
      1: 'Anolis sagrei',
      2: 'Anolis cristatellus',
      3: 'Anolis carolinensis',
      4: 'Anolis equestris',
    };
    return scientificNames[classId] || 'Unknown';
  }

  /**
   * Get species color for visualization
   */
  static getSpeciesColor(speciesName: string): string {
    const colors: Record<string, string> = {
      'Bark Anole': '#8B4513',
      'Brown Anole': '#A0522D',
      'Crested Anole': '#CD853F',
      'Green Anole': '#228B22',
      'Knight Anole': '#006400',
    };
    return colors[speciesName] || '#808080';
  }

  /**
   * Check if ONNX is available and initialized
   */
  static isOnnxReady(): boolean {
    return this.isOnnxInitialized && OnnxDetectionService.isReady();
  }

  /**
   * Get available species names
   */
  static getSpeciesNames(): string[] {
    return [
      'Bark Anole',
      'Brown Anole',
      'Crested Anole',
      'Green Anole',
      'Knight Anole'
    ];
  }
}

