import { API_URL } from './config';
import { OnnxDetectionService } from './OnnxDetectionService';

export interface AlternateConfidence {
  classIndex: number;
  species: string;
  scientificName: string;
  confidence: number;
  relativeConfidence: number;
}

export interface AnolePrediction {
  species: string;
  scientificName: string;
  confidence: number;
  count: number;
  boundingBox?: number[]; // [x1, y1, x2, y2]
  detectionConfidence?: number;
  alternateConfidences?: AlternateConfidence[];
}

export interface AnoleDetectionResult {
  totalLizards: number;
  predictions: AnolePrediction[];
  processingTime?: number;
}

export type DetectionMode = 'backend' | 'backend-pytorch' | 'onnx-frontend' | 'auto';

export class AnoleDetectionService {
  private static isOnnxInitialized = false;
  /** Cached result of client-side ONNX environment probe (session). */
  private static clientOnnxProbeResult: boolean | null = null;
  private static clientOnnxProbeInFlight: Promise<boolean> | null = null;

  /** Clear cached probe (e.g. when user switches back to Auto in the UI). */
  static invalidateClientOnnxProbe(): void {
    this.clientOnnxProbeResult = null;
    this.clientOnnxProbeInFlight = null;
  }

  /**
   * Whether browser ONNX is likely to work: models + WASM loader reachable,
   * sane MIME for .mjs, and network/device hints do not favor server-only.
   */
  static async clientOnnxEnvironmentOk(): Promise<boolean> {
    if (this.clientOnnxProbeResult !== null) {
      return this.clientOnnxProbeResult;
    }
    if (this.clientOnnxProbeInFlight) {
      return this.clientOnnxProbeInFlight;
    }
    this.clientOnnxProbeInFlight = this.probeClientOnnxEnvironment()
      .then((ok) => {
        this.clientOnnxProbeResult = ok;
        return ok;
      })
      .finally(() => {
        this.clientOnnxProbeInFlight = null;
      });
    return this.clientOnnxProbeInFlight;
  }

  private static async probeClientOnnxEnvironment(): Promise<boolean> {
    if (typeof window === 'undefined') {
      return false;
    }

    const nav = navigator as Navigator & {
      connection?: { saveData?: boolean; effectiveType?: string };
      deviceMemory?: number;
    };

    if (nav.connection?.saveData) {
      console.log('Auto inference: server preferred (save-data)');
      return false;
    }
    const slow = new Set(['slow-2g', '2g']);
    if (nav.connection?.effectiveType && slow.has(nav.connection.effectiveType)) {
      console.log(`Auto inference: server preferred (network ${nav.connection.effectiveType})`);
      return false;
    }
    if (typeof nav.deviceMemory === 'number' && nav.deviceMemory < 4) {
      console.log('Auto inference: server preferred (deviceMemory < 4 GiB)');
      return false;
    }

    const origin = window.location.origin;
    /** SPA fallback is often index.html (~0.5–3 KiB) with text/html — not a real ONNX. */
    const MIN_ONNX_BYTES = 500_000;
    const assets: { url: string; requireJavascriptMime?: boolean }[] = [
      { url: `${origin}/models/yolo_best.onnx` },
      { url: `${origin}/models/swin_model.onnx` },
      { url: `${origin}/ort-wasm-simd-threaded.jsep.mjs`, requireJavascriptMime: true },
    ];

    try {
      for (const { url, requireJavascriptMime } of assets) {
        const res = await fetch(url, { method: 'HEAD', cache: 'no-store' });
        if (!res.ok) {
          console.log('Auto inference: server preferred (asset HEAD failed)', url, res.status);
          return false;
        }
        if (url.endsWith('.onnx')) {
          const ct = (res.headers.get('content-type') || '').toLowerCase();
          if (ct.includes('html')) {
            console.log('Auto inference: server preferred (.onnx URL serves HTML, likely missing file):', url);
            return false;
          }
          const len = Number(res.headers.get('content-length'));
          if (!Number.isFinite(len) || len < MIN_ONNX_BYTES) {
            console.log(
              'Auto inference: server preferred (.onnx too small or unknown size — not a real model):',
              url,
              'content-length=',
              res.headers.get('content-length')
            );
            return false;
          }
        }
        if (requireJavascriptMime) {
          const ct = (res.headers.get('content-type') || '').toLowerCase();
          if (!ct.includes('javascript')) {
            console.log('Auto inference: server preferred (.mjs MIME not JavaScript):', ct || '(empty)');
            return false;
          }
        }
      }
      console.log('Auto inference: browser ONNX looks viable — will try client first');
      return true;
    } catch {
      console.log('Auto inference: server preferred (probe error)');
      return false;
    }
  }

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
          detectionConfidence: detection.score,
          alternateConfidences: (classification as any).alternateConfidences,
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
   *               - 'auto': Prefer client ONNX when probe passes; else server PyTorch
   */
  static async detect(
    imageFile: File,
    mode: DetectionMode = 'backend'  // Default: backend PyTorch-CPU
  ): Promise<AnoleDetectionResult> {
    if (mode === 'backend') {
      return this.detectBackend(imageFile, true);   // PyTorch (default)
    } else if (mode === 'backend-pytorch') {
      return this.detectBackend(imageFile, true);   // PyTorch
    } else if (mode === 'auto') {
      const useClient = await this.clientOnnxEnvironmentOk();
      if (useClient) {
        try {
          return await this.detectFrontendOnnx(imageFile);
        } catch (error) {
          console.warn('Auto: client ONNX failed, using server PyTorch:', error);
          return this.detectBackend(imageFile, true);
        }
      }
      return this.detectBackend(imageFile, true);
    } else {
      try {
        return await this.detectFrontendOnnx(imageFile);
      } catch (error) {
        console.warn('Frontend ONNX failed, falling back to backend PyTorch:', error);
        return this.detectBackend(imageFile, true);
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

  /**
   * Classify a single cropped region from an image
   * @param imageFile - Original image file
   * @param box - Bounding box coordinates [x1, y1, x2, y2] in original image coordinates
   * @param mode - Detection mode to use
   * @returns Classification result
   */
  static async classifyRegion(
    imageFile: File,
    box: [number, number, number, number],
    mode: DetectionMode = 'backend'
  ): Promise<AnolePrediction> {
    try {
      let effective: Exclude<DetectionMode, 'auto'>;
      if (mode === 'auto') {
        effective = (await this.clientOnnxEnvironmentOk()) ? 'onnx-frontend' : 'backend';
      } else {
        effective = mode;
      }

      // Load the original image
      const img = await this.loadImage(imageFile);
      
      // Extract box coordinates
      const [x1, y1, x2, y2] = box;
      const width = x2 - x1;
      const height = y2 - y1;
      
      // Crop the region
      const croppedCanvas = await this.cropImage(img, x1, y1, width, height);
      
      if (effective === 'onnx-frontend') {
        // Use frontend ONNX classification
        if (!this.isOnnxInitialized) {
          await this.initializeOnnx();
        }
        
        const classification = await OnnxDetectionService.classifyAnole(croppedCanvas);
        
        return {
          species: classification.class,
          scientificName: this.getScientificName(classification.classId),
          confidence: classification.score,
          count: 1,
          boundingBox: box,
        };
      } else {
        // Use backend classification - convert canvas to blob and send
        return new Promise((resolve, reject) => {
          croppedCanvas.toBlob(async (blob) => {
            if (!blob) {
              reject(new Error('Failed to create blob from cropped image'));
              return;
            }
            
            try {
              // Create a File from the blob
              const croppedFile = new File([blob], 'cropped.jpg', { type: 'image/jpeg' });
              
              // Send to backend for classification
              const formData = new FormData();
              formData.append('file', croppedFile);
              
              const url = `${API_URL}/api/predict?use_onnx=${effective !== 'backend' && effective !== 'backend-pytorch'}`;
              const response = await fetch(url, {
                method: 'POST',
                body: formData,
              });
              
              if (!response.ok) {
                throw new Error(`Backend classification failed: ${response.statusText}`);
              }
              
              const result = await response.json();
              
              // Extract the first prediction (since we're classifying a single region)
              if (result.predictions && result.predictions.length > 0) {
                const pred = result.predictions[0];
                resolve({
                  species: pred.species,
                  scientificName: pred.scientificName,
                  confidence: pred.confidence,
                  count: 1,
                  boundingBox: box,
                  alternateConfidences: pred.alternateConfidences,
                });
              } else {
                reject(new Error('No classification result returned'));
              }
            } catch (error) {
              reject(error);
            }
          }, 'image/jpeg', 0.95);
        });
      }
    } catch (error) {
      console.error('Region classification error:', error);
      throw new Error(`Failed to classify region: ${error}`);
    }
  }
}

