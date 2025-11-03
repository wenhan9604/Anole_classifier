import * as ort from 'onnxruntime-web';

export interface Detection {
  bbox: number[]; // [x, y, width, height]
  class: string;
  score: number;
}

export class OnnxDetectionService {
  private static yoloSession: ort.InferenceSession | null = null;
  private static swinSession: ort.InferenceSession | null = null;
  private static isInitialized = false;
  private static isInitializing = false;
  private static classNames: string[] = [];

  // Model configuration - ONNX models expect specific input sizes
  private static readonly YOLO_INPUT_SIZE = 640;
  private static readonly SWIN_INPUT_SIZE = 384; // Swin base patch4 window12 384
  
  // Anole species class names
  private static readonly DEFAULT_CLASS_NAMES = [
    'Bark Anole',
    'Brown Anole',
    'Crested Anole',
    'Green Anole',
    'Knight Anole'
  ];

  /**
   * Initialize ONNX Runtime and load both models
   * @param yoloModelUrl - URL to the YOLO ONNX model
   * @param swinModelUrl - URL to the Swin ONNX model
   * @param customClassNames - Optional custom class names array
   */
  static async initialize(
    yoloModelUrl?: string,
    swinModelUrl?: string,
    customClassNames?: string[]
  ): Promise<void> {
    if (this.isInitialized) {
      console.log('ONNX models already initialized');
      return;
    }

    if (this.isInitializing) {
      console.log('ONNX models are already initializing...');
      while (this.isInitializing) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      return;
    }

    try {
      this.isInitializing = true;
      console.log('Initializing ONNX Runtime...');

      // Configure ONNX Runtime environment
      // Let ONNX Runtime auto-detect the best configuration
      if (typeof window !== 'undefined') {
        const baseUrl = window.location.origin;
        ort.env.wasm.wasmPaths = baseUrl + '/';
        console.log('WASM paths set to:', baseUrl + '/');
      }
      
      // Note: Not setting numThreads or simd explicitly - let ONNX Runtime decide
      // based on available WASM files (ort-wasm-simd-threaded.wasm is available)

      const defaultYoloUrl = yoloModelUrl || '/models/yolo_best.onnx';
      const defaultSwinUrl = swinModelUrl || '/models/swin_model.onnx';
      
      console.log(`Loading YOLO ONNX model from ${defaultYoloUrl}...`);
      console.log(`Loading Swin ONNX model from ${defaultSwinUrl}...`);

      // Simplified session options - use WASM backend only for compatibility
      const sessionOptions = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic' as const,
        enableCpuMemArena: true,
        enableMemPattern: true,
      };

      // Load YOLO model
      console.log('Loading YOLO model with WASM backend...');
      try {
        this.yoloSession = await ort.InferenceSession.create(defaultYoloUrl, sessionOptions);
        console.log('✓ YOLO model loaded successfully');
      } catch (error) {
        console.error('Failed to load YOLO model:', error);
        throw new Error(`Failed to initialize YOLO model: ${error}`);
      }

      // Load Swin model
      console.log('Loading Swin model with WASM backend...');
      try {
        this.swinSession = await ort.InferenceSession.create(defaultSwinUrl, sessionOptions);
        console.log('✓ Swin model loaded successfully');
      } catch (error) {
        console.error('Failed to load Swin model:', error);
        throw new Error(`Failed to initialize Swin model: ${error}`);
      }

      console.log('YOLO inputs:', this.yoloSession.inputNames);
      console.log('YOLO outputs:', this.yoloSession.outputNames);
      console.log('Swin inputs:', this.swinSession.inputNames);
      console.log('Swin outputs:', this.swinSession.outputNames);

      // Set class names
      this.classNames = customClassNames || this.DEFAULT_CLASS_NAMES;
      console.log(`Using ${this.classNames.length} class names:`, this.classNames);

      this.isInitialized = true;
      console.log('ONNX models loaded and ready');
    } catch (error) {
      console.error('Failed to initialize ONNX models:', error);
      throw new Error(`Failed to load ONNX models: ${error}`);
    } finally {
      this.isInitializing = false;
    }
  }

  /**
   * Check if models are initialized
   */
  static isReady(): boolean {
    return this.isInitialized && this.yoloSession !== null && this.swinSession !== null;
  }

  /**
   * Get all available class names
   */
  static getClassNames(): string[] {
    return [...this.classNames];
  }

  /**
   * Preprocess image with letterboxing for YOLO
   */
  private static preprocessYoloImage(imageElement: HTMLImageElement | HTMLCanvasElement): {
    tensor: ort.Tensor;
    originalWidth: number;
    originalHeight: number;
    padX: number;
    padY: number;
    scale: number;
  } {
    const canvas = document.createElement('canvas');
    canvas.width = this.YOLO_INPUT_SIZE;
    canvas.height = this.YOLO_INPUT_SIZE;
    const ctx = canvas.getContext('2d')!;

    const originalWidth = imageElement.width || (imageElement as HTMLImageElement).naturalWidth;
    const originalHeight = imageElement.height || (imageElement as HTMLImageElement).naturalHeight;

    // Calculate scaling factor (letterboxing)
    const scale = Math.min(
      this.YOLO_INPUT_SIZE / originalWidth,
      this.YOLO_INPUT_SIZE / originalHeight
    );
    
    const scaledWidth = originalWidth * scale;
    const scaledHeight = originalHeight * scale;
    
    const padX = (this.YOLO_INPUT_SIZE - scaledWidth) / 2;
    const padY = (this.YOLO_INPUT_SIZE - scaledHeight) / 2;

    // Fill with gray background (114, 114, 114) - match backend exactly
    ctx.fillStyle = 'rgb(114, 114, 114)';
    ctx.fillRect(0, 0, this.YOLO_INPUT_SIZE, this.YOLO_INPUT_SIZE);

    // Draw letterboxed image
    ctx.drawImage(imageElement, padX, padY, scaledWidth, scaledHeight);

    // Get image data and convert to tensor
    const imageData = ctx.getImageData(0, 0, this.YOLO_INPUT_SIZE, this.YOLO_INPUT_SIZE);
    const pixels = imageData.data;

    // Convert to RGB channels and normalize
    const r: number[] = [];
    const g: number[] = [];
    const b: number[] = [];
    
    for (let i = 0; i < pixels.length; i += 4) {
      r.push(pixels[i] / 255.0);
      g.push(pixels[i + 1] / 255.0);
      b.push(pixels[i + 2] / 255.0);
    }

    const input = Float32Array.from([...r, ...g, ...b]);
    const tensor = new ort.Tensor('float32', input, [1, 3, this.YOLO_INPUT_SIZE, this.YOLO_INPUT_SIZE]);

    return { tensor, originalWidth, originalHeight, padX, padY, scale };
  }

  /**
   * Preprocess image for Swin Transformer (384x384)
   */
  private static preprocessSwinImage(imageElement: HTMLImageElement | HTMLCanvasElement): ort.Tensor {
    const canvas = document.createElement('canvas');
    canvas.width = this.SWIN_INPUT_SIZE;
    canvas.height = this.SWIN_INPUT_SIZE;
    const ctx = canvas.getContext('2d')!;

    // Draw resized image to 384x384
    ctx.drawImage(imageElement, 0, 0, this.SWIN_INPUT_SIZE, this.SWIN_INPUT_SIZE);

    // Get image data
    const imageData = ctx.getImageData(0, 0, this.SWIN_INPUT_SIZE, this.SWIN_INPUT_SIZE);
    const pixels = imageData.data;

    // ImageNet normalization
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    const r: number[] = [];
    const g: number[] = [];
    const b: number[] = [];
    
    for (let i = 0; i < pixels.length; i += 4) {
      r.push((pixels[i] / 255.0 - mean[0]) / std[0]);
      g.push((pixels[i + 1] / 255.0 - mean[1]) / std[1]);
      b.push((pixels[i + 2] / 255.0 - mean[2]) / std[2]);
    }

    const input = Float32Array.from([...r, ...g, ...b]);
    return new ort.Tensor('float32', input, [1, 3, this.SWIN_INPUT_SIZE, this.SWIN_INPUT_SIZE]);
  }

  /**
   * Detect anoles in an image using YOLO
   */
  static async detectAnoles(
    imageElement: HTMLImageElement | HTMLCanvasElement,
    scoreThreshold: number = 0.25,
    iouThreshold: number = 0.45
  ): Promise<Detection[]> {
    console.log('🔍 detectAnoles called with threshold:', scoreThreshold);
    
    if (!this.isReady()) {
      throw new Error('ONNX models not initialized. Call initialize() first.');
    }

    try {
      console.log('🔍 Starting YOLO preprocessing...');
      // Preprocess image
      const { tensor, originalWidth, originalHeight, padX, padY, scale } = 
        this.preprocessYoloImage(imageElement);
      
      console.log('🔍 Running YOLO inference...');

      // Run YOLO inference
      const feeds: Record<string, ort.Tensor> = {};
      feeds[this.yoloSession!.inputNames[0]] = tensor;
      const results = await this.yoloSession!.run(feeds);

      // Get output tensor
      const output = results[this.yoloSession!.outputNames[0]];

      // Process detections
      const detections = this.processYoloOutput(
        output,
        originalWidth,
        originalHeight,
        padX,
        padY,
        scale,
        scoreThreshold,
        iouThreshold
      );

      return detections;
    } catch (error) {
      console.error('Detection error:', error);
      throw error;
    }
  }

  /**
   * Classify an anole crop using Swin Transformer
   */
  static async classifyAnole(
    imageElement: HTMLImageElement | HTMLCanvasElement
  ): Promise<{ class: string; score: number; classId: number }> {
    if (!this.isReady()) {
      throw new Error('ONNX models not initialized. Call initialize() first.');
    }

    try {
      // Preprocess image
      const tensor = this.preprocessSwinImage(imageElement);

      // Run Swin inference
      const feeds: Record<string, ort.Tensor> = {};
      feeds[this.swinSession!.inputNames[0]] = tensor;
      const results = await this.swinSession!.run(feeds);

      // Get output tensor (logits)
      const output = results[this.swinSession!.outputNames[0]];
      const logits = output.data as Float32Array;

      // Apply softmax
      const maxLogit = Math.max(...Array.from(logits));
      const expLogits = Array.from(logits).map(l => Math.exp(l - maxLogit));
      const sumExp = expLogits.reduce((a, b) => a + b, 0);
      const probs = expLogits.map(e => e / sumExp);

      // Get predicted class
      const classId = probs.indexOf(Math.max(...probs));
      const score = probs[classId];
      const className = this.classNames[classId] || `Class ${classId}`;

      return { class: className, score, classId };
    } catch (error) {
      console.error('Classification error:', error);
      throw error;
    }
  }

  /**
   * Process YOLO output and apply NMS
   */
  private static processYoloOutput(
    output: ort.Tensor,
    originalWidth: number,
    originalHeight: number,
    padX: number,
    padY: number,
    scale: number,
    scoreThreshold: number,
    iouThreshold: number
  ): Detection[] {
    const data = output.data as Float32Array;
    const dims = output.dims;
    
    console.log('YOLO Output dims:', dims);
    console.log('Score threshold:', scoreThreshold);

    // Handle different output formats
    let numDetections: number;
    let valuesPerDetection: number;
    
    if (dims[1] < dims[2]) {
      valuesPerDetection = dims[1];
      numDetections = dims[2];
    } else {
      numDetections = dims[1];
      valuesPerDetection = dims[2];
    }
    
    const numClasses = valuesPerDetection - 5;
    const isTransposed = dims[1] < dims[2];
    const isSingleClass = numClasses === 0; // Single-class model (just objectness)
    
    console.log(`Processing ${numDetections} detections, ${numClasses} classes, transposed=${isTransposed}, singleClass=${isSingleClass}`);

    const detections: Detection[] = [];
    const boxes: number[][] = [];
    const scores: number[] = [];
    const classIds: number[] = [];
    
    let maxScoreSeen = 0;
    let countAboveThreshold = 0;

    // Parse detections
    for (let i = 0; i < numDetections; i++) {
      const getValue = (valueIndex: number) => {
        return isTransposed 
          ? data[valueIndex * numDetections + i]
          : data[i * valuesPerDetection + valueIndex];
      };

      const x = getValue(0);
      const y = getValue(1);
      const w = getValue(2);
      const h = getValue(3);
      const objectness = this.sigmoid(getValue(4));
      
      let maxScore = 0;
      let maxClassId = 0;
      
      if (isSingleClass) {
        // Single-class model: use objectness directly
        maxScore = objectness;
        maxClassId = 0; // Only one class
      } else {
        // Multi-class model: find best class
        for (let c = 0; c < numClasses; c++) {
          const classScore = this.sigmoid(getValue(5 + c));
          const finalScore = objectness * classScore;
          if (finalScore > maxScore) {
            maxScore = finalScore;
            maxClassId = c;
          }
        }
      }
      
      if (maxScore > maxScoreSeen) {
        maxScoreSeen = maxScore;
      }

      if (maxScore > scoreThreshold) {
        countAboveThreshold++;
        boxes.push([x, y, w, h]);
        scores.push(maxScore);
        classIds.push(maxClassId);
      }
    }
    
    console.log(`Max score seen: ${maxScoreSeen.toFixed(4)}`);
    console.log(`Detections above threshold (${scoreThreshold}): ${countAboveThreshold}`);
    console.log(`Top 5 scores:`, scores.slice(0, 5).map(s => s.toFixed(4)));

    // Apply NMS
    if (boxes.length > 0) {
      const selectedIndices = this.nonMaxSuppression(boxes, scores, iouThreshold);
      
      for (const idx of selectedIndices) {
        const box = boxes[idx];
        const [cx, cy, w, h] = box;
        
        // Convert from center format to corner format
        const x1_640 = cx - w / 2;
        const y1_640 = cy - h / 2;
        const x2_640 = cx + w / 2;
        const y2_640 = cy + h / 2;
        
        // Remove letterbox padding and scale back to original size
        const x1 = (x1_640 - padX) / scale;
        const y1 = (y1_640 - padY) / scale;
        const x2 = (x2_640 - padX) / scale;
        const y2 = (y2_640 - padY) / scale;
        
        const width = x2 - x1;
        const height = y2 - y1;

        detections.push({
          bbox: [x1, y1, width, height],
          class: this.classNames[classIds[idx]] || `class_${classIds[idx]}`,
          score: scores[idx],
        });
      }
    }

    return detections;
  }

  /**
   * Sigmoid activation function
   */
  private static sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  /**
   * Non-Maximum Suppression
   */
  private static nonMaxSuppression(
    boxes: number[][],
    scores: number[],
    iouThreshold: number
  ): number[] {
    const indices = scores
      .map((score, idx) => ({ score, idx }))
      .sort((a, b) => b.score - a.score)
      .map(item => item.idx);

    const selected: number[] = [];
    const suppressed = new Set<number>();

    for (const idx of indices) {
      if (suppressed.has(idx)) continue;

      selected.push(idx);

      for (const otherIdx of indices) {
        if (otherIdx === idx || suppressed.has(otherIdx)) continue;

        const iou = this.calculateIoU(boxes[idx], boxes[otherIdx]);
        if (iou > iouThreshold) {
          suppressed.add(otherIdx);
        }
      }
    }

    return selected;
  }

  /**
   * Calculate Intersection over Union
   */
  private static calculateIoU(box1: number[], box2: number[]): number {
    const x1Min = box1[0] - box1[2] / 2;
    const y1Min = box1[1] - box1[3] / 2;
    const x1Max = box1[0] + box1[2] / 2;
    const y1Max = box1[1] + box1[3] / 2;

    const x2Min = box2[0] - box2[2] / 2;
    const y2Min = box2[1] - box2[3] / 2;
    const x2Max = box2[0] + box2[2] / 2;
    const y2Max = box2[1] + box2[3] / 2;

    const xMin = Math.max(x1Min, x2Min);
    const yMin = Math.max(y1Min, y2Min);
    const xMax = Math.min(x1Max, x2Max);
    const yMax = Math.min(y1Max, y2Max);

    if (xMax < xMin || yMax < yMin) return 0;

    const intersection = (xMax - xMin) * (yMax - yMin);
    const area1 = box1[2] * box1[3];
    const area2 = box2[2] * box2[3];
    const union = area1 + area2 - intersection;

    return intersection / union;
  }

  /**
   * Dispose of models and free memory
   */
  static async dispose(): Promise<void> {
    if (this.yoloSession) {
      await this.yoloSession.release();
      this.yoloSession = null;
    }
    if (this.swinSession) {
      await this.swinSession.release();
      this.swinSession = null;
    }
    this.isInitialized = false;
    console.log('ONNX models disposed');
  }
}

