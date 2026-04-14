/**
 * Browser ONNX inference (YOLO + Swin) via onnxruntime-web.
 * Supports WASM-only, WebGPU-first (with WASM fallback), or auto selection.
 */
import type * as OrtTypes from 'onnxruntime-web';

type OrtModule = typeof OrtTypes;

export type OnnxExecutionPreference = 'auto' | 'webgpu' | 'wasm';

export interface YoloStageTimings {
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
}

export interface Detection {
  bbox: number[]; // [x, y, width, height]
  class: string;
  score: number;
}

export interface DetectAnolesResult {
  detections: Detection[];
  yoloTimings: YoloStageTimings;
}

export interface ClassifyAnoleResult {
  class: string;
  score: number;
  classId: number;
  preprocessMs: number;
  inferenceMs: number;
}

export class OnnxDetectionService {
  private static yoloSession: OrtTypes.InferenceSession | null = null;
  private static swinSession: OrtTypes.InferenceSession | null = null;
  private static ortNs: OrtModule | null = null;
  private static isInitialized = false;
  private static isInitializing = false;
  private static classNames: string[] = [];
  private static initExecutionPreference: OnnxExecutionPreference | null = null;
  private static activeExecutionProviderLabel = '';

  private static readonly YOLO_INPUT_SIZE = 640;
  private static readonly SWIN_INPUT_SIZE = 384;

  private static readonly DEFAULT_CLASS_NAMES = [
    'Bark Anole',
    'Brown Anole',
    'Crested Anole',
    'Green Anole',
    'Knight Anole',
  ];

  private static ort(): OrtModule {
    if (!this.ortNs) {
      throw new Error('ONNX Runtime not loaded. Call initialize() first.');
    }
    return this.ortNs;
  }

  /** Whether WebGPU API exists (does not guarantee model support). */
  static isWebGpuApiAvailable(): boolean {
    return typeof navigator !== 'undefined' && !!(navigator as Navigator & { gpu?: unknown }).gpu;
  }

  static getExecutionProviderLabel(): string {
    return this.activeExecutionProviderLabel || 'unknown';
  }

  static getInitExecutionPreference(): OnnxExecutionPreference | null {
    return this.initExecutionPreference;
  }

  private static async createSessions(
    ortMod: OrtModule,
    yoloUrl: string,
    swinUrl: string,
    executionProviders: string[],
    label: string,
  ): Promise<void> {
    if (typeof window !== 'undefined') {
      const baseUrl = window.location.origin + '/';
      ortMod.env.wasm.wasmPaths = baseUrl;
      console.log('WASM paths set to:', baseUrl);
    }

    const sessionOptions = {
      executionProviders,
      graphOptimizationLevel: 'basic' as const,
      enableCpuMemArena: true,
      enableMemPattern: true,
    };

    console.log(`Creating ONNX sessions with providers=[${executionProviders.join(', ')}] (${label})...`);

    const yoloSession = await ortMod.InferenceSession.create(yoloUrl, sessionOptions);
    try {
      const swinSession = await ortMod.InferenceSession.create(swinUrl, sessionOptions);
      this.yoloSession = yoloSession;
      this.swinSession = swinSession;
      this.ortNs = ortMod;
      this.activeExecutionProviderLabel = label;
      console.log(`✓ ONNX sessions ready (${label})`);
    } catch (err) {
      await yoloSession.release();
      throw err;
    }
  }

  /**
   * Initialize ONNX Runtime and load both models.
   */
  static async initialize(
    yoloModelUrl?: string,
    swinModelUrl?: string,
    customClassNames?: string[],
    executionPreference: OnnxExecutionPreference = 'auto',
  ): Promise<void> {
    if (this.isInitialized && this.initExecutionPreference === executionPreference) {
      console.log('ONNX models already initialized for preference:', executionPreference);
      return;
    }

    if (this.isInitializing) {
      console.log('ONNX models are already initializing...');
      while (this.isInitializing) {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      if (this.isInitialized && this.initExecutionPreference === executionPreference) {
        return;
      }
    }

    try {
      this.isInitializing = true;

      if (this.isInitialized && this.initExecutionPreference !== executionPreference) {
        console.log('Execution preference changed; disposing previous sessions...');
        await this.dispose();
      }

      console.log('Initializing ONNX Runtime... preference=', executionPreference);

      const defaultYoloUrl = yoloModelUrl || '/models/yolo_best.onnx';
      const defaultSwinUrl = swinModelUrl || '/models/swin_model.onnx';
      console.log(`Loading YOLO ONNX model from ${defaultYoloUrl}...`);
      console.log(`Loading Swin ONNX model from ${defaultSwinUrl}...`);

      const webgpuApi = this.isWebGpuApiAvailable();

      if (executionPreference === 'wasm') {
        const ortWasm: OrtModule = await import('onnxruntime-web');
        await this.createSessions(ortWasm, defaultYoloUrl, defaultSwinUrl, ['wasm'], 'wasm');
      } else if (executionPreference === 'webgpu') {
        if (!webgpuApi) {
          throw new Error('WebGPU requested but navigator.gpu is not available in this browser.');
        }
        const ortWebgpu: OrtModule = await import('onnxruntime-web/webgpu');
        await this.createSessions(
          ortWebgpu,
          defaultYoloUrl,
          defaultSwinUrl,
          ['webgpu', 'wasm'],
          'webgpu+wasm',
        );
      } else {
        // auto: prefer WebGPU bundle + providers when API exists; fall back to WASM-only.
        if (webgpuApi) {
          try {
            const ortWebgpu: OrtModule = await import('onnxruntime-web/webgpu');
            await this.createSessions(
              ortWebgpu,
              defaultYoloUrl,
              defaultSwinUrl,
              ['webgpu', 'wasm'],
              'webgpu+wasm',
            );
          } catch (e) {
            console.warn('WebGPU path failed, falling back to WASM-only:', e);
            const ortWasm: OrtModule = await import('onnxruntime-web');
            await this.createSessions(ortWasm, defaultYoloUrl, defaultSwinUrl, ['wasm'], 'wasm');
          }
        } else {
          const ortWasm: OrtModule = await import('onnxruntime-web');
          await this.createSessions(ortWasm, defaultYoloUrl, defaultSwinUrl, ['wasm'], 'wasm');
        }
      }

      console.log('YOLO inputs:', this.yoloSession?.inputNames);
      console.log('YOLO outputs:', this.yoloSession?.outputNames);
      console.log('Swin inputs:', this.swinSession?.inputNames);
      console.log('Swin outputs:', this.swinSession?.outputNames);

      this.classNames = customClassNames || this.DEFAULT_CLASS_NAMES;
      console.log(`Using ${this.classNames.length} class names:`, this.classNames);

      this.initExecutionPreference = executionPreference;
      this.isInitialized = true;
      console.log('ONNX models loaded. Active execution provider label:', this.activeExecutionProviderLabel);
    } catch (error) {
      console.error('Failed to initialize ONNX models:', error);
      this.ortNs = null;
      this.yoloSession = null;
      this.swinSession = null;
      this.initExecutionPreference = null;
      this.activeExecutionProviderLabel = '';
      this.isInitialized = false;
      throw new Error(`Failed to load ONNX models: ${error}`);
    } finally {
      this.isInitializing = false;
    }
  }

  static isReady(): boolean {
    return this.isInitialized && this.yoloSession !== null && this.swinSession !== null;
  }

  static getClassNames(): string[] {
    return [...this.classNames];
  }

  private static preprocessYoloImage(imageElement: HTMLImageElement | HTMLCanvasElement): {
    tensor: OrtTypes.Tensor;
    originalWidth: number;
    originalHeight: number;
    padX: number;
    padY: number;
    scale: number;
  } {
    const ort = this.ort();
    const canvas = document.createElement('canvas');
    canvas.width = this.YOLO_INPUT_SIZE;
    canvas.height = this.YOLO_INPUT_SIZE;
    const ctx = canvas.getContext('2d')!;

    const originalWidth = imageElement.width || (imageElement as HTMLImageElement).naturalWidth;
    const originalHeight = imageElement.height || (imageElement as HTMLImageElement).naturalHeight;

    const scale = Math.min(
      this.YOLO_INPUT_SIZE / originalWidth,
      this.YOLO_INPUT_SIZE / originalHeight,
    );

    const scaledWidth = originalWidth * scale;
    const scaledHeight = originalHeight * scale;

    const padX = (this.YOLO_INPUT_SIZE - scaledWidth) / 2;
    const padY = (this.YOLO_INPUT_SIZE - scaledHeight) / 2;

    ctx.fillStyle = 'rgb(114, 114, 114)';
    ctx.fillRect(0, 0, this.YOLO_INPUT_SIZE, this.YOLO_INPUT_SIZE);
    ctx.drawImage(imageElement, padX, padY, scaledWidth, scaledHeight);

    const imageData = ctx.getImageData(0, 0, this.YOLO_INPUT_SIZE, this.YOLO_INPUT_SIZE);
    const pixels = imageData.data;

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

  private static preprocessSwinImage(imageElement: HTMLImageElement | HTMLCanvasElement): OrtTypes.Tensor {
    const ort = this.ort();
    const canvas = document.createElement('canvas');
    canvas.width = this.SWIN_INPUT_SIZE;
    canvas.height = this.SWIN_INPUT_SIZE;
    const ctx = canvas.getContext('2d')!;

    ctx.drawImage(imageElement, 0, 0, this.SWIN_INPUT_SIZE, this.SWIN_INPUT_SIZE);

    const imageData = ctx.getImageData(0, 0, this.SWIN_INPUT_SIZE, this.SWIN_INPUT_SIZE);
    const pixels = imageData.data;

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

  static async detectAnoles(
    imageElement: HTMLImageElement | HTMLCanvasElement,
    scoreThreshold: number = 0.25,
    iouThreshold: number = 0.45,
  ): Promise<DetectAnolesResult> {
    console.log('🔍 detectAnoles called with threshold:', scoreThreshold);

    if (!this.isReady()) {
      throw new Error('ONNX models not initialized. Call initialize() first.');
    }

    try {
      const tPre0 = performance.now();
      const { tensor, originalWidth, originalHeight, padX, padY, scale } =
        this.preprocessYoloImage(imageElement);
      const tPre1 = performance.now();

      const feeds: Record<string, OrtTypes.Tensor> = {};
      feeds[this.yoloSession!.inputNames[0]] = tensor;
      const results = await this.yoloSession!.run(feeds);
      const tInf1 = performance.now();

      const output = results[this.yoloSession!.outputNames[0]];

      const detections = this.processYoloOutput(
        output,
        originalWidth,
        originalHeight,
        padX,
        padY,
        scale,
        scoreThreshold,
        iouThreshold,
      );
      const tPost1 = performance.now();

      return {
        detections,
        yoloTimings: {
          preprocessMs: tPre1 - tPre0,
          inferenceMs: tInf1 - tPre1,
          postprocessMs: tPost1 - tInf1,
        },
      };
    } catch (error) {
      console.error('Detection error:', error);
      throw error;
    }
  }

  static async classifyAnole(
    imageElement: HTMLImageElement | HTMLCanvasElement,
  ): Promise<ClassifyAnoleResult> {
    if (!this.isReady()) {
      throw new Error('ONNX models not initialized. Call initialize() first.');
    }

    try {
      const tPre0 = performance.now();
      const tensor = this.preprocessSwinImage(imageElement);
      const tPre1 = performance.now();

      const feeds: Record<string, OrtTypes.Tensor> = {};
      feeds[this.swinSession!.inputNames[0]] = tensor;
      const results = await this.swinSession!.run(feeds);
      const tInf1 = performance.now();

      const output = results[this.swinSession!.outputNames[0]];
      const logits = output.data as Float32Array;

      const maxLogit = Math.max(...Array.from(logits));
      const expLogits = Array.from(logits).map((l) => Math.exp(l - maxLogit));
      const sumExp = expLogits.reduce((a, b) => a + b, 0);
      const probs = expLogits.map((e) => e / sumExp);

      const classId = probs.indexOf(Math.max(...probs));
      const score = probs[classId];
      const className = this.classNames[classId] || `Class ${classId}`;

      return {
        class: className,
        score,
        classId,
        preprocessMs: tPre1 - tPre0,
        inferenceMs: tInf1 - tPre1,
      };
    } catch (error) {
      console.error('Classification error:', error);
      throw error;
    }
  }

  private static processYoloOutput(
    output: OrtTypes.Tensor,
    _originalWidth: number,
    _originalHeight: number,
    padX: number,
    padY: number,
    scale: number,
    scoreThreshold: number,
    iouThreshold: number,
  ): Detection[] {
    const data = output.data as Float32Array;
    const dims = output.dims;

    console.log('YOLO Output dims:', dims);
    console.log('Score threshold:', scoreThreshold);

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
    const isSingleClass = numClasses === 0;

    console.log(
      `Processing ${numDetections} detections, ${numClasses} classes, transposed=${isTransposed}, singleClass=${isSingleClass}`,
    );

    const detections: Detection[] = [];
    const boxes: number[][] = [];
    const scores: number[] = [];
    const classIds: number[] = [];

    let maxScoreSeen = 0;
    let countAboveThreshold = 0;

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
        maxScore = objectness;
        maxClassId = 0;
      } else {
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
    console.log(
      `Top 5 scores:`,
      scores.slice(0, 5).map((s) => s.toFixed(4)),
    );

    if (boxes.length > 0) {
      const selectedIndices = this.nonMaxSuppression(boxes, scores, iouThreshold);

      for (const idx of selectedIndices) {
        const box = boxes[idx];
        const [cx, cy, w, h] = box;

        const x1_640 = cx - w / 2;
        const y1_640 = cy - h / 2;
        const x2_640 = cx + w / 2;
        const y2_640 = cy + h / 2;

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

  private static sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  private static nonMaxSuppression(boxes: number[][], scores: number[], iouThreshold: number): number[] {
    const indices = scores
      .map((score, idx) => ({ score, idx }))
      .sort((a, b) => b.score - a.score)
      .map((item) => item.idx);

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

  static async dispose(): Promise<void> {
    if (this.yoloSession) {
      await this.yoloSession.release();
      this.yoloSession = null;
    }
    if (this.swinSession) {
      await this.swinSession.release();
      this.swinSession = null;
    }
    this.ortNs = null;
    this.isInitialized = false;
    this.initExecutionPreference = null;
    this.activeExecutionProviderLabel = '';
    console.log('ONNX models disposed');
  }
}
