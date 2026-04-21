import { useState, useRef, useEffect, useCallback } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  AnoleDetectionService,
  type DetectionMode,
  type FrontendInferenceTimings,
} from "../services/AnoleDetectionService";
import { ResizableBoundingBox } from "../components/ResizableBoundingBox";
import { toast } from 'react-hot-toast';
import {
  iNaturalistAPI,
  getCurrentLocation,
  type iNaturalistAuthStatus,
} from "../services/iNaturalistService";
import { PersistenceService } from "../services/PersistenceService";

// Define the 5 Florida anole species (for reference)
const FLORIDA_ANOLE_SPECIES = [
  { name: "Green Anole", scientific: "Anolis carolinensis", common: "American Green Anole" },
  { name: "Brown Anole", scientific: "Anolis sagrei", common: "Cuban Brown Anole" },
  { name: "Crested Anole", scientific: "Anolis cristatellus", common: "Puerto Rican Crested Anole" },
  { name: "Knight Anole", scientific: "Anolis equestris", common: "Cuban Knight Anole" },
  { name: "Bark Anole", scientific: "Anolis distichus", common: "Bark Anole" }
];

/** Canonical mode for `<select>` (server paths collapse to `backend`). */
function normalizeInferenceModeForSelect(mode: DetectionMode): DetectionMode {
  if (mode === "backend-pytorch") return "backend";
  if (mode === "onnx-frontend") return "onnx-frontend-auto";
  return mode;
}

const INFERENCE_LOCATION_OPTIONS: { value: DetectionMode; label: string }[] = [
  { value: "auto", label: "Auto" },
  { value: "backend", label: "Server — PyTorch (CPU)" },
  { value: "onnx-frontend-auto", label: "This device — GPU if available (WebGPU or WASM)" },
  { value: "onnx-frontend-wasm", label: "This device — WASM only (no WebGPU)" },
  { value: "onnx-frontend-gpu", label: "This device — WebGPU only" },
];

function detectionModeToGpuQuery(mode: DetectionMode): string | null {
  switch (mode) {
    case "auto":
      return null;
    case "backend":
    case "backend-pytorch":
      return null;
    case "onnx-frontend":
    case "onnx-frontend-auto":
      return "client-side";
    case "onnx-frontend-wasm":
      return "client-wasm";
    case "onnx-frontend-gpu":
      return "client-gpu";
    default:
      return null;
  }
}

interface AlternateConfidence {
  classIndex: number;
  species: string;
  scientificName: string;
  confidence: number;
  relativeConfidence: number;
}

interface PredictionResult {
  species: string;
  scientificName: string;
  confidence: number;
  count: number;
  box?: [number, number, number, number]; // [x1, y1, x2, y2] bounding box coordinates
  altConfidences?: AlternateConfidence[];
  isManualCorrection?: boolean;
}

interface DetectionResult {
  totalLizards: number;
  predictions: PredictionResult[];
  imageUrl?: string;
  /** Client ONNX perf (browser only). */
  clientTimings?: FrontendInferenceTimings;
}

export default function PredictionPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadingToiNaturalist, setUploadingToiNaturalist] = useState(false);
  const [inatStatus, setInatStatus] = useState<iNaturalistAuthStatus | null>(null);
  const [inatStatusLoading, setInatStatusLoading] = useState(false);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const [detectionMode, setDetectionMode] = useState<DetectionMode>('auto');
  const [reclassifyingIndex, setReclassifyingIndex] = useState<number | null>(null);
  const [isRestored, setIsRestored] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawingBox, setDrawingBox] = useState<{ startX: number; startY: number; endX: number; endY: number } | null>(null);
  const [isClassifyingDrawnBox, setIsClassifyingDrawnBox] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const reclassifyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);




  // Inference mode from ?gpu=… (default / no param = Auto probe; client-side = WebGPU-first auto)
  useEffect(() => {
    const gpuParam = searchParams.get('gpu');
    if (gpuParam === 'client-side' || gpuParam === 'client') {
      setDetectionMode('onnx-frontend-auto');
      console.log('Client-side ONNX (auto: WebGPU if available, else WASM) via ?gpu=client-side');
    } else if (gpuParam === 'client-wasm') {
      setDetectionMode('onnx-frontend-wasm');
      console.log('Client-side ONNX (WASM only) via ?gpu=client-wasm');
    } else if (gpuParam === 'client-gpu') {
      setDetectionMode('onnx-frontend-gpu');
      console.log('Client-side ONNX (WebGPU required) via ?gpu=client-gpu');
    } else if (gpuParam === 'server') {
      setDetectionMode('backend');
      console.log('Backend PyTorch (CPU) via ?gpu=server');
    } else if (gpuParam === 'auto' || gpuParam == null) {
      setDetectionMode('auto');
      console.log('Auto inference (probe client ONNX vs server)');
    } else {
      setDetectionMode('auto');
      console.log('Auto inference (default; unrecognized ?gpu= value)');
    }
  }, [searchParams]);

  const refreshInatStatus = useCallback(async () => {
    if (!iNaturalistAPI.isBackendConfigured()) {
      setInatStatus(null);
      return;
    }
    setInatStatusLoading(true);
    try {
      const s = await iNaturalistAPI.getAuthStatus();
      setInatStatus(s);
    } catch (e) {
      console.warn("iNaturalist status:", e);
      setInatStatus({ connected: false, expiresAt: null });
    } finally {
      setInatStatusLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshInatStatus();
  }, [refreshInatStatus]);

  useEffect(() => {
    if (searchParams.get("inat") === "connected") {
      toast.success("Connected to iNaturalist");
      void refreshInatStatus();
      const next = new URLSearchParams(searchParams);
      next.delete("inat");
      setSearchParams(next, { replace: true });
    }
  }, [searchParams, setSearchParams, refreshInatStatus]);

  useEffect(() => {
    return () => {
      void AnoleDetectionService.disposeClientOnnx();
    };
  }, []);

  useEffect(() => {
    if (detectionMode === 'backend' || detectionMode === 'backend-pytorch') {
      void AnoleDetectionService.disposeClientOnnx();
    }
  }, [detectionMode]);

  // Persistence: Restore image and result on mount
  useEffect(() => {
    const restore = async () => {
      try {
        const [savedFile, savedResult] = await Promise.all([
          PersistenceService.loadImage(),
          PersistenceService.loadResult()
        ]);

        if (savedFile) {
          setSelectedFile(savedFile);
          const url = URL.createObjectURL(savedFile);
          setPreviewUrl(url);
          console.log("Restored image from persistence");
        }

        if (savedResult) {
          setDetectionResult(savedResult);
          console.log("Restored detection result from persistence");
        }
      } catch (e) {
        console.warn("Failed to restore from persistence:", e);
      } finally {
        setIsRestored(true);
      }
    };
    void restore();
  }, []);

  // Persistence: Save image when it changes
  useEffect(() => {
    if (isRestored && selectedFile) {
      void PersistenceService.saveImage(selectedFile);
    }
  }, [selectedFile, isRestored]);

  // Persistence: Save result when it changes
  useEffect(() => {
    if (isRestored) {
      // We save null too, so clearing works
      void PersistenceService.saveResult(detectionResult);
    }
  }, [detectionResult, isRestored]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (reclassifyTimeoutRef.current) {
        clearTimeout(reclassifyTimeoutRef.current);
      }
    };
  }, []);

  const handleInferenceLocationChange = useCallback(
    (mode: DetectionMode) => {
      if (mode === "auto") {
        AnoleDetectionService.invalidateClientOnnxProbe();
      }
      const canonical =
        mode === "backend-pytorch"
          ? "backend"
          : mode === "onnx-frontend"
            ? "onnx-frontend-auto"
            : mode;
      setDetectionMode(canonical);
      setDetectionResult(null);
      const gpu = detectionModeToGpuQuery(canonical);
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (gpu) next.set("gpu", gpu);
          else next.delete("gpu");
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setDetectionResult(null); // Clear previous results
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;
    
    setIsLoading(true);
    
    try {
      // Use AnoleDetectionService with the selected detection mode
      const modeLabel =
        detectionMode === "auto"
          ? "Auto (client ONNX if viable, else server)"
          : AnoleDetectionService.isOnnxFrontendMode(detectionMode)
            ? `client-side ONNX (${AnoleDetectionService.modeToOnnxPreference(detectionMode)})`
            : "backend PyTorch-CPU (best.pt)";
      console.log(`Starting detection with ${modeLabel}...`);
      const data = await AnoleDetectionService.detect(selectedFile, detectionMode);
      
      // Transform API response to match our DetectionResult interface
      const result: DetectionResult = {
        totalLizards: data.totalLizards,
        predictions: data.predictions.map((pred: any) => {
          const box = pred.boundingBox ?? pred.box;
          const altConfidences = Array.isArray(pred.alternateConfidences)
            ? (pred.alternateConfidences as any[])
                .filter(
                  (alt) =>
                    alt &&
                    typeof alt.confidence === "number" &&
                    typeof alt.species === "string"
                )
                .map((alt) => ({
                  classIndex: typeof alt.classIndex === "number" ? alt.classIndex : -1,
                  species: alt.species,
                  scientificName: alt.scientificName ?? "",
                  confidence: alt.confidence,
                  relativeConfidence:
                    typeof alt.relativeConfidence === "number"
                      ? alt.relativeConfidence
                      : (() => {
                          const remaining = 1 - (typeof pred.confidence === "number" ? pred.confidence : 0);
                          return remaining > 0 ? (alt.confidence ?? 0) / remaining : 0;
                        })(),
                }))
            : undefined;

          return {
          species: pred.species,
          scientificName: pred.scientificName,
          confidence: pred.confidence,
          count: pred.count,
            box: Array.isArray(box) && box.length === 4 ? [
              box[0],
              box[1],
              box[2],
              box[3]
            ] : undefined,
            altConfidences,
          };
        }),
        imageUrl: previewUrl || undefined,
        clientTimings: data.timings,
      };
      
      // Debug: log bounding boxes
      console.log("Detection result:", result);
      console.log("Boxes:", result.predictions.map(p => p.box));
      console.log("Processing time:", data.processingTime, "seconds");
      
      setDetectionResult(result);
    } catch (error) {
      console.error("Prediction failed:", error);
      alert(`Prediction failed: ${error instanceof Error ? error.message : 'Unknown error'}. ONNX inference failed and backend fallback failed.`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleConnectInaturalist = () => {
    try {
      iNaturalistAPI.connectAccount();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Cannot start iNaturalist login");
    }
  };

  const handleDisconnectInaturalist = async () => {
    try {
      await iNaturalistAPI.disconnect();
      await refreshInatStatus();
      toast.success("Disconnected from iNaturalist");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Disconnect failed");
    }
  };

  const handleUploadObservationToInat = async () => {
    if (!detectionResult || !selectedFile) return;
    if (!inatStatus?.connected) {
      toast.error("Connect to iNaturalist first");
      return;
    }

    const predictions = detectionResult.predictions.filter(
      (p) => p.species !== "Failed" && p.confidence > 0,
    );
    const best =
      predictions.length > 0
        ? [...predictions].sort((a, b) => b.confidence - a.confidence)[0]
        : null;
    if (!best) {
      toast.error("No valid species prediction to upload");
      return;
    }

    setUploadingToiNaturalist(true);
    toast.dismiss();
    const toastId = toast.loading("Uploading observation…");
    try {
      const location = await getCurrentLocation();
      await iNaturalistAPI.uploadObservation({
        species: best.species,
        scientificName: best.scientificName,
        confidence: best.confidence,
        count: best.count ?? 1,
        imageFile: selectedFile,
        location: location ?? undefined,
      });
      toast.dismiss(toastId);
      toast.success("Observation uploaded to iNaturalist");
    } catch (e) {
      toast.dismiss(toastId);
      const msg = e instanceof Error ? e.message : "Upload failed";
      if (msg.includes("401")) {
        toast.error("Session expired. Connect to iNaturalist again.");
        await refreshInatStatus();
      } else {
        toast.error(msg);
      }
    } finally {
      setUploadingToiNaturalist(false);
    }
  };

  const handleSpeciesCorrection = (index: number, newSpeciesName: string) => {
    if (!detectionResult) return;

    const selectedSpecies = FLORIDA_ANOLE_SPECIES.find(s => s.name === newSpeciesName);
    if (!selectedSpecies) return;

    const updatedPredictions = [...detectionResult.predictions];
    updatedPredictions[index] = {
      ...updatedPredictions[index],
      species: selectedSpecies.name,
      scientificName: selectedSpecies.scientific,
      confidence: 1.0, // Manual correction implies 100% confidence
      isManualCorrection: true,
    };

    setDetectionResult({
      ...detectionResult,
      predictions: updatedPredictions,
    });
  };

  const handleBoxResize = useCallback((index: number, newBox: { x: number; y: number; width: number; height: number }) => {
    if (!detectionResult || !selectedFile || !imageDimensions) return;

    // Update the box coordinates in the detection result
    const updatedPredictions = [...detectionResult.predictions];
    const [x1, y1, x2, y2] = [
      newBox.x,
      newBox.y,
      newBox.x + newBox.width,
      newBox.y + newBox.height
    ];
    
    updatedPredictions[index] = {
      ...updatedPredictions[index],
      box: [x1, y1, x2, y2] as [number, number, number, number],
    };

    setDetectionResult({
      ...detectionResult,
      predictions: updatedPredictions,
    });

    // Debounce reclassification - wait 500ms after user stops adjusting
    if (reclassifyTimeoutRef.current) {
      clearTimeout(reclassifyTimeoutRef.current);
    }

    reclassifyTimeoutRef.current = setTimeout(async () => {
      try {
        setReclassifyingIndex(index);
        console.log(`Reclassifying box ${index} with new coordinates: [${x1}, ${y1}, ${x2}, ${y2}]`);
        
        const classification = await AnoleDetectionService.classifyRegion(
          selectedFile,
          [x1, y1, x2, y2],
          detectionMode
        );

        // Update the prediction with new classification
        const finalPredictions = [...detectionResult.predictions];
        finalPredictions[index] = {
          species: classification.species,
          scientificName: classification.scientificName,
          confidence: classification.confidence,
          count: 1,
          box: [x1, y1, x2, y2] as [number, number, number, number],
          altConfidences: classification.alternateConfidences,
        };

        setDetectionResult({
          ...detectionResult,
          predictions: finalPredictions,
        });

        console.log(`Reclassification complete: ${classification.species} (${(classification.confidence * 100).toFixed(1)}%)`);
      } catch (error) {
        console.error(`Failed to reclassify box ${index}:`, error);
        
        // Update the prediction with "Failed" label
        const finalPredictions = [...detectionResult.predictions];
        finalPredictions[index] = {
          species: "Failed",
          scientificName: "Unknown",
          confidence: 0,
          count: 1,
          box: [x1, y1, x2, y2] as [number, number, number, number],
          altConfidences: undefined,
        };

        setDetectionResult({
          ...detectionResult,
          predictions: finalPredictions,
        });
      } finally {
        setReclassifyingIndex(null);
      }
    }, 500);
  }, [detectionResult, selectedFile, imageDimensions, detectionMode]);

  // Convert display coordinates to natural image coordinates
  const displayToNatural = useCallback((displayX: number, displayY: number): [number, number] => {
    if (!imageDimensions || !imageRef.current) return [0, 0];
    const scaleX = imageDimensions.width / imageRef.current.clientWidth;
    const scaleY = imageDimensions.height / imageRef.current.clientHeight;
    return [displayX * scaleX, displayY * scaleY];
  }, [imageDimensions]);

  // Convert natural image coordinates to display coordinates
  const naturalToDisplay = useCallback((naturalX: number, naturalY: number): [number, number] => {
    if (!imageDimensions || !imageRef.current) return [0, 0];
    const scaleX = imageRef.current.clientWidth / imageDimensions.width;
    const scaleY = imageRef.current.clientHeight / imageDimensions.height;
    return [naturalX * scaleX, naturalY * scaleY];
  }, [imageDimensions]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!isDrawing || !imageContainerRef.current || !imageRef.current || !imageDimensions) return;
    
    e.preventDefault();
    const rect = imageContainerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Check if click is within image bounds
    if (x < 0 || y < 0 || x > imageRef.current.clientWidth || y > imageRef.current.clientHeight) {
      return;
    }
    
    const [naturalX, naturalY] = displayToNatural(x, y);
    setDrawingBox({ startX: naturalX, startY: naturalY, endX: naturalX, endY: naturalY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing || !drawingBox || !imageContainerRef.current || !imageRef.current) return;
    
    e.preventDefault();
    const rect = imageContainerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Clamp to image bounds
    const clampedX = Math.max(0, Math.min(x, imageRef.current.clientWidth));
    const clampedY = Math.max(0, Math.min(y, imageRef.current.clientHeight));
    
    const [naturalX, naturalY] = displayToNatural(clampedX, clampedY);
    setDrawingBox({ ...drawingBox, endX: naturalX, endY: naturalY });
  };

  const handleMouseUp = async () => {
    if (!isDrawing || !drawingBox || !selectedFile || !imageDimensions) return;
    
    // Calculate box coordinates
    const x1 = Math.min(drawingBox.startX, drawingBox.endX);
    const y1 = Math.min(drawingBox.startY, drawingBox.endY);
    const x2 = Math.max(drawingBox.startX, drawingBox.endX);
    const y2 = Math.max(drawingBox.startY, drawingBox.endY);
    
    // Ensure minimum box size
    const minSize = 20;
    const width = Math.max(minSize, x2 - x1);
    const height = Math.max(minSize, y2 - y1);
    
    // Clamp to image bounds
    const finalX1 = Math.max(0, Math.min(x1, imageDimensions.width - width));
    const finalY1 = Math.max(0, Math.min(y1, imageDimensions.height - height));
    const finalX2 = Math.min(imageDimensions.width, finalX1 + width);
    const finalY2 = Math.min(imageDimensions.height, finalY1 + height);
    
    const finalBox: [number, number, number, number] = [finalX1, finalY1, finalX2, finalY2];
    
    // Clear drawing state
    setDrawingBox(null);
    setIsDrawing(false);
    
    // Classify the drawn region
    try {
      setIsClassifyingDrawnBox(true);
      console.log(`Classifying drawn box: [${finalX1}, ${finalY1}, ${finalX2}, ${finalY2}]`);
      
      const classification = await AnoleDetectionService.classifyRegion(
        selectedFile,
        finalBox,
        detectionMode
      );
      
      // Add to detection results
      const newPrediction: PredictionResult = {
        species: classification.species,
        scientificName: classification.scientificName,
        confidence: classification.confidence,
        count: 1,
        box: finalBox,
        altConfidences: classification.alternateConfidences,
      };
      
      if (detectionResult) {
        setDetectionResult({
          ...detectionResult,
          totalLizards: detectionResult.totalLizards + 1,
          predictions: [...detectionResult.predictions, newPrediction],
        });
      } else {
        setDetectionResult({
          totalLizards: 1,
          predictions: [newPrediction],
        });
      }
      
      console.log(`Classification complete: ${classification.species} (${(classification.confidence * 100).toFixed(1)}%)`);
    } catch (error) {
      console.error('Failed to classify drawn box:', error);
      
      // Add a "Failed" prediction to results
      const failedPrediction: PredictionResult = {
        species: "Failed",
        scientificName: "Unknown",
        confidence: 0,
        count: 1,
        box: finalBox,
        altConfidences: undefined,
      };
      
      if (detectionResult) {
        setDetectionResult({
          ...detectionResult,
          totalLizards: detectionResult.totalLizards + 1,
          predictions: [...detectionResult.predictions, failedPrediction],
        });
      } else {
        setDetectionResult({
          totalLizards: 1,
          predictions: [failedPrediction],
        });
      }
    } finally {
      setIsClassifyingDrawnBox(false);
    }
  };

  return (
    <div className="container" style={{ textAlign: "center" }}>
      <h1 style={{ color: "#2E7D32", fontSize: "2.5rem", marginBottom: "0.5rem" }}>🦎 Lizard Lens</h1>
      
      {/* Back to home + inference location */}
      <div
        style={{
          marginBottom: "1.5rem",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "1rem",
        }}
      >
        <Link
          to="/"
          style={{
            color: "#2E7D32",
            textDecoration: "none",
            fontSize: "14px",
            fontWeight: "500",
          }}
        >
          ← Back to Home
        </Link>

        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "center",
            gap: "0.75rem",
            maxWidth: "min(100%, 520px)",
          }}
        >
          <label
            htmlFor="inference-location"
            style={{ fontSize: "14px", fontWeight: 600, color: "#333" }}
          >
            Inference location
          </label>
          <select
            id="inference-location"
            value={normalizeInferenceModeForSelect(detectionMode)}
            onChange={(e) =>
              handleInferenceLocationChange(e.target.value as DetectionMode)
            }
            disabled={
              isLoading ||
              reclassifyingIndex !== null ||
              isDrawing ||
              isClassifyingDrawnBox
            }
            style={{
              minWidth: "260px",
              maxWidth: "100%",
              padding: "8px 12px",
              fontSize: "14px",
              borderRadius: "8px",
              border: "1px solid #ced4da",
              backgroundColor: "#fff",
              color: "#212529",
              cursor:
                isLoading ||
                reclassifyingIndex !== null ||
                isDrawing ||
                isClassifyingDrawnBox
                  ? "not-allowed"
                  : "pointer",
            }}
          >
            {INFERENCE_LOCATION_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {(detectionMode === "auto" || AnoleDetectionService.isOnnxFrontendMode(detectionMode)) &&
          detectionResult?.clientTimings?.executionProvider && (
            <p style={{ margin: 0, fontSize: "12px", color: "#666" }}>
              Last run execution provider:{" "}
              <strong>{detectionResult.clientTimings.executionProvider}</strong>
              {" · "}
              preference{" "}
              <strong>{detectionResult.clientTimings.onnxPreference}</strong>
            </p>
          )}
      </div>

      {/* Species Information */}
      <div className="species-info">
        <h3 style={{ margin: "0 0 0.5rem 0", fontSize: "1.3rem" }}>Florida Anole Species</h3>
        <p style={{ margin: 0, fontSize: "1rem", lineHeight: "1.5" }}>
          This app can identify 5 species: Green Anole, Brown Anole, Crested Anole, Knight Anole, and Bark Anole
        </p>
      </div>

      {/* File upload section - redesigned */}
      <div style={{ 
        marginBottom: "2rem",
        maxWidth: "600px",
        margin: "0 auto 2rem auto",
        padding: "0 1rem"
      }}>
        {/* Upload card */}
        <div style={{
          backgroundColor: "#f8f9fa",
          padding: "1.5rem",
          borderRadius: "12px",
          border: previewUrl ? "2px solid #4CAF50" : "2px dashed #4CAF50",
          boxShadow: previewUrl ? "0 4px 12px rgba(76, 175, 80, 0.2)" : "none",
          transition: "all 0.3s ease"
        }}>
          {!previewUrl ? (
            /* Upload area - no image selected */
            <div style={{ textAlign: "center" }}>
              <label
                htmlFor="file-upload"
                style={{
                  display: "block",
                  cursor: "pointer",
                  padding: "2rem 1rem",
                  borderRadius: "8px",
                  backgroundColor: "#E8F5E9",
                  border: "2px dashed #4CAF50",
                  transition: "all 0.2s ease"
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = "#C8E6C9";
                  e.currentTarget.style.transform = "scale(1.02)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "#E8F5E9";
                  e.currentTarget.style.transform = "scale(1)";
                }}
              >
                <div style={{ fontSize: "3rem", marginBottom: "0.5rem" }}>📸</div>
                <div style={{ 
                  fontSize: "1.1rem", 
                  fontWeight: "600", 
                  color: "#2E7D32",
                  marginBottom: "0.5rem"
                }}>
                  Choose an Image
                </div>
                <div style={{ 
                  fontSize: "0.9rem", 
                  color: "#666",
                  marginBottom: "1rem"
                }}>
                  Tap to select a photo from your device
                </div>
                <input
                  id="file-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  style={{ display: "none" }}
                />
              </label>
            </div>
          ) : (
            /* Image preview area */
            <div style={{ textAlign: "center" }}>
              {isDrawing && (
                <div style={{
                  marginBottom: "0.5rem",
                  padding: "8px 16px",
                  backgroundColor: "#E3F2FD",
                  color: "#1976D2",
                  borderRadius: "8px",
                  fontSize: "14px",
                  fontWeight: "500",
                  textAlign: "center"
                }}>
                  ✏️ Drawing mode: Click and drag on the image to draw a bounding box
                </div>
              )}
              <div 
                ref={imageContainerRef}
                style={{ 
                  position: "relative", 
                  display: "inline-block",
                  marginBottom: "1rem",
                  cursor: isDrawing ? "crosshair" : "default",
                  userSelect: isDrawing ? "none" : "auto"
                }}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={() => {
                  // Cancel drawing if mouse leaves
                  if (isDrawing && drawingBox) {
                    setDrawingBox(null);
                    setIsDrawing(false);
                  }
                }}
              >
                <img 
                  ref={imageRef}
                  src={previewUrl} 
                  alt="Preview" 
                  className="image-preview"
                  style={{ 
                    maxWidth: "100%",
                    width: "auto",
                    height: "auto",
                    maxHeight: "400px",
                    borderRadius: "8px",
                    border: "2px solid #4CAF50",
                    display: "block",
                    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.15)",
                    pointerEvents: isDrawing ? "none" : "auto"
                  }} 
                  onLoad={(e) => {
                    const img = e.currentTarget;
                    setImageDimensions({
                      width: img.naturalWidth,
                      height: img.naturalHeight
                    });
                  }}
                />
                {/* Drawing box overlay */}
                {isDrawing && drawingBox && imageRef.current && imageDimensions && (() => {
                  const [displayX1, displayY1] = naturalToDisplay(drawingBox.startX, drawingBox.startY);
                  const [displayX2, displayY2] = naturalToDisplay(drawingBox.endX, drawingBox.endY);
                  const left = Math.min(displayX1, displayX2);
                  const top = Math.min(displayY1, displayY2);
                  const width = Math.abs(displayX2 - displayX1);
                  const height = Math.abs(displayY2 - displayY1);
                  
                  return (
                    <div
                      style={{
                        position: "absolute",
                        left: `${left}px`,
                        top: `${top}px`,
                        width: `${width}px`,
                        height: `${height}px`,
                        border: "2px dashed #2196F3",
                        backgroundColor: "rgba(33, 150, 243, 0.1)",
                        borderRadius: "4px",
                        pointerEvents: "none",
                        zIndex: 20
                      }}
                    />
                  );
                })()}
                {/* Draw resizable bounding boxes overlay */}
                {(() => {
                  if (!detectionResult || !detectionResult.predictions || !imageDimensions || !imageRef.current) {
                    return null;
                  }
                  
                  const boxesWithCoords = detectionResult.predictions.filter(p => p.box && Array.isArray(p.box) && p.box.length === 4);
                  
                  if (boxesWithCoords.length === 0) {
                    return null;
                  }
                  
                  const displayWidth = imageRef.current.clientWidth;
                  const displayHeight = imageRef.current.clientHeight;
                  
                  return (
                    <div 
                      style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        width: `${displayWidth}px`,
                        height: `${displayHeight}px`,
                        pointerEvents: isDrawing ? "none" : "auto",
                        overflow: "visible"
                      }}
                    >
                      {boxesWithCoords.map((prediction, idx) => {
                        if (!prediction.box || prediction.box.length !== 4) return null;
                        
                        // Extract bounding box coordinates [x1, y1, x2, y2] in natural image coordinates
                        const [x1, y1, x2, y2] = prediction.box;
                        const boxWidth = x2 - x1;
                        const boxHeight = y2 - y1;
                        
                        // Color based on confidence (green=high, yellow=medium, red=low, gray for failed)
                        const isFailed = prediction.species === "Failed" || prediction.confidence === 0;
                        const color = isFailed ? "#6c757d" : prediction.confidence > 0.8 ? "#28a745" : prediction.confidence > 0.6 ? "#ffc107" : "#dc3545";
                        
                        const label = isFailed 
                          ? `Failed (0.00%)${reclassifyingIndex === idx ? ' 🔄' : ''}`
                          : `${prediction.species} (${(prediction.confidence * 100).toFixed(1)}%)${reclassifyingIndex === idx ? ' 🔄' : ''}`;
                        
                        return (
                          <ResizableBoundingBox
                            key={idx}
                            x={x1}
                            y={y1}
                            width={boxWidth}
                            height={boxHeight}
                            color={color}
                            label={label}
                            onResize={(newBox) => handleBoxResize(idx, newBox)}
                            imageNaturalWidth={imageDimensions.width}
                            imageNaturalHeight={imageDimensions.height}
                            imageDisplayWidth={displayWidth}
                            imageDisplayHeight={displayHeight}
                            disabled={isLoading || reclassifyingIndex !== null || isDrawing}
                          />
                        );
                      })}
                    </div>
                  );
                })()}
              </div>
              
              {/* Action buttons */}
              <div 
                className="upload-actions"
                style={{ 
                  display: "flex", 
                  gap: "0.75rem", 
                  justifyContent: "center",
                  flexWrap: "wrap"
                }}
              >
                <label
                  htmlFor="file-upload"
                  style={{
                    display: "inline-block",
                    padding: "10px 20px",
                    backgroundColor: "#E8F5E9",
                    color: "#2E7D32",
                    border: "2px solid #4CAF50",
                    borderRadius: "8px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                    textAlign: "center"
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = "#C8E6C9";
                    e.currentTarget.style.transform = "translateY(-1px)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "#E8F5E9";
                    e.currentTarget.style.transform = "translateY(0)";
                  }}
                >
                  Change Image
                </label>
                <input
                  id="file-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  style={{ display: "none" }}
                />
                
                <button
                  onClick={handlePredict}
                  disabled={!selectedFile || isLoading || isDrawing}
                  className="button"
                  style={{
                    padding: "10px 24px",
                    backgroundColor: selectedFile && !isLoading && !isDrawing ? "#4CAF50" : "#9e9e9e",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    fontSize: "16px",
                    fontWeight: "600",
                    cursor: selectedFile && !isLoading && !isDrawing ? "pointer" : "not-allowed",
                    transition: "all 0.2s ease",
                    boxShadow: selectedFile && !isLoading && !isDrawing ? "0 4px 12px rgba(76, 175, 80, 0.3)" : "none",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "0.5rem"
                  }}
                  onMouseEnter={(e) => {
                    if (selectedFile && !isLoading && !isDrawing) {
                      e.currentTarget.style.transform = "translateY(-2px)";
                      e.currentTarget.style.boxShadow = "0 6px 16px rgba(76, 175, 80, 0.4)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (selectedFile && !isLoading && !isDrawing) {
                      e.currentTarget.style.transform = "translateY(0)";
                      e.currentTarget.style.boxShadow = "0 4px 12px rgba(76, 175, 80, 0.3)";
                    }
                  }}
                >
                  {isLoading ? (
                    <>
                      <div style={{
                        width: "16px",
                        height: "16px",
                        border: "2px solid #ffffff",
                        borderTop: "2px solid transparent",
                        borderRadius: "50%",
                        animation: "spin 1s linear infinite",
                        display: "inline-block"
                      }} />
                      Classifying...
                    </>
                  ) : (
                    <>🔍 Classify Species</>
                  )}
                </button>

                {previewUrl && imageDimensions && (
                  <button
                    onClick={() => {
                      if (isDrawing) {
                        setDrawingBox(null);
                      }
                      setIsDrawing(!isDrawing);
                    }}
                    disabled={isLoading || isClassifyingDrawnBox}
                    style={{
                      padding: "10px 20px",
                      backgroundColor: isDrawing ? "#2196F3" : "#E3F2FD",
                      color: isDrawing ? "white" : "#1976D2",
                      border: `2px solid ${isDrawing ? "#1976D2" : "#2196F3"}`,
                      borderRadius: "8px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: isLoading || isClassifyingDrawnBox ? "not-allowed" : "pointer",
                      transition: "all 0.2s ease",
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "0.5rem"
                    }}
                    onMouseEnter={(e) => {
                      if (!isLoading && !isClassifyingDrawnBox) {
                        e.currentTarget.style.transform = "translateY(-1px)";
                        e.currentTarget.style.boxShadow = "0 4px 8px rgba(33, 150, 243, 0.3)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isLoading && !isClassifyingDrawnBox) {
                        e.currentTarget.style.transform = "translateY(0)";
                        e.currentTarget.style.boxShadow = "none";
                      }
                    }}
                  >
                    {isClassifyingDrawnBox ? (
                      <>
                        <div style={{
                          width: "14px",
                          height: "14px",
                          border: "2px solid currentColor",
                          borderTop: "2px solid transparent",
                          borderRadius: "50%",
                          animation: "spin 1s linear infinite",
                          display: "inline-block"
                        }} />
                        Classifying...
                      </>
                    ) : isDrawing ? (
                      <>✏️ Cancel Drawing</>
                    ) : (
                      <>✏️ Draw Box</>
                    )}
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Detection Results */}
      {detectionResult && (
        <div className="result-card" style={{
          backgroundColor: "#f8f9fa",
          padding: "1.5rem",
          borderRadius: "12px",
          border: "2px solid #4CAF50",
          maxWidth: "600px",
          margin: "0 auto",
          textAlign: "left",
          boxShadow: "0 4px 12px rgba(76, 175, 80, 0.2)"
        }}>
          <h3 style={{ textAlign: "center", marginBottom: "1rem", color: "#2E7D32", fontSize: "1.5rem" }}>
            Detection Results
          </h3>
          
          <div style={{ marginBottom: "1rem", textAlign: "center" }}>
            <strong>Total Lizards Detected: {detectionResult.totalLizards}</strong>
          </div>

          {detectionResult.clientTimings && (
            <details
              style={{
                marginBottom: "1rem",
                fontSize: "0.85rem",
                textAlign: "left",
                backgroundColor: "#fff",
                padding: "0.75rem",
                borderRadius: "8px",
                border: "1px solid #dee2e6",
              }}
            >
              <summary style={{ cursor: "pointer", fontWeight: 600 }}>
                Inference timings (client ONNX)
              </summary>
              <pre
                style={{
                  marginTop: "0.5rem",
                  overflow: "auto",
                  maxHeight: "240px",
                  fontSize: "0.75rem",
                }}
              >
                {JSON.stringify(detectionResult.clientTimings, null, 2)}
              </pre>
            </details>
          )}
          
          {detectionResult.predictions.map((prediction, index) => (
            <div key={index} className="species-card" style={{
              backgroundColor: "white",
              padding: "1rem",
              marginBottom: "0.5rem",
              borderRadius: "8px",
              border: "2px solid #E8F5E9",
              boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem", flexWrap: "wrap", gap: "0.5rem" }}>
                <div style={{ flex: "1", minWidth: "200px", display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
                  <h4 style={{ margin: 0, color: "#495057" }}>
                    {prediction.species === "Failed" ? "Failed" : `${prediction.species} (${prediction.scientificName})`}
                  </h4>
                  <select
                    value=""
                    onChange={(e) => {
                      if (e.target.value) {
                        handleSpeciesCorrection(index, e.target.value);
                      }
                    }}
                    style={{
                      padding: "4px 8px",
                      borderRadius: "4px",
                      border: "1px solid #ced4da",
                      fontSize: "0.85rem",
                      backgroundColor: "white",
                      cursor: "pointer",
                      color: "#495057"
                    }}
                  >
                    <option value="" disabled>Correct...</option>
                    {FLORIDA_ANOLE_SPECIES.map((s) => (
                      <option key={s.name} value={s.name}>
                        {s.name}
                      </option>
                    ))}
                  </select>
                </div>
                <span className={`confidence-${prediction.confidence > 0.8 ? 'high' : prediction.confidence > 0.6 ? 'medium' : 'low'}`} style={{ 
                  padding: "0.25rem 0.5rem",
                  borderRadius: "4px",
                  fontSize: "0.875rem",
                  fontWeight: "bold",
                  whiteSpace: "nowrap",
                  backgroundColor: prediction.species === "Failed" ? "#6c757d" : prediction.isManualCorrection ? "#1976D2" : undefined,
                  color: prediction.species === "Failed" || prediction.isManualCorrection ? "white" : undefined
                }}>
                  {prediction.species === "Failed" ? "0.00% confidence" :
                    prediction.isManualCorrection ? "100% (User Corrected)" :
                      `${(prediction.confidence * 100).toFixed(1)}% confidence`}
                </span>
              </div>
              
              <div style={{ marginBottom: "0.5rem" }}>
                <div className="progress-bar">
                  <div 
                    className={`progress-fill ${prediction.confidence > 0.8 ? 'high' : prediction.confidence > 0.6 ? 'medium' : 'low'}`}
                    style={{
                      width: `${prediction.confidence * 100}%`,
                      height: "100%"
                    }} 
                  />
                </div>
              </div>
              
              {prediction.altConfidences && prediction.altConfidences.length > 0 && prediction.confidence < 0.999 && (
                <div style={{ marginBottom: "0.5rem", fontSize: "0.85rem", color: "#495057" }}>
                  <strong>Other possibilities:&nbsp;</strong>
                  {prediction.altConfidences
                    .slice(0, 3)
                    .map((alt) => {
                      const overallPercent = Math.max(alt.confidence * 100, 0);
                      return `${overallPercent.toFixed(1)}% ${alt.species}`;
                    })
                    .join(", ")}
                </div>
              )}
              
              <p style={{ margin: 0, fontSize: "0.9rem", color: "#6c757d" }}>
                Count: {prediction.count} individual{prediction.count > 1 ? 's' : ''}
              </p>
            </div>
          ))}
          
          {/* iNaturalist: OAuth on API + upload */}
          <div
            style={{
              marginTop: "1rem",
              padding: "1rem",
              borderRadius: "8px",
              border: "1px solid #bee5eb",
              backgroundColor: "#f1fbfd",
              textAlign: "center",
            }}
          >
            <p style={{ margin: "0 0 0.5rem 0", color: "#0c5460", fontWeight: 600 }}>
              Share on iNaturalist
            </p>
            {!iNaturalistAPI.isBackendConfigured() ? (
              <p style={{ margin: "0 0 0.75rem 0", fontSize: "0.9rem", color: "#0c5460" }}>
                Set <code style={{ fontSize: "0.85rem" }}>VITE_API_BASE_URL</code> to your API
                origin so OAuth and uploads can use your backend session.
              </p>
            ) : (
              <>
                <p style={{ margin: "0 0 0.75rem 0", fontSize: "0.9rem", color: "#0c5460" }}>
                  {inatStatusLoading
                    ? "Checking iNaturalist connection…"
                    : inatStatus?.connected
                      ? "You are connected. Upload uses your session on the server (no token in the browser)."
                      : "Connect once with iNaturalist, then upload this image as an observation."}
                </p>
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "0.5rem",
                    justifyContent: "center",
                    alignItems: "center",
                    marginBottom: "0.5rem",
                  }}
                >
                  {!inatStatusLoading && !inatStatus?.connected && (
                    <button
                      type="button"
                      onClick={handleConnectInaturalist}
                      disabled={uploadingToiNaturalist}
                      className="inaturalist-button"
                    >
                      Connect iNaturalist
                    </button>
                  )}
                  {!inatStatusLoading && inatStatus?.connected && (
                    <>
                      <span
                        style={{
                          fontSize: "0.85rem",
                          color: "#155724",
                          fontWeight: 600,
                          padding: "4px 10px",
                          background: "#d4edda",
                          borderRadius: "6px",
                        }}
                      >
                        Connected
                      </span>
                      <button
                        type="button"
                        onClick={handleDisconnectInaturalist}
                        disabled={uploadingToiNaturalist}
                        style={{
                          padding: "8px 14px",
                          borderRadius: "6px",
                          border: "1px solid #6c757d",
                          background: "#fff",
                          color: "#495057",
                          cursor: uploadingToiNaturalist ? "not-allowed" : "pointer",
                          fontSize: "14px",
                        }}
                      >
                        Disconnect
                      </button>
                    </>
                  )}
                </div>
                <button
                  type="button"
                  onClick={handleUploadObservationToInat}
                  disabled={
                    uploadingToiNaturalist ||
                    inatStatusLoading ||
                    !inatStatus?.connected
                  }
                  className="inaturalist-button"
                >
                  {uploadingToiNaturalist ? (
                    <>
                      <div
                        style={{
                          width: "16px",
                          height: "16px",
                          border: "2px solid #ffffff",
                          borderTop: "2px solid transparent",
                          borderRadius: "50%",
                          animation: "spin 1s linear infinite",
                        }}
                      />
                      Uploading…
                    </>
                  ) : (
                    <>Upload observation</>
                  )}
                </button>
                <p style={{ margin: "0.75rem 0 0 0", fontSize: "0.8rem", color: "#6c757d" }}>
                  <a
                    href="https://www.inaturalist.org/observations/upload"
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: "#138496" }}
                  >
                    Open iNaturalist upload page
                  </a>{" "}
                  if you prefer to post manually.
                </p>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}