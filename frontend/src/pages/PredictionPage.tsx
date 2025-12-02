import { useState, useRef, useEffect, useCallback } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { iNaturalistAPI, getCurrentLocation } from "../services/iNaturalistService";
import type { iNaturalistObservation } from "../services/iNaturalistService";
import { AnoleDetectionService } from "../services/AnoleDetectionService";
import type { DetectionMode } from "../services/AnoleDetectionService";
import { ResizableBoundingBox } from "../components/ResizableBoundingBox";

// Define the 5 Florida anole species (for reference)
// const FLORIDA_ANOLE_SPECIES = [
//   { name: "Green Anole", scientific: "Anolis carolinensis", common: "American Green Anole" },
//   { name: "Brown Anole", scientific: "Anolis sagrei", common: "Cuban Brown Anole" },
//   { name: "Crested Anole", scientific: "Anolis cristatellus", common: "Puerto Rican Crested Anole" },
//   { name: "Knight Anole", scientific: "Anolis equestris", common: "Cuban Knight Anole" },
//   { name: "Bark Anole", scientific: "Anolis distichus", common: "Bark Anole" }
// ];

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
}

interface DetectionResult {
  totalLizards: number;
  predictions: PredictionResult[];
  imageUrl?: string;
}

export default function PredictionPage() {
  const [searchParams] = useSearchParams();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadingToiNaturalist, setUploadingToiNaturalist] = useState(false);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const [detectionMode, setDetectionMode] = useState<DetectionMode>('backend');
  const [reclassifyingIndex, setReclassifyingIndex] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawingBox, setDrawingBox] = useState<{ startX: number; startY: number; endX: number; endY: number } | null>(null);
  const [isClassifyingDrawnBox, setIsClassifyingDrawnBox] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const reclassifyTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Check for gpu query parameter
  useEffect(() => {
    const gpuParam = searchParams.get('gpu');
    if (gpuParam === 'client-side') {
      setDetectionMode('onnx-frontend');
      console.log('🖥️ Client-side ONNX mode enabled via ?gpu=client-side');
    } else if (gpuParam === 'server') {
      setDetectionMode('backend-pytorch');
      console.log('🎮 Backend PyTorch mode enabled via ?gpu=server');
    } else {
      setDetectionMode('backend');
      console.log('☁️ Backend PyTorch-CPU mode (default, uses best.pt)');
    }
  }, [searchParams]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (reclassifyTimeoutRef.current) {
        clearTimeout(reclassifyTimeoutRef.current);
      }
    };
  }, []);

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
      const modeLabel = detectionMode === 'onnx-frontend' 
        ? 'client-side ONNX' 
        : detectionMode === 'backend-pytorch'
        ? 'backend PyTorch'
        : 'backend PyTorch-CPU (best.pt)';
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
        imageUrl: previewUrl || undefined
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

  const handleUploadToiNaturalist = async () => {
    if (!detectionResult || !selectedFile) return;
    
    setUploadingToiNaturalist(true);
    
    try {
      // Get current location if available
      const location = await getCurrentLocation();
      
      // Create observations for each detected species
      const uploadPromises = detectionResult.predictions.map(async (prediction) => {
        const observation: iNaturalistObservation = {
          species: prediction.species,
          scientificName: prediction.scientificName,
          confidence: prediction.confidence,
          count: prediction.count,
          imageFile: selectedFile,
          location: location || undefined,
          notes: `Detected by AI`
        };
        
        return iNaturalistAPI.uploadObservation(observation);
      });
      
      await Promise.all(uploadPromises);
      
      alert(`Successfully uploaded ${detectionResult.predictions.length} observation(s) to iNaturalist! Your contributions to citizen science are appreciated.`);
    } catch (error) {
      console.error("iNaturalist upload failed:", error);
      alert("Failed to upload to iNaturalist. Please try again.");
    } finally {
      setUploadingToiNaturalist(false);
    }
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
      <h1 style={{ color: "#2E7D32", fontSize: "2.5rem", marginBottom: "0.5rem" }}>🦎 Anole Species Classification</h1>
      
      {/* Back to home link */}
      <div style={{ marginBottom: "2rem", display: "flex", alignItems: "center", gap: "1rem", justifyContent: "center" }}>
        <Link 
          to="/" 
          style={{ 
            color: "#2E7D32", 
            textDecoration: "none",
            fontSize: "14px",
            fontWeight: "500"
          }}
        >
          ← Back to Home
        </Link>
        
        {/* Detection mode indicator */}
        <span style={{
          padding: "4px 12px",
          borderRadius: "12px",
          fontSize: "12px",
          fontWeight: "500",
          backgroundColor: detectionMode === 'onnx-frontend' ? '#e3f2fd' : detectionMode === 'backend-pytorch' ? '#e8f5e9' : '#f5f5f5',
          color: detectionMode === 'onnx-frontend' ? '#1976d2' : detectionMode === 'backend-pytorch' ? '#2e7d32' : '#666',
          border: `1px solid ${detectionMode === 'onnx-frontend' ? '#90caf9' : detectionMode === 'backend-pytorch' ? '#66bb6a' : '#ddd'}`
        }}>
          {detectionMode === 'onnx-frontend' ? '🖥️ Client-side ONNX' : detectionMode === 'backend-pytorch' ? '🎮 Backend PyTorch' : '☁️ PyTorch CPU (best.pt)'}
        </span>
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
                <h4 style={{ margin: 0, color: "#495057", flex: "1", minWidth: "200px" }}>
                  {prediction.species === "Failed" ? "Failed" : `${prediction.species} (${prediction.scientificName})`}
                </h4>
                <span className={`confidence-${prediction.confidence > 0.8 ? 'high' : prediction.confidence > 0.6 ? 'medium' : 'low'}`} style={{ 
                  padding: "0.25rem 0.5rem",
                  borderRadius: "4px",
                  fontSize: "0.875rem",
                  fontWeight: "bold",
                  whiteSpace: "nowrap",
                  backgroundColor: prediction.species === "Failed" ? "#6c757d" : undefined,
                  color: prediction.species === "Failed" ? "white" : undefined
                }}>
                  {prediction.species === "Failed" ? "0.00% confidence" : `${(prediction.confidence * 100).toFixed(1)}% confidence`}
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
          
          {/* iNaturalist Upload Button */}
          <div style={{ textAlign: "center", marginTop: "1rem" }}>
            <button
              onClick={handleUploadToiNaturalist}
              disabled={uploadingToiNaturalist}
              className="inaturalist-button"
            >
              {uploadingToiNaturalist ? (
                <>
                  <div style={{
                    width: "16px",
                    height: "16px",
                    border: "2px solid #ffffff",
                    borderTop: "2px solid transparent",
                    borderRadius: "50%",
                    animation: "spin 1s linear infinite"
                  }} />
                  Uploading...
                </>
              ) : (
                <>
                  Upload to iNaturalist
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}