import { useState, useRef, useEffect } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { iNaturalistAPI, getCurrentLocation } from "../services/iNaturalistService";
import type { iNaturalistObservation } from "../services/iNaturalistService";
import { AnoleDetectionService } from "../services/AnoleDetectionService";
import type { DetectionMode } from "../services/AnoleDetectionService";

// Define the 5 Florida anole species (for reference)
// const FLORIDA_ANOLE_SPECIES = [
//   { name: "Green Anole", scientific: "Anolis carolinensis", common: "American Green Anole" },
//   { name: "Brown Anole", scientific: "Anolis sagrei", common: "Cuban Brown Anole" },
//   { name: "Crested Anole", scientific: "Anolis cristatellus", common: "Puerto Rican Crested Anole" },
//   { name: "Knight Anole", scientific: "Anolis equestris", common: "Cuban Knight Anole" },
//   { name: "Bark Anole", scientific: "Anolis distichus", common: "Bark Anole" }
// ];

interface PredictionResult {
  species: string;
  scientificName: string;
  confidence: number;
  count: number;
  box?: [number, number, number, number]; // [x1, y1, x2, y2] bounding box coordinates
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
  const imageRef = useRef<HTMLImageElement>(null);

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
        predictions: data.predictions.map((pred) => ({
          species: pred.species,
          scientificName: pred.scientificName,
          confidence: pred.confidence,
          count: pred.count,
          box: pred.boundingBox ? [
            pred.boundingBox[0],
            pred.boundingBox[1],
            pred.boundingBox[2],
            pred.boundingBox[3]
          ] : undefined
        })),
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

  return (
    <div className="container" style={{ textAlign: "center" }}>
      <h1>Anole Species Classification</h1>
      
      {/* Back to home link */}
      <div style={{ marginBottom: "2rem", display: "flex", alignItems: "center", gap: "1rem", justifyContent: "center" }}>
        <Link 
          to="/" 
          style={{ 
            color: "#007bff", 
            textDecoration: "none",
            fontSize: "14px"
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
        <h3 style={{ margin: "0 0 0.5rem 0" }}>Florida Anole Species</h3>
        <p style={{ margin: 0, fontSize: "0.9rem" }}>
          This app can identify 5 species: Green Anole, Brown Anole, Crested Anole, Knight Anole, and Bark Anole
        </p>
      </div>

      {/* File upload section */}
      <div style={{ marginBottom: "2rem" }}>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="upload-input"
          style={{
            marginBottom: "1rem",
            width: "100%",
            maxWidth: "400px"
          }}
        />
        
        {previewUrl && (
          <div style={{ marginBottom: "1rem", position: "relative", display: "inline-block" }}>
            <img 
              ref={imageRef}
              src={previewUrl} 
              alt="Preview" 
              className="image-preview"
              style={{ 
                maxWidth: "300px", 
                height: "auto", 
                borderRadius: "6px",
                border: "2px solid #ddd",
                display: "block"
              }} 
              onLoad={(e) => {
                const img = e.currentTarget;
                setImageDimensions({
                  width: img.naturalWidth,
                  height: img.naturalHeight
                });
              }}
            />
            {/* Draw bounding boxes overlay - matching pipeline_evaluation.py coordinate system */}
            {(() => {
              if (!detectionResult || !detectionResult.predictions || !imageDimensions || !imageRef.current) {
                return null;
              }
              
              const boxesWithCoords = detectionResult.predictions.filter(p => p.box && Array.isArray(p.box) && p.box.length === 4);
              console.log("Rendering boxes:", boxesWithCoords.length, "boxes found");
              console.log("Image dimensions:", imageDimensions);
              console.log("Display dimensions:", imageRef.current.clientWidth, imageRef.current.clientHeight);
              
              if (boxesWithCoords.length === 0) {
                return null;
              }
              
              return (
                <div 
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: `${imageRef.current.clientWidth}px`,
                    height: `${imageRef.current.clientHeight}px`,
                    pointerEvents: "none",
                    overflow: "visible"
                  }}
                >
                  {boxesWithCoords.map((prediction, idx) => {
                    if (!prediction.box || prediction.box.length !== 4 || !imageRef.current) return null;
                    
                    // Get actual image dimensions (what YOLO used)
                    const imgNaturalWidth = imageDimensions.width;
                    const imgNaturalHeight = imageDimensions.height;
                    
                    // Get displayed image dimensions
                    const displayWidth = imageRef.current.clientWidth;
                    const displayHeight = imageRef.current.clientHeight;
                    
                    // Calculate scale factors (matching pipeline_evaluation.py: boxes are in absolute pixels)
                    // YOLO returns coordinates in original image pixel space
                    const scaleX = displayWidth / imgNaturalWidth;
                    const scaleY = displayHeight / imgNaturalHeight;
                    
                    // Extract bounding box coordinates [x1, y1, x2, y2] from pipeline_evaluation.py format
                    const [x1, y1, x2, y2] = prediction.box;
                    
                    // Scale coordinates to displayed image size
                    const left = x1 * scaleX;
                    const top = y1 * scaleY;
                    const width = (x2 - x1) * scaleX;
                    const height = (y2 - y1) * scaleY;
                    
                    console.log(`Box ${idx}: [${x1}, ${y1}, ${x2}, ${y2}] -> scaled: [${left}, ${top}, ${width}, ${height}]`);
                    
                    // Color based on confidence (green=high, yellow=medium, red=low)
                    const color = prediction.confidence > 0.8 ? "#28a745" : prediction.confidence > 0.6 ? "#ffc107" : "#dc3545";
                    
                    return (
                      <div key={idx}>
                        {/* Bounding box rectangle */}
                        <div
                          style={{
                            position: "absolute",
                            left: `${left}px`,
                            top: `${top}px`,
                            width: `${width}px`,
                            height: `${height}px`,
                            border: `3px solid ${color}`,
                            borderRadius: "4px",
                            boxSizing: "border-box"
                          }}
                        />
                        {/* Label above box */}
                        <div
                          style={{
                            position: "absolute",
                            left: `${left}px`,
                            top: `${Math.max(0, top - 25)}px`,
                            backgroundColor: color,
                            color: "white",
                            padding: "2px 6px",
                            borderRadius: "4px",
                            fontSize: "11px",
                            fontWeight: "bold",
                            whiteSpace: "nowrap",
                            pointerEvents: "none",
                            zIndex: 10
                          }}
                        >
                          {prediction.species} ({(prediction.confidence * 100).toFixed(1)}%)
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            })()}
          </div>
        )}
        
        <button
          onClick={handlePredict}
          disabled={!selectedFile || isLoading}
          className="button"
          style={{
            padding: "12px 24px",
            backgroundColor: selectedFile && !isLoading ? "#28a745" : "#6c757d",
            color: "white",
            border: "none",
            borderRadius: "6px",
            fontSize: "16px",
            cursor: selectedFile && !isLoading ? "pointer" : "not-allowed",
            marginLeft: "20px"
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
                display: "inline-block",
                marginRight: "8px"
              }} />
              Classifying...
            </>
          ) : (
            "🔍 Classify Species"
          )}
        </button>
      </div>

      {/* Detection Results */}
      {detectionResult && (
        <div className="result-card" style={{
          backgroundColor: "#f8f9fa",
          padding: "1.5rem",
          borderRadius: "8px",
          border: "1px solid #dee2e6",
          maxWidth: "600px",
          margin: "0 auto",
          textAlign: "left"
        }}>
          <h3 style={{ textAlign: "center", marginBottom: "1rem", color: "#28a745" }}>
            🦎 Detection Results
          </h3>
          
          <div style={{ marginBottom: "1rem", textAlign: "center" }}>
            <strong>Total Lizards Detected: {detectionResult.totalLizards}</strong>
          </div>
          
          {detectionResult.predictions.map((prediction, index) => (
            <div key={index} className="species-card" style={{
              backgroundColor: "white",
              padding: "1rem",
              marginBottom: "0.5rem",
              borderRadius: "6px",
              border: "1px solid #e9ecef"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem", flexWrap: "wrap", gap: "0.5rem" }}>
                <h4 style={{ margin: 0, color: "#495057", flex: "1", minWidth: "200px" }}>
                  {prediction.species} ({prediction.scientificName})
                </h4>
                <span className={`confidence-${prediction.confidence > 0.8 ? 'high' : prediction.confidence > 0.6 ? 'medium' : 'low'}`} style={{ 
                  padding: "0.25rem 0.5rem",
                  borderRadius: "4px",
                  fontSize: "0.875rem",
                  fontWeight: "bold",
                  whiteSpace: "nowrap"
                }}>
                  {(prediction.confidence * 100).toFixed(1)}% confidence
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
                  📊 Upload to iNaturalist
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}