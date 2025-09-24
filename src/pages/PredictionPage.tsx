import { useState } from "react";
import { Link } from "react-router-dom";
import { iNaturalistAPI, getCurrentLocation } from "../services/iNaturalistService";
import type { iNaturalistObservation } from "../services/iNaturalistService";

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
}

interface DetectionResult {
  totalLizards: number;
  predictions: PredictionResult[];
  imageUrl?: string;
}

export default function PredictionPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadingToiNaturalist, setUploadingToiNaturalist] = useState(false);

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
      // TODO: Replace with actual ML model API call
      // For now, simulate realistic predictions with confidence scores
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      const mockResult: DetectionResult = {
        totalLizards: Math.floor(Math.random() * 3) + 1, // 1-3 lizards
        predictions: [
          {
            species: "Green Anole",
            scientificName: "Anolis carolinensis",
            confidence: Math.random() * 0.3 + 0.7, // 70-100%
            count: 1
          },
          ...(Math.random() > 0.5 ? [{
            species: "Brown Anole",
            scientificName: "Anolis sagrei",
            confidence: Math.random() * 0.4 + 0.6, // 60-100%
            count: 1
          }] : [])
        ],
        imageUrl: previewUrl || undefined
      };
      
      setDetectionResult(mockResult);
    } catch (error) {
      console.error("Prediction failed:", error);
      alert("Prediction failed. Please try again.");
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
          notes: `Detected by AI with ${Math.round(prediction.confidence * 100)}% confidence`
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
      <div style={{ marginBottom: "2rem" }}>
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
          <div style={{ marginBottom: "1rem" }}>
            <img 
              src={previewUrl} 
              alt="Preview" 
              className="image-preview"
              style={{ 
                maxWidth: "300px", 
                height: "auto", 
                borderRadius: "6px",
                border: "2px solid #ddd"
              }} 
            />
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
            cursor: selectedFile && !isLoading ? "pointer" : "not-allowed"
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
                  {Math.round(prediction.confidence * 100)}% confidence
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