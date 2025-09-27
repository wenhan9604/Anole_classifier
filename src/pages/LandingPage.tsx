import { Link } from "react-router-dom";

export default function LandingPage() {
  return (
    <div className="container" style={{ textAlign: "center", minHeight: "100vh", display: "flex", flexDirection: "column", justifyContent: "center" }}>
      <h1 style={{ fontSize: "2.5rem", marginBottom: "1rem" }}>🦎 Florida Anole Classification</h1>
      <p style={{ fontSize: "1rem", marginBottom: "2rem", color: "#666" }}>
        Upload an image of an anole to classify its species and contribute to citizen science!
      </p>
      
      {/* Main content area with side-by-side layout */}
      <div style={{ 
        display: "flex", 
        alignItems: "flex-start", 
        justifyContent: "center", 
        gap: "2rem",
        marginBottom: "2rem",
        flexWrap: "wrap"
      }}>
        {/* Image section */}
        <div style={{ flex: "0 0 auto" }}>
          <img 
            src="/36514_221109568.jpg" 
            alt="Florida Anole" 
            className="image-preview"
            style={{ 
              maxWidth: "300px", 
              height: "auto", 
              borderRadius: "8px",
              boxShadow: "0 4px 8px rgba(0,0,0,0.1)"
            }} 
          />
        </div>
        
        {/* Features section with button */}
        <div style={{ 
          backgroundColor: "#f8f9fa", 
          padding: "1.5rem", 
          borderRadius: "8px", 
          textAlign: "left",
          maxWidth: "400px",
          flex: "1 1 300px"
        }}>
          <h3 style={{ textAlign: "center", marginBottom: "1rem", color: "#28a745", fontSize: "1.2rem" }}>
            ✨ Features
          </h3>
          <ul style={{ margin: 0, paddingLeft: "1.2rem", fontSize: "0.9rem", lineHeight: "1.4", marginBottom: "1.5rem" }}>
            <li>Detects the number of lizards in one image</li>
            <li>Classifies 5 Florida anole species</li>
            <li>Shows confidence levels for each prediction</li>
            <li>Uploads to iNaturalist for citizen science</li>
            <li>Mobile-friendly for field use</li>
          </ul>
          
          {/* Navigation button inside features block */}
          <div style={{ textAlign: "center" }}>
            <Link 
              to="/predict" 
              style={{
                display: "inline-block",
                padding: "10px 20px",
                backgroundColor: "#28a745",
                color: "white",
                textDecoration: "none",
                borderRadius: "20px",
                fontSize: "15px",
                fontWeight: "500",
                transition: "all 0.2s ease",
                boxShadow: "0 2px 8px rgba(40, 167, 69, 0.2)",
                border: "2px solid transparent",
                letterSpacing: "0.3px",
                width: "140px",
                maxWidth: "140px"
              }}
              onMouseOver={(e) => {
                (e.target as HTMLElement).style.backgroundColor = "#218838";
                (e.target as HTMLElement).style.transform = "translateY(-1px)";
                (e.target as HTMLElement).style.boxShadow = "0 4px 12px rgba(40, 167, 69, 0.3)";
              }}
              onMouseOut={(e) => {
                (e.target as HTMLElement).style.backgroundColor = "#28a745";
                (e.target as HTMLElement).style.transform = "translateY(0)";
                (e.target as HTMLElement).style.boxShadow = "0 2px 8px rgba(40, 167, 69, 0.2)";
              }}
            >
              🔍 Classify
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}