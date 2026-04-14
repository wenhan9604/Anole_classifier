import { Link } from "react-router-dom";

export default function LandingPage() {
  return (
    <div className="container" style={{ textAlign: "center", minHeight: "100vh", display: "flex", flexDirection: "column", justifyContent: "center" }}>
      <h1 style={{ fontSize: "2.5rem", marginBottom: "1rem", color: "#2E7D32", fontWeight: "bold" }}>🦎 Florida Anole Classification</h1>
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
          backgroundColor: "#E8F5E9", 
          padding: "1.5rem", 
          borderRadius: "12px", 
          textAlign: "left",
          maxWidth: "400px",
          flex: "1 1 300px",
          border: "2px solid #4CAF50",
          boxShadow: "0 4px 12px rgba(76, 175, 80, 0.2)"
        }}>
          <h3 style={{ textAlign: "center", marginBottom: "1rem", color: "#2E7D32", fontSize: "1.4rem", fontWeight: "bold" }}>
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
                padding: "12px 24px",
                backgroundColor: "#4CAF50",
                color: "white",
                textDecoration: "none",
                borderRadius: "25px",
                fontSize: "16px",
                fontWeight: "600",
                transition: "all 0.2s ease",
                boxShadow: "0 4px 12px rgba(76, 175, 80, 0.3)",
                border: "2px solid #2E7D32",
                letterSpacing: "0.3px",
                width: "160px",
                maxWidth: "160px"
              }}
              onMouseOver={(e) => {
                (e.target as HTMLElement).style.backgroundColor = "#45a049";
                (e.target as HTMLElement).style.transform = "translateY(-2px)";
                (e.target as HTMLElement).style.boxShadow = "0 6px 16px rgba(76, 175, 80, 0.4)";
              }}
              onMouseOut={(e) => {
                (e.target as HTMLElement).style.backgroundColor = "#4CAF50";
                (e.target as HTMLElement).style.transform = "translateY(0)";
                (e.target as HTMLElement).style.boxShadow = "0 4px 12px rgba(76, 175, 80, 0.3)";
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