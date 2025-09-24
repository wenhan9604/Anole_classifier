import { Link } from "react-router-dom";

export default function LandingPage() {
  return (
    <div className="container" style={{ textAlign: "center" }}>
      <h1>🦎 Florida Anole Classification</h1>
      <p style={{ fontSize: "1.1rem", marginBottom: "2rem" }}>
        Upload an image of an anole to classify its species
      </p>
      
      {/* Display the anole image */}
      <div style={{ margin: "2rem 0" }}>
        <img 
          src="/36514_221109568.jpg" 
          alt="Florida Anole" 
          className="image-preview"
          style={{ 
            maxWidth: "400px", 
            height: "auto", 
            borderRadius: "8px",
            boxShadow: "0 4px 8px rgba(0,0,0,0.1)"
          }} 
        />
      </div>
      
      {/* Features list */}
      <div style={{ 
        backgroundColor: "#f8f9fa", 
        padding: "1.5rem", 
        borderRadius: "8px", 
        margin: "2rem auto",
        textAlign: "left",
        maxWidth: "500px"
      }}>
        <h3 style={{ textAlign: "center", marginBottom: "1rem", color: "#28a745" }}>
          Features
        </h3>
        <ul style={{ margin: 0, paddingLeft: "1.5rem" }}>
          <li>Detects the number of lizards in one image</li>
          <li>Classifies 5 Florida anole species</li>
          <li>Shows confidence levels for each prediction</li>
          <li>Uploads to iNaturalist for citizen science</li>
          <li>Mobile-friendly for field use</li>
        </ul>
      </div>
      
      {/* Navigation button to prediction page */}
      <Link 
        to="/predict" 
        className="button"
        style={{
          display: "inline-block",
          padding: "12px 24px",
          backgroundColor: "#007bff",
          color: "white",
          textDecoration: "none",
          borderRadius: "6px",
          fontSize: "16px",
          fontWeight: "500",
          transition: "background-color 0.2s"
        }}
        onMouseOver={(e) => (e.target as HTMLElement).style.backgroundColor = "#0056b3"}
        onMouseOut={(e) => (e.target as HTMLElement).style.backgroundColor = "#007bff"}
      >
        Start Classification
      </Link>
    </div>
  );
}