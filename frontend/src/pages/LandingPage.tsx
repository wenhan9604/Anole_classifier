import { Link } from "react-router-dom";
import AppMetrics from "../components/AppMetrics";

export default function LandingPage() {
  return (
    <main style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #fbfcf8 0%, #eef6e8 52%, #f7f1df 100%)",
      color: "#1d2b1f",
      padding: "48px 24px"
    }}>
      <style>{`
        .landing-shell {
          max-width: 1180px;
          margin: 0 auto;
        }

        .landing-hero {
          text-align: center;
          margin-bottom: 34px;
        }

        .landing-grid {
          display: grid;
          grid-template-columns: minmax(230px, 0.75fr) minmax(320px, 1fr) minmax(380px, 1.25fr);
          gap: 24px;
          align-items: stretch;
        }

        .landing-card {
          border: 1px solid rgba(58, 112, 57, 0.2);
          border-radius: 24px;
          background: rgba(255, 255, 255, 0.74);
          box-shadow: 0 24px 60px rgba(45, 69, 39, 0.12);
          backdrop-filter: blur(10px);
        }

        @media (max-width: 1080px) {
          .landing-grid {
            grid-template-columns: 1fr 1fr;
          }
          .landing-community {
            grid-column: 1 / -1;
          }
        }

        @media (max-width: 720px) {
          main {
            padding: 28px 16px !important;
          }
          .landing-grid {
            grid-template-columns: 1fr;
          }
          .landing-title {
            font-size: 2.4rem !important;
          }
        }
      `}</style>

      <div className="landing-shell">
        <header className="landing-hero">
          <h1 className="landing-title" style={{
            fontSize: "3.2rem",
            margin: "0 0 0.8rem",
            color: "#1f5f2a",
            fontWeight: 850,
            letterSpacing: "-0.045em",
            lineHeight: 1
          }}>
            Lizard Lens
          </h1>
          <p style={{ fontSize: "1.02rem", margin: "0 auto", color: "#50614a", maxWidth: 650, lineHeight: 1.65 }}>
            Upload an anole photo, classify supported Florida species, and contribute observation data back to citizen science.
          </p>
        </header>

        <div className="landing-grid">
          <section className="landing-card" style={{ padding: 0, overflow: "hidden" }}>
            <img
              src="/36514_221109568.jpg"
              alt="Florida anole perched on bamboo"
              className="image-preview"
              style={{
                width: "100%",
                height: "100%",
                minHeight: 320,
                objectFit: "cover",
                display: "block"
              }}
            />
          </section>

          <section className="landing-card" style={{
            padding: "1.65rem",
            textAlign: "left",
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            gap: "1.5rem"
          }}>
            <div>
              <div style={{ color: "#2E7D32", fontSize: "0.75rem", fontWeight: 850, letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "0.7rem" }}>
                Classification workflow
              </div>
              <h2 style={{ color: "#1d3d22", fontSize: "1.55rem", lineHeight: 1.15, margin: "0 0 0.85rem", letterSpacing: "-0.02em" }}>
                From field photo to species record.
              </h2>
              <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "grid", gap: "0.8rem", color: "#43543c", fontSize: "0.92rem", lineHeight: 1.45 }}>
                {[
                  "Detects one or more lizards in a single image",
                  "Classifies five supported Florida anole species",
                  "Shows confidence scores for each prediction",
                  "Supports iNaturalist submissions for citizen science",
                  "Optimized for mobile field use"
                ].map((feature) => (
                  <li key={feature} style={{ display: "grid", gridTemplateColumns: "18px 1fr", gap: "0.55rem", alignItems: "start" }}>
                    <span style={{ width: 9, height: 9, borderRadius: "50%", background: "#4CAF50", marginTop: 6, boxShadow: "0 0 0 4px rgba(76, 175, 80, 0.12)" }} />
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
            </div>

            <Link
              to="/predict"
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                padding: "13px 22px",
                background: "linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%)",
                color: "white",
                textDecoration: "none",
                borderRadius: "999px",
                fontSize: "0.96rem",
                fontWeight: 800,
                boxShadow: "0 16px 34px rgba(46, 125, 50, 0.28)",
                letterSpacing: "0.01em"
              }}
            >
              Classify an image
            </Link>
          </section>

          <section className="landing-card landing-community" style={{
            padding: "1.65rem",
            textAlign: "center",
            background: "linear-gradient(180deg, rgba(246, 251, 241, 0.95), rgba(255, 255, 255, 0.78))"
          }}>
            <AppMetrics />
          </section>
        </div>
      </div>
    </main>
  );
}