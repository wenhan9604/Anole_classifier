import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { Toaster } from 'react-hot-toast';
import LandingPage from "./pages/LandingPage";
import PredictionPage from "./pages/PredictionPage";
import NewDesignPage from "./pages/NewDesignPage";
import "./App.css";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Landing page */}
        <Route path="/" element={<LandingPage />} />

        {/* Upload / prediction page */}
        <Route path="/predict" element={<PredictionPage />} />

        {/* New sophisticated design */}
        <Route path="/newdesign" element={<NewDesignPage />} />

        {/* Redirect all unknown routes back to home */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <Toaster position="bottom-right" />
    </BrowserRouter>
  );
}

export default App;