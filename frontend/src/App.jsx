import { useState, useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useLocation,
} from "react-router-dom"; // 👈 ADD THIS
import "./App.css";
import Header from "./components/Header";
import ImageUpload from "./components/ImageUpload";
import ResultDisplay from "./components/ResultDisplay";
import DiseasesInfo from "./components/DiseaseInfo";
import Footer from "./components/Footer";
import Hero from "./components/Hero";
import About from "./components/About"; // 👈 ADD THIS
import { useTranslation } from "react-i18next";

function MainApp() {
  // 👈 Rename to avoid conflict
  const [selectedImage, setSelectedImage] = useState(null);
  const [detectionResult, setDetectionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const handleImageSelect = async (file) => {
    setSelectedImage(file);
    setError(null);
    setDetectionResult(null);
    setSuccess(false);
  };

  const handleDetection = async (file) => {
    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/detect", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Detection failed. Please try again.");
      }

      const data = await response.json();

      if (!data.detections || data.detections.length === 0) {
        throw new Error("No disease detected.");
      }

      setDetectionResult(data);
      setSuccess(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => {
        setSuccess(false);
      }, 4000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  const handleClear = () => {
    setSelectedImage(null);
    setDetectionResult(null);
    setError(null);
    setSuccess(false);
  };

  const { t, i18n } = useTranslation();

  return (
    <div className="app">
      <Header />
      <Routes>
        {/* Home Route */}
        <Route
          path="/"
          element={
            <>
              <Hero />
              <main className="app-main">
                <div className="container">
                  <div className="content-wrapper">
                    <div className="titles">
                      <h1>{t("app.title")}</h1>
                      <p>{t("app.description")}</p>
                    </div>

                    <ImageUpload
                      onImageSelect={handleImageSelect}
                      selectedImage={selectedImage}
                      onDetect={handleDetection}
                      loading={loading}
                      error={error}
                      success={success}
                      onSuccessClose={() => setSuccess(false)}
                    />

                    <div className="divider"></div>

                    {detectionResult ? (
                      <ResultDisplay
                        result={detectionResult}
                        onClear={handleClear}
                      />
                    ) : (
                      <DiseasesInfo />
                    )}
                  </div>
                </div>
              </main>
            </>
          }
        />

        {/* About Route */}
        <Route path="/about" element={<About />} />
      </Routes>
      <Footer />
    </div>
  );
}

function App() {
  return (
    <Router>
      {" "}
      {/* 👈 Wrap with Router */}
      <MainApp />
    </Router>
  );
}

export default App;
