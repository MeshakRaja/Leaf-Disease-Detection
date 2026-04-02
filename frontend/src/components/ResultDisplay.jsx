import { useRef, useEffect, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Zap,
  RotateCcw,
  MapPin,
  BarChart2,
  ChevronDown,
  ChevronUp,
  Layers,
  Eye,
} from "lucide-react";
import "./ResultDisplay.css";
import { useTranslation } from "react-i18next";

/* ─────────────────────────────────────────────
   Helper: draw bounding boxes on a canvas
───────────────────────────────────────────── */
function BBoxCanvas({ imageUrl, detections, imageSize }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !imageUrl) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const maxW = canvas.parentElement.clientWidth || 500;
      const scale = Math.min(maxW / img.width, 420 / img.height, 1);
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      const DISEASE_COLORS = {
        "Healthy Leaf": "#4caf50",
        "Bacterial Spot": "#ff9800",
        "Early Blight": "#f44336",
        "Late Blight": "#d32f2f",
        "Septoria Leaf Spot": "#ff6f00",
        "Yellow Leaf Curl Virus": "#fbc02d",
        "Mosaic Virus": "#e91e63",
        "Leaf Mold": "#9c27b0",
        "Spider Mites": "#673ab7",
      };

      detections.forEach((det, i) => {
        if (!det.bbox) return;
        const [x, y, w, h] = det.bbox;
        const sx = x * scale;
        const sy = y * scale;
        const sw = w * scale;
        const sh = h * scale;
        const color = DISEASE_COLORS[det.name] || "#5a9e44";

        // Glow effect
        ctx.shadowColor = color;
        ctx.shadowBlur = 12;

        // Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.strokeRect(sx, sy, sw, sh);

        ctx.shadowBlur = 0;

        // Label background
        const label = `${det.name} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 12px DM Sans, sans-serif";
        const textW = ctx.measureText(label).width + 12;
        const textH = 22;
        const labelY = sy > textH + 4 ? sy - textH - 2 : sy + 4;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.roundRect(sx, labelY, textW, textH, 4);
        ctx.fill();

        ctx.fillStyle = "#fff";
        ctx.fillText(label, sx + 6, labelY + 15);
      });
    };
    img.src = imageUrl;
  }, [imageUrl, detections, imageSize]);

  return <canvas ref={canvasRef} className="bbox-canvas" />;
}

/* ─────────────────────────────────────────────
   Helper: mini confidence bar chart
───────────────────────────────────────────── */
function ConfidenceChart({ detections }) {
  const DISEASE_COLORS = {
    "Healthy Leaf": "#4caf50",
    "Bacterial Spot": "#ff9800",
    "Early Blight": "#f44336",
    "Late Blight": "#d32f2f",
    "Septoria Leaf Spot": "#ff6f00",
    "Yellow Leaf Curl Virus": "#fbc02d",
    "Mosaic Virus": "#e91e63",
    "Leaf Mold": "#9c27b0",
    "Spider Mites": "#673ab7",
  };

  return (
    <div className="conf-chart">
      {detections.map((det, i) => (
        <div key={i} className="conf-row">
          <span className="conf-label">{det.name}</span>
          <div className="conf-track">
            <div
              className="conf-bar"
              style={{
                width: `${det.confidence * 100}%`,
                background: DISEASE_COLORS[det.name] || "#5a9e44",
                animationDelay: `${i * 0.1}s`,
              }}
            />
          </div>
          <span className="conf-pct">{(det.confidence * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

/* ─────────────────────────────────────────────
   Helper: bbox location label
───────────────────────────────────────────── */
function getBBoxLocation(bbox, imgW = 640, imgH = 640) {
  if (!bbox) return "—";
  const [x, y, w, h] = bbox;
  const cx = x + w / 2;
  const cy = y + h / 2;
  const vert = cy < imgH / 3 ? "Top" : cy > (imgH * 2) / 3 ? "Bottom" : "Mid";
  const horiz =
    cx < imgW / 3 ? "Left" : cx > (imgW * 2) / 3 ? "Right" : "Center";
  return `${vert}-${horiz}`;
}

/* ─────────────────────────────────────────────
   Main Component
───────────────────────────────────────────── */
function ResultDisplay({ result, onClear }) {
   const { t } = useTranslation();
  const [showDetails, setShowDetails] = useState(false);
  const [showChart, setShowChart] = useState(false);

  const getDiseaseColor = (disease) => {
    const map = {
      "Healthy Leaf": "healthy",
      "Bacterial Spot": "bacterial",
      "Early Blight": "blight",
      "Late Blight": "blight-late",
      "Septoria Leaf Spot": "septoria",
      "Yellow Leaf Curl Virus": "ylcv",
      "Mosaic Virus": "mosaic",
      "Leaf Mold": "mold",
      "Spider Mites": "mites",
    };
    return map[disease] || "unknown";
  };

  const getSeverityLabel = (confidence,className) => {
      if (className === "Healthy Leaf") {
        return { label: "Healthy", cls: "sev-healthy" };
      }
    if (confidence >= 0.9) return { label: "Critical", cls: "sev-critical" };
    if (confidence >= 0.7) return { label: "High", cls: "sev-high" };
    if (confidence >= 0.5) return { label: "Medium", cls: "sev-medium" };
    return { label: "Low", cls: "sev-low" };
  };
const normalizeName = (name) => {
  return name
    ?.replace(/_/g, " ")
    ?.replace(/\b\w/g, (c) => c.toUpperCase())
    ?.trim();
};
  const CONFIDENCE_THRESHOLD = 0.6;
  const allDetections = (result.detections || []).map((d) => ({
    ...d,
    name: normalizeName(d.name),
  }));
  const topDetection = allDetections[0]
    ? { ...allDetections[0], name: normalizeName(allDetections[0].name) }
    : null;
  const isValidDetection =
    topDetection && topDetection.confidence >= CONFIDENCE_THRESHOLD;

  /* image with bboxes — expects result.imageUrl (base64 or URL) */
  const hasImage = !!result.imageUrl;

  const diseaseDescriptions = {
    "Healthy Leaf": {
      description: "Your leaf is perfectly healthy!",
      recommendations: [
        "Continue regular watering and care",
        "Monitor for any changes",
        "Maintain proper ventilation",
      ],
    },
    "Bacterial Spot": {
      description: "Bacterial leaf spots are caused by bacterial pathogens.",
      recommendations: [
        "Remove infected leaves",
        "Apply copper-based fungicide",
        "Improve air circulation",
        "Avoid overhead watering",
      ],
    },
    "Early Blight": {
      description:
        "Early blight is a fungal disease caused by Alternaria solani.",
      recommendations: [
        "Remove infected lower leaves",
        "Apply fungicide (Mancozeb)",
        "Avoid wetting foliage",
        "Space plants properly",
      ],
    },
    "Late Blight": {
      description:
        "Late blight is a serious fungal disease caused by Phytophthora infestans.",
      recommendations: [
        "Isolate plant immediately",
        "Remove and destroy infected parts",
        "Apply systemic fungicide",
        "Ensure good drainage",
      ],
    },
    "Septoria Leaf Spot": {
      description:
        "Septoria leaf spot is caused by the fungus Septoria lycopersici.",
      recommendations: [
        "Remove lower infected leaves",
        "Apply fungicide regularly",
        "Improve air circulation",
        "Mulch to prevent spore splash",
      ],
    },
    "Yellow Leaf Curl Virus": {
      description: "YLCV is a viral disease spread by whiteflies.",
      recommendations: [
        "Remove entire plant if heavily infected",
        "Control whitefly population",
        "Plant resistant varieties",
        "Use reflective mulch",
      ],
    },
    "Mosaic Virus": {
      description: "Mosaic virus causes mottled patterns on leaves.",
      recommendations: [
        "Remove infected plant",
        "Control aphids and other vectors",
        "Sanitize tools between plants",
        "Plant virus-resistant varieties",
      ],
    },
    "Leaf Mold": {
      description: "Leaf mold is a fungal disease favoring high humidity.",
      recommendations: [
        "Increase air circulation",
        "Reduce humidity levels",
        "Apply sulfur fungicide",
        "Remove infected leaves",
      ],
    },
    "Spider Mites": {
      description:
        "Spider mites are tiny pests that cause stippling on leaves.",
      recommendations: [
        "Spray with neem oil",
        "Increase humidity",
        "Prune heavily infested areas",
        "Use water spray to dislodge mites",
      ],
    },
  };

  const diseaseInfo = isValidDetection
    ? diseaseDescriptions[topDetection.name] || {
        description: "Unknown disease detected.",
        recommendations: [
          "Consult a plant pathologist",
          "Monitor plant condition closely",
        ],
      }
    : {
        description: "This does not appear to be a tomato leaf.",
        recommendations: [
          "Upload a clear tomato leaf image",
          "Ensure the leaf fills most of the frame",
          "Avoid unrelated objects",
        ],
      };

 const severity = isValidDetection
   ? getSeverityLabel(topDetection.confidence, topDetection.name)
   : null;


  return (
    <div className="result-display">
      {/* ── Header ── */}
      <div className="result-header">
        <h2 className="result-title">
          {" "}
          {t("result-display.result-display-title")}
        </h2>
        <button
          className="btn-reset"
          onClick={onClear}
          title="Analyze another image"
        >
          <RotateCcw size={18} />
        </button>
      </div>

      {/* ── Section A: Visual detection overlay ── */}
      {hasImage && (
        <div className="section bbox-section">
          <div className="section-label">
            <Eye size={13} /> {t("result-display.section-label")}
          </div>
          <div className="bbox-wrapper">
            <BBoxCanvas
              imageUrl={result.imageUrl}
              detections={allDetections.filter((d) => d.bbox)}
            />
            {!allDetections.some((d) => d.bbox) && (
              <p className="bbox-note">{t("result-display.bbox-note")}</p>
            )}
          </div>
          <p className="bbox-caption">{t("result-display.bbox-caption")}</p>
        </div>
      )}

      {/* ── Section B: Primary result card ── */}
      {isValidDetection ? (
        <div
          className={`result-card disease-${getDiseaseColor(topDetection.name)}`}
        >
          <div className="result-icon">
            {topDetection.name === "Healthy Leaf" ? (
              <CheckCircle2 size={48} />
            ) : (
              <AlertCircle size={48} />
            )}
          </div>
          <div className="result-content">
            <div className="disease-header-row">
              <h3 className="disease-name">{topDetection.name}</h3>
              <span className={`severity-pill ${severity.cls}`}>
                {severity.label}
              </span>
            </div>

            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{ width: `${topDetection.confidence * 100}%` }}
              />
            </div>
            <p className="confidence-text">
              {t("result-display.confidence-label")}{" "}
              <strong>{(topDetection.confidence * 100).toFixed(1)}%</strong>
            </p>

            {/* Summary stats */}
            <div className="summary-stats">
              <div className="stat-chip">
                <Layers size={12} />
                {allDetections.length} region
                {allDetections.length !== 1 ? "s" : ""} detected
              </div>
              {topDetection.bbox && (
                <div className="stat-chip">
                  <MapPin size={12} />
                  Primary: {getBBoxLocation(topDetection.bbox)}
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="result-card disease-unknown">
          <div className="result-icon">
            <AlertCircle size={48} />
          </div>
          <div className="result-content">
            <h3 className="disease-name">
              {" "}
              {t("result-display.disease-name")}
            </h3>
            <p className="confidence-text">
              {t("result-display.confidence-text")}
            </p>
          </div>
        </div>
      )}

      {/* ── Section B2: Per-region details (expandable) ── */}
      {allDetections.length > 0 && (
        <div className="section">
          <button
            className="toggle-btn"
            onClick={() => setShowDetails((v) => !v)}
          >
            <span>
              <Layers size={13} /> {t("result-display.per-region-details")} (
              {allDetections.length})
            </span>
            {showDetails ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          </button>

          {showDetails && (
            <div className="details-table">
              <div className="dt-head">
                <span>#</span>
                <span>{t("result-display.dt-head-one")}</span>
                <span>{t("result-display.dt-head-two")}</span>
                <span>{t("result-display.dt-head-three")}</span>
              </div>
              {allDetections.map((det, i) => (
                <div key={i} className="dt-row">
                  <span className="dt-idx">{i + 1}</span>
                  <span className="dt-name">{det.name}</span>
                  <span className="dt-conf">
                    <span
                      className="dt-conf-bar"
                      style={{ width: `${det.confidence * 100}%` }}
                    />
                    {(det.confidence * 100).toFixed(1)}%
                  </span>
                  <span className="dt-loc">
                    <MapPin size={11} />
                    {getBBoxLocation(det.bbox)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Section D: Confidence chart (expert view) ── */}
      {allDetections.length > 0 && (
        <div className="section">
          <button
            className="toggle-btn"
            onClick={() => setShowChart((v) => !v)}
          >
            <span>
              <BarChart2 size={13} />
              {t("result-display.confidence-distribution")}
            </span>
            {showChart ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          </button>

          {showChart && <ConfidenceChart detections={allDetections} />}
        </div>
      )}

      {/* ── Section C: Disease info ── */}
      <div className="disease-info">
        <h4 className="info-title">
          <Zap size={18} /> {t("result-display.info-title")}
        </h4>
        <p className="info-description">{diseaseInfo.description}</p>
      </div>

      {/* ── Recommendations ── */}
      <div className="recommendations">
        <h4 className="recommendations-title">
          {t("result-display.recommendations-title")}
        </h4>
        <ul className="recommendations-list">
          {diseaseInfo.recommendations.map((rec, i) => (
            <li key={i} className="recommendation-item">
              <span className="recommendation-bullet" />
              {rec}
            </li>
          ))}
        </ul>
      </div>

      {/* ── Footer ── */}
      <div className="result-footer">
        <p className="footer-note">{t("result-display.footer-note")}</p>
      </div>
    </div>
  );
}

export default ResultDisplay;
