import { useState, useRef, useCallback } from "react";
import Webcam from "react-webcam";
import { Upload, Camera, AlertCircle, X, CheckCircle } from "lucide-react";
import styles from "./ImageUpload.module.css";
import { useTranslation } from "react-i18next";

function ImageUpload({
  onImageSelect,
  selectedImage,
  onDetect,
  loading,
  error,
  success, // New prop from parent
  onSuccessClose, // New prop from parent
}) {
  const [cameraActive, setCameraActive] = useState(false);
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);
  const { t } = useTranslation();

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "environment",
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) onImageSelect(file);
  };

  const handleRemoveImage = () => {
    onImageSelect(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const captureImage = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (!imageSrc) return;

    fetch(imageSrc)
      .then((res) => res.blob())
      .then((blob) => {
        const file = new File([blob], "camera-image.jpg", {
          type: "image/jpeg",
        });
        onImageSelect(file);
        setCameraActive(false);
      })
      .catch((err) => console.error("Capture failed:", err));
  }, [onImageSelect]);

  const stopCamera = () => {
    const stream = webcamRef.current?.video?.srcObject;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    setCameraActive(false);
  };

  const handleDetectClick = () => {
    if (selectedImage && onDetect) {
      onDetect(selectedImage);
    }
  };

  return (
    <div className={styles["image-upload"]} id="imageUploader">
      {/* TITLE — full width */}
      <h2 className={styles["section-title"]}>
        {t("image-upload.section-title")}
      </h2>

      {/* TWO-COLUMN GRID */}
      <div className={styles["main-grid"]}>
        {/* ── LEFT COLUMN ── */}
        <div className={styles["left-column"]}>
          {cameraActive ? (
            /* CAMERA VIEW */
            <div className={styles["camera-container"]}>
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                className={styles["camera-video"]}
              />
              <div className={styles["camera-controls"]}>
                <button
                  onClick={captureImage}
                  className={styles["btn-primary"]}
                >
                  <Camera size={18} />
                  {t("image-upload.btn-primary")}
                </button>
                <button
                  onClick={stopCamera}
                  className={styles["btn-secondary"]}
                >
                  {t("image-upload.btn-secondary")}
                </button>
              </div>
            </div>
          ) : (
            <>
              {/* UPLOAD AREA or PREVIEW */}
              {selectedImage ? (
                <div className={styles["image-preview"]}>
                  <button
                    className={styles["remove-image-btn"]}
                    onClick={handleRemoveImage}
                  >
                    <X size={18} />
                  </button>
                  <img
                    src={URL.createObjectURL(selectedImage)}
                    alt="preview"
                    className={styles["preview-image"]}
                  />
                  <p>{selectedImage.name}</p>
                </div>
              ) : (
                <div
                  className={styles["upload-area"]}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={40} />
                  <p>{t("image-upload.upload-area")}</p>
                  <input
                    type="file"
                    ref={fileInputRef}
                    accept="image/*"
                    onChange={handleFileChange}
                    className={styles["file-input"]}
                  />
                </div>
              )}

              {/* BUTTONS — stacked vertically */}
              <div className={styles["button-group"]}>
                <button
                  className={styles["btn-primary"]}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={18} /> {t("image-upload.option-one")}
                </button>
                <button
                  className={styles["btn-secondary"]}
                  onClick={() => setCameraActive(true)}
                >
                  <Camera size={18} /> {t("image-upload.option-two")}
                </button>
              </div>
            </>
          )}

          {/* DETECT BUTTON */}
          {selectedImage && !cameraActive && (
            <button
              className={styles["btn-detect"]}
              onClick={handleDetectClick}
              disabled={loading}
            >
              {loading
                ? t("image-upload.detection-analyze")
                : t("image-upload.detection-btn")}
            </button>
          )}

          {/* ERROR MESSAGE */}
          {error && (
            <div className={styles["error-message"]}>
              <AlertCircle size={16} />
              {error}
            </div>
          )}
        </div>
        {/* ── END LEFT COLUMN ── */}

        {/* ── RIGHT COLUMN ── */}
        <div className={styles["right-column"]}>
          <div className={styles["how-it-works"]}>
            <h3 className={styles["how-title"]}>
              {t("disease-info.how-title")}
            </h3>
            <div className={styles["steps"]}>
              <div className={styles["step"]}>
                <div className={styles["step-number"]}>1</div>
                <div>
                  <h4>{t("disease-info.step-one")}</h4>
                  <p>{t("disease-info.step-one-desc")}</p>
                </div>
              </div>
              <div className={styles["step"]}>
                <div className={styles["step-number"]}>2</div>
                <div>
                  <h4>{t("disease-info.step-two")}</h4>
                  <p>{t("disease-info.step-two-desc")}</p>
                </div>
              </div>
              <div className={styles["step"]}>
                <div className={styles["step-number"]}>3</div>
                <div>
                  <h4>{t("disease-info.step-three")}</h4>
                  <p>{t("disease-info.step-three-desc")}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        {/* ── END RIGHT COLUMN ── */}
      </div>
      {/* END GRID */}

      {/* TIPS — centered below both columns */}
      <div className={styles["tips-section"]}>
        <h3 className={styles["tips-title"]}>
          📸 {t("disease-info.tips-title")}
        </h3>
        <ul className={styles["tips-list"]}>
          <li>{t("disease-info.tips-list-one")}</li>
          <li>{t("disease-info.tips-list-two")}</li>
          <li>{t("disease-info.tips-list-three")}</li>
          <li>{t("disease-info.tips-list-four")}</li>
          <li>{t("disease-info.tips-list-five")}</li>
        </ul>
      </div>

      {/* SUCCESS POPUP - Full screen overlay */}
      {success && (
        <div className={styles["success-popup"]}>
          <div className={styles["popup-content"]}>
            <CheckCircle size={24} className={styles["success-icon"]} />
            <span>Scroll below to see the results</span>
            <button
              onClick={onSuccessClose}
              className={styles["close-btn"]}
              aria-label="Close success message"
            >
              <X size={18} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImageUpload;
