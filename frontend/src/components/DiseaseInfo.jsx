import {
  AlertCircle,
  CheckCircle2,
  Microscope,
  Bug,
  Droplets,
} from "lucide-react";
import "./DiseaseInfo.css";
import { useTranslation } from "react-i18next";
function DiseasesInfo() {
  const { t, i18n } = useTranslation();
  const diseases = [
    {
      name: "disease-name-one",
      icon: CheckCircle2,
      color: "healthy",
      description: "disease-one-description",
    },
    {
      name: "disease-name-two",
      icon: AlertCircle,
      color: "bacterial",
      description: "disease-two-description",
    },
    {
      name: "disease-name-three",
      icon: AlertCircle,
      color: "blight",
      description: "disease-three-description",
    },
    {
      name: "disease-name-four",
      icon: AlertCircle,
      color: "blight-late",
      description: "disease-four-description",
    },
    {
      name: "disease-name-five",
      icon: AlertCircle,
      color: "septoria",
      description: "disease-five-description",
    },

    {
      name: "disease-name-six",
      icon: Droplets,
      color: "mold",
      description: "disease-six-description",
    },
    {
      name: "disease-name-seven",
      icon: Bug,
      color: "mites",
      description: "disease-seven-description",
    },
  ];

  return (
    <div id="DiseaseInfo" className="diseases-info">
      <h2 className="section-title">{t("disease-info.section-title")}</h2>
      <p className="section-subtitle">{t("disease-info.section-subtitle")}</p>

      <div className="diseases-grid">
        {diseases.map((disease, index) => {
          const IconComponent = disease.icon;
          return (
            <div
              key={index}
              className={`disease-card disease-${disease.color}`}
            >
              <div className="disease-card-icon">
                <IconComponent size={24} />
              </div>
              <h3 className="disease-card-name">
                {t(`disease-info.${disease.name}`)}
              </h3>
              <p className="disease-card-description">
                {t(`disease-info.${disease.description}`)}
              </p>
            </div>
          );
        })}
      </div>


    </div>
  );
}

export default DiseasesInfo;
