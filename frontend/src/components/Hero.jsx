import "./Hero.css";
import { ArrowRight, ScanLine, Zap } from "lucide-react";
import heroImage from "../assets/hero.webp"; // replace with your image
import { useTranslation } from "react-i18next";

function Hero() {
  const { t, i18n } = useTranslation();
  return (
    <section className="hero" data-lang={i18n.language}>
      <div className="hero-container">
        {/* LEFT IMAGE */}
        <div className="hero-image">
          <img src={heroImage} alt="Tomato Detection" />
          <div className="scanner-overlay">
            <div className="scanner-line"></div>
          </div>
        </div>

        {/* RIGHT CONTENT */}
        <div className="hero-content">
          <div className="hero-badge">
            <Zap size={16} />
            {t("hero.hero-badge")}
          </div>

          <h1 className="hero-title">
            {t("hero.hero-title")}
            <span>{t("hero.hero-title-span")}</span>
          </h1>

          <p className="hero-description">{t("hero.hero-description")}</p>

          <div className="hero-buttons">
            <a className="hero-anchors" href="#imageUploader">
              {" "}
              <button className="btn-primary">
                <ScanLine size={18} />
                {t("hero.btn-primary")}
              </button>
            </a>

            <a className="hero-anchors" href="#DiseaseInfo">
              <button className="btn-secondary">
                {t("hero.btn-secondary")}
                <ArrowRight size={18} />
              </button>
            </a>
          </div>

          {/* <div className="hero-trust">
            <div className="avatars">
              <div className="avatar"></div>
              <div className="avatar"></div>
              <div className="avatar"></div>
            </div>

            <p>
              Trusted by <strong>2,000+</strong> farmers worldwide
            </p>
          </div> */}
        </div>
      </div>
    </section>
  );
}

export default Hero;
