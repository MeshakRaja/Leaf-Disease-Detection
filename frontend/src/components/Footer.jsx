import "./Footer.css";
import { useTranslation } from "react-i18next";
import Logo from "../assets/icon1.png";
function Footer() {
 const { t, i18n } = useTranslation();
  return (
    
    <footer className="footer">
      <div className="footer-container">
        {/* Brand Section */}
        <div className="footer-section brand">
          <div className='logo-icon'><img src={Logo} alt="logo" /></div>
          <h2>{t("footer.footer-title")}</h2>
          <p>{t("footer.footer-desc")}</p>
        </div>

        {/* Features */}
        <div className="footer-section">
          <h3>{t("footer.feature-title")}</h3>
          <ul>
            <li>{t("footer.feature-list-one")}</li>
            <li>{t("footer.feature-list-two")}</li>
            <li>{t("footer.feature-list-three")}</li>
            <li>{t("footer.feature-list-four")}</li>
          </ul>
        </div>

        {/* Resources */}
        <div className="footer-section">
          <h3>{t("footer.resources-title")}</h3>
          <ul>
            <li>
              <a href="https://docs.ultralytics.com" target="_blank">
                {t("footer.resources-list-one")}
              </a>
            </li>
            <li>
              <a href="https://react.dev" target="_blank">
                {t("footer.resources-list-two")}
              </a>
            </li>
            <li>
              <a href="https://vitejs.dev" target="_blank">
                {t("footer.resources-list-three")}
              </a>
            </li>
          </ul>
        </div>
      </div>

      <div className="footer-bottom">
        © 2026 LeafGuard Vision AI | All Rights Reserved
      </div>
    </footer>
  );
}

export default Footer;
