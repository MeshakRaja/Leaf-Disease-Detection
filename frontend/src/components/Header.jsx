import "./Header.css";
// import { Leaf } from "lucide-react";
import { useTranslation } from "react-i18next";
import {Link} from 'react-router-dom'
import Logo from '../assets/icon1.png';
function Header() {
  const { t, i18n } = useTranslation();

  return (
    <header className="header" data-lang={i18n.language}>
      <div className="header-container">
        {/* Logo Section */}
        <div className="logo-section">
          <div className="logo-icon">
            {/* <Leaf size={28} /> */}
            <img src={Logo} alt="logo" />
          </div>

          <div className="logo-text">
            <h1 className="header-title">{t("header.title")}</h1>
            <p className="header-subtitle">{t("header.subtitle")}</p>
          </div>
        </div>

        {/* Navigation + Language Switcher */}
        <nav className="nav-links">
        
             <Link to="/" className="nav-link">Home</Link>
              <Link to="/about" className="nav-link">About</Link> 
 

          <div className="language-switcher">
            <button
              className={`lang-btn ${i18n.language === "en" ? "active" : ""}`}
              onClick={() => i18n.changeLanguage("en")}
            >
              EN
            </button>
            <button
              className={`lang-btn ${i18n.language === "ta" ? "active" : ""}`}
              onClick={() => i18n.changeLanguage("ta")}
            >
              தமிழ்
            </button>
          </div>
        </nav>
      </div>
    </header>
  );
}

export default Header;
