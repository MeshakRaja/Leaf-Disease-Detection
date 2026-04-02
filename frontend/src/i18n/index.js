import i18n from "i18next";
import { initReactI18next } from "react-i18next"; // 👈 THIS WAS MISSING!
import LanguageDetector from "i18next-browser-languagedetector";

import en from "./languages/en.json";
import ta from "./languages/ta.json";

i18n
  .use(initReactI18next) // 👈 CRITICAL LINE
  .use(LanguageDetector)
  .init({
    resources: {
      en: { translation: en },
      ta: { translation: ta },
    },
    lng: "en", // Default English
    fallbackLng: "en",
    interpolation: {
      escapeValue: false,
    },
  });

export default i18n;
