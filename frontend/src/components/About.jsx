// About.jsx
import React from "react";
import { Github, Brain, Smartphone, Zap } from "lucide-react";
import "./About.css";

const About = () => {
  return (
    <section className="about-page">
      <div className="about-container">
        {/* Mission */}
        <div className="about-card mission-card">
          <div className="card-icon">🎯</div>
          <h2>Our Mission</h2>
          <p>
            Empowering farmers with instant AI crop disease detection using
            YOLOv8 - from leaf photo to treatment in seconds.
          </p>
        </div>

        {/* Problem */}
        <div className="about-card problem-card">
          <div className="card-icon">⚠️</div>
          <h2>The Problem We Solve</h2>
          <p>
            80% of crop losses happen due to untreated diseases. Farmers wait
            days for lab results. We deliver instant diagnosis on their phone.
          </p>
        </div>

        {/* How It Started */}
        <div className="about-card story-card">
          <div className="card-icon">🌱</div>
          <h2>How It Started</h2>
          <p>Developed as our Final Year Project (2026)</p>
          <div className="team-list">
            <div className="team-member">
              <span className="emoji">🌱</span>
              <strong>Mohamed Shalik M</strong> - Full Stack Developer (React +
              FastAPI)
            </div>
            <div className="team-member">
              <span className="emoji">🚀</span>
              <strong>Meshak Raja A</strong> - AI/ML Engineer (YOLOv8 Model
              Training)
            </div>
            <div className="team-member">
              <span className="emoji">⚙️</span>
              <strong>Mohammad Sheik Myddin A</strong> - Backend + Database
            </div>
            <div className="team-member">
              <span className="emoji">🎨</span>
              <strong>Karthick S</strong> - UI/UX Design + Animations
            </div>
          </div>
          <p className="motivation">
            Witnessed farmers in tenkasi lose 40% crops to common leaf diseases
            like early blight, rust, and powdery mildew due to delayed
            diagnosis. Built this AI solution to identify ANY leaf disease
            instantly from smartphone photos.
          </p>
        </div>

        {/* Tech Stack */}
        <div className="about-card tech-card">
          <div className="card-icon">⚡</div>
          <h2>Our Tech</h2>
          <div className="tech-grid">
            <div className="tech-item">
              <Brain size={32} />
              <h3>YOLOv8 AI</h3>
              <p>98% accuracy</p>
            </div>
            <div className="tech-item">
              <Zap size={32} />
              <h3>React + FastAPI</h3>
              <p>Blazing fast</p>
            </div>
            <div className="tech-item">
              <Smartphone size={32} />
              <h3>Mobile First</h3>
              <p>Offline capable</p>
            </div>
            <div className="tech-item">
              <Github size={32} />
              <h3>Open Source</h3>
              <p>GitHub hosted</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
