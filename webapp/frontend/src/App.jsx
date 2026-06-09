import { useEffect, useState } from "react";
import { getInfo } from "./api.js";
import UploadTab from "./components/UploadTab.jsx";
import GalleryTab from "./components/GalleryTab.jsx";
import ResultsGrid from "./components/ResultsGrid.jsx";
import MetricCard from "./components/MetricCard.jsx";

export default function App() {
  const [info, setInfo] = useState(null);
  const [tab, setTab] = useState("gallery");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getInfo().then(setInfo).catch(() => {});
  }, []);

  const handleResult = (res) => {
    setResult(res);
    setError(null);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="title">
          <h1>DCAN · Gland Segmentation</h1>
          <p className="subtitle">
            Deep Contour-Aware Network — separating touching glands in H&amp;E histology
          </p>
        </div>
        {info && (
          <div className="badges">
            <span className="badge">{info.backbone}</span>
            <span className="badge badge-accent">device: {info.device}</span>
            <span className="badge badge-metric">
              CV Dice {info.benchmarks.cv_dice} · Part A Dice {info.benchmarks.partA_dice}
            </span>
          </div>
        )}
      </header>

      <nav className="tabs">
        <button className={tab === "gallery" ? "tab active" : "tab"} onClick={() => setTab("gallery")}>
          Test gallery (with metrics)
        </button>
        <button className={tab === "upload" ? "tab active" : "tab"} onClick={() => setTab("upload")}>
          Upload your image
        </button>
      </nav>

      <main className="main">
        <section className="controls">
          {tab === "gallery" ? (
            <GalleryTab onResult={handleResult} setLoading={setLoading} setError={setError} />
          ) : (
            <UploadTab onResult={handleResult} setLoading={setLoading} setError={setError} />
          )}
        </section>

        {error && <div className="error">⚠ {error}</div>}
        {loading && (
          <div className="loading">
            <div className="spinner" />
            <span>Running DCAN inference…</span>
          </div>
        )}

        {result && !loading && (
          <section className="results">
            <MetricCard stats={result.stats} metrics={result.metrics} source={result.source} />
            <ResultsGrid panels={result.panels} />
          </section>
        )}
      </main>

      <footer className="footer">
        Reproduction of Chen et al., <em>DCAN</em>, CVPR 2016 · GlaS @ MICCAI'2015
      </footer>
    </div>
  );
}
