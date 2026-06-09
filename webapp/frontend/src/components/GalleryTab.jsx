import { useEffect, useState } from "react";
import { getGallery, predictGallery } from "../api.js";

export default function GalleryTab({ onResult, setLoading, setError }) {
  const [images, setImages] = useState([]);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    getGallery()
      .then((d) => setImages(d.images))
      .catch((e) => setError(e.message));
  }, [setError]);

  const choose = async (img) => {
    setSelected(`${img.split}/${img.name}`);
    setLoading(true);
    setError(null);
    try {
      const res = await predictGallery(img.split, img.name);
      onResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="gallery">
      <p className="gallery-help">
        Pick a GlaS test image (these have ground truth, so real F1 / Dice / Hausdorff are shown):
      </p>
      <div className="thumb-grid">
        {images.map((img) => (
          <button
            key={`${img.split}/${img.name}`}
            className={selected === `${img.split}/${img.name}` ? "thumb selected" : "thumb"}
            onClick={() => choose(img)}
            title={`${img.part} · ${img.name}`}
          >
            <img src={img.thumb} alt={img.name} />
            <span className="thumb-label">{img.part}</span>
          </button>
        ))}
        {images.length === 0 && <p className="hint">Loading gallery…</p>}
      </div>
    </div>
  );
}
