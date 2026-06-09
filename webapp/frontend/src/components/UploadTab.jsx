import { useRef, useState } from "react";
import { predictUpload } from "../api.js";

export default function UploadTab({ onResult, setLoading, setError }) {
  const inputRef = useRef(null);
  const [fileName, setFileName] = useState(null);
  const [dragging, setDragging] = useState(false);

  const submit = async (file) => {
    if (!file) return;
    setFileName(file.name);
    setLoading(true);
    setError(null);
    try {
      const res = await predictUpload(file);
      onResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files?.[0]) submit(e.dataTransfer.files[0]);
  };

  return (
    <div
      className={dragging ? "dropzone dragging" : "dropzone"}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => submit(e.target.files?.[0])}
      />
      <div className="dropzone-inner">
        <div className="dropzone-icon">⬆</div>
        <p>Drag &amp; drop an H&amp;E histology image, or click to choose</p>
        {fileName && <p className="filename">{fileName}</p>}
        <p className="hint">No ground truth → glands are separated and counted (no accuracy metrics).</p>
      </div>
    </div>
  );
}
