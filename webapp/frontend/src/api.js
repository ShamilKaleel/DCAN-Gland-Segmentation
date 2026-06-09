// Thin fetch wrappers around the FastAPI backend.

export async function getInfo() {
  const r = await fetch("/api/info");
  if (!r.ok) throw new Error("Failed to load model info");
  return r.json();
}

export async function getGallery() {
  const r = await fetch("/api/gallery");
  if (!r.ok) throw new Error("Failed to load gallery");
  return r.json();
}

export async function predictUpload(file) {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch("/api/predict", { method: "POST", body: form });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || "Prediction failed");
  }
  return r.json();
}

export async function predictGallery(split, name) {
  const r = await fetch("/api/predict_gallery", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ split, name }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || "Prediction failed");
  }
  return r.json();
}
