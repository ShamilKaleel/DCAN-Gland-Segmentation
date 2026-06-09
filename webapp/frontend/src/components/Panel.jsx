export default function Panel({ title, src, subtitle }) {
  if (!src) return null;
  return (
    <figure className="panel">
      <img src={src} alt={title} />
      <figcaption>
        <strong>{title}</strong>
        {subtitle && <span className="panel-sub">{subtitle}</span>}
      </figcaption>
    </figure>
  );
}
