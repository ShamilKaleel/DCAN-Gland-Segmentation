export default function MetricCard({ stats, metrics, source }) {
  return (
    <div className="metric-card">
      <div className="metric-head">
        <h3>Result</h3>
        {source && <span className="source">{source}</span>}
      </div>

      <div className="metric-row">
        <div className="metric">
          <span className="metric-value">{stats?.n_glands ?? "–"}</span>
          <span className="metric-label">glands detected</span>
        </div>
        <div className="metric">
          <span className="metric-value">{stats?.runtime_s ?? "–"}s</span>
          <span className="metric-label">inference time</span>
        </div>
        {metrics && (
          <div className="metric">
            <span className="metric-value">{metrics.gt_glands}</span>
            <span className="metric-label">ground-truth glands</span>
          </div>
        )}
      </div>

      {metrics ? (
        <div className="metric-row metrics-accuracy">
          <div className="metric">
            <span className="metric-value">{metrics.F1?.toFixed?.(3) ?? metrics.F1}</span>
            <span className="metric-label">Detection F1 ↑</span>
          </div>
          <div className="metric">
            <span className="metric-value">{metrics.Dice_obj?.toFixed?.(3) ?? metrics.Dice_obj}</span>
            <span className="metric-label">Object-Dice ↑</span>
          </div>
          <div className="metric">
            <span className="metric-value">{metrics.Hausdorff_obj?.toFixed?.(1) ?? metrics.Hausdorff_obj}</span>
            <span className="metric-label">Object-Hausdorff ↓</span>
          </div>
        </div>
      ) : (
        <p className="hint">
          Uploaded image has no ground truth, so accuracy metrics aren't computed — use the
          test gallery to see real F1 / Dice / Hausdorff.
        </p>
      )}
    </div>
  );
}
