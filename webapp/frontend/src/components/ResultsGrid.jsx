import Panel from "./Panel.jsx";

export default function ResultsGrid({ panels }) {
  if (!panels) return null;
  return (
    <div className="panel-grid">
      <Panel title="Input (H&E)" src={panels.original} />
      <Panel title="Gland separation" subtitle="each color = one detected gland" src={panels.separation} />
      {panels.ground_truth && (
        <Panel title="Ground truth" subtitle="expert annotation" src={panels.ground_truth} />
      )}
      <Panel title="Boundary overlay" subtitle="predicted gland outlines" src={panels.overlay} />
      <Panel title="Object probability" subtitle="DCAN object branch" src={panels.prob_obj} />
      <Panel title="Contour probability" subtitle="DCAN contour branch" src={panels.prob_cnt} />
    </div>
  );
}
