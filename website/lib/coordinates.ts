export function globalToSvgCoord(
  x: number,
  y: number,
  scale: number,
  circleXPercent: number,
  fieldWidth: number,
  fieldHeight: number,
): [number, number] {
  const centerX = circleXPercent * fieldWidth * scale;
  const centerY = (fieldHeight / 2) * scale;

  const px = x * scale;
  const py = y * scale;

  const svgX = centerX + px;
  const svgY = centerY - py;

  return [svgX, svgY];
}
