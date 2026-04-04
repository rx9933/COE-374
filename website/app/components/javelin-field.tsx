import { globalToSvgCoord } from "@/lib/coordinates";
import { circleFieldDimsForThrowType, Infraction, ThrowType } from "@/lib/schemas";

interface JavelinFieldProps {
  landingPoint?: [number, number];
  infractions: Infraction[];
}

const SCALE = 50; // 1 m = 100 px

export const JavelinField: React.FC<JavelinFieldProps> = ({
  landingPoint,
  infractions,
}) => {
  const { circleDiameter, fieldLength } = circleFieldDimsForThrowType(ThrowType.JAVELIN);
  const arcRadiusMeters = circleDiameter / 2.0;
  const arcCenterToStraightVerticalThrowLinesMeters = 7.746;
  const runwayToArcCenterMeters = 26;
  const overallWidthMeters = runwayToArcCenterMeters + arcCenterToStraightVerticalThrowLinesMeters + fieldLength + 2.0 // meter of padding on left and right
  const overallHeightMeters = 2.0 * (arcRadiusMeters + fieldLength) * Math.sin(28.96 / 2.0 * Math.PI / 180) + 2.0; // meter of padding on top and bottom
  const circleXPercent = (1.0 + runwayToArcCenterMeters ) / overallWidthMeters; // 1m padding
  const circleXMeters = 0.0;
  const circleYMeters = 0.0;
  const circleStrokeWidthMeters = 0.07;
  const lineStrokeWidthMeters = 0.05;
  const lowerRunwayStartCoord = [circleXMeters - runwayToArcCenterMeters, circleYMeters - 2.0];
  const lowerRunwayEndCoord = [circleXMeters + arcCenterToStraightVerticalThrowLinesMeters, circleYMeters - 2.0];
  const upperRunwayStartCoord = [circleXMeters - runwayToArcCenterMeters, circleYMeters + 2.0];
  const upperRunwayEndCoord = [circleXMeters + arcCenterToStraightVerticalThrowLinesMeters, circleYMeters + 2.0];
  const arcStartCoord = [arcCenterToStraightVerticalThrowLinesMeters, -2.0];
  const arcEndCoord = [arcCenterToStraightVerticalThrowLinesMeters, 2.0];
  const lowerSectorStartCoord = [...arcStartCoord];
  const lowerSectorEndCoord = [(arcRadiusMeters + fieldLength) * Math.cos(28.96 / 2.0 * Math.PI / 180), (arcRadiusMeters + fieldLength) * -Math.sin(28.96 / 2.0 * Math.PI / 180)];
  const upperSectorStartCoord = [...arcEndCoord];
  const upperSectorEndCoord = [(arcRadiusMeters + fieldLength) * Math.cos(28.96 / 2.0 * Math.PI / 180), (arcRadiusMeters + fieldLength) * Math.sin(28.96 / 2.0 * Math.PI / 180)];

  const [arcStartPxX, arcStartPxY] = globalToSvgCoord(arcStartCoord[0], arcStartCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const [arcEndPxX, arcEndPxY] = globalToSvgCoord(arcEndCoord[0], arcEndCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const arcRadiusPx = arcRadiusMeters * SCALE;
  const arcStrokeWidthPx = circleStrokeWidthMeters * SCALE;

  const lowerRunwayStartPx = globalToSvgCoord(lowerRunwayStartCoord[0], lowerRunwayStartCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const lowerRunwayEndPx = globalToSvgCoord(lowerRunwayEndCoord[0], lowerRunwayEndCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const upperRunwayStartPx = globalToSvgCoord(upperRunwayStartCoord[0], upperRunwayStartCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const upperRunwayEndPx = globalToSvgCoord(upperRunwayEndCoord[0], upperRunwayEndCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);

  const lowerSectorStartPx = globalToSvgCoord(lowerSectorStartCoord[0], lowerSectorStartCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const lowerSectorEndPx = globalToSvgCoord(lowerSectorEndCoord[0], lowerSectorEndCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const upperSectorStartPx = globalToSvgCoord(upperSectorStartCoord[0], upperSectorStartCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const upperSectorEndPx = globalToSvgCoord(upperSectorEndCoord[0], upperSectorEndCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const lineStrokeWidthPx = lineStrokeWidthMeters * SCALE;
  const landingPointPx = landingPoint ? globalToSvgCoord(landingPoint[0], landingPoint[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters) : null;

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${overallWidthMeters * SCALE} ${overallHeightMeters * SCALE}`}
      preserveAspectRatio="xMidYMid meet"
      className="bg-green-500 rounded-lg"
    >
      {/* Arc */}
      <path
        d={`
          M ${arcEndPxX} ${arcEndPxY}
          A ${arcRadiusPx} ${arcRadiusPx} 0 0 1 ${arcStartPxX} ${arcStartPxY}
        `}
        fill="none"
        stroke={infractions.includes(Infraction.CIRCLE) ? "#ff0000" : "#ffffff"}
        strokeWidth={arcStrokeWidthPx}
      />

      {/* Lower Runway Line */}
      <path
        d={`M ${lowerRunwayStartPx[0]} ${lowerRunwayStartPx[1]} L ${lowerRunwayEndPx[0]} ${lowerRunwayEndPx[1]}`}
        stroke="#ffffff"
        strokeWidth={lineStrokeWidthPx}
        fill="none"
      />

      {/* Upper Runway Line */}
      <path
        d={`M ${upperRunwayStartPx[0]} ${upperRunwayStartPx[1]} L ${upperRunwayEndPx[0]} ${upperRunwayEndPx[1]}`}
        stroke="#ffffff"
        strokeWidth={lineStrokeWidthPx}
        fill="none"
      />

      {/* Upper sector line */}
      <path
        d={`M ${upperSectorStartPx[0]} ${upperSectorStartPx[1]} L ${upperSectorEndPx[0]} ${upperSectorEndPx[1]}`}
        stroke={infractions.includes(Infraction.LEFT_SECTOR) ? "#ff0000" : "#ffffff"}
        strokeWidth={lineStrokeWidthPx}
        fill="none"
      />

      {/* Lower sector line */}
      <path
        d={`M ${lowerSectorStartPx[0]} ${lowerSectorStartPx[1]} L ${lowerSectorEndPx[0]} ${lowerSectorEndPx[1]}`}
        stroke={infractions.includes(Infraction.RIGHT_SECTOR) ? "#ff0000" : "#ffffff"}
        strokeWidth={lineStrokeWidthPx}
        fill="none"
      />

      {/* Landing dot */}
      {landingPointPx && (
        <circle
          cx={landingPointPx[0]}
          cy={landingPointPx[1]}
          r={.08 * SCALE * 4} // shit too small fuck scaling
          fill={infractions.length ? "#ff0000" : "#000000"}
        />
      )}
    </svg>
  );
};
