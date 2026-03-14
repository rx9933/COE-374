import { globalToSvgCoord } from "@/lib/coordinates";
import { circleFieldDimsForThrowType, Infraction, ThrowType } from "@/lib/schemas";

interface CircleFieldProps {
  throwType: ThrowType;
  landingPoint?: [number, number];
  infractions: Infraction[];
}

const SCALE = 50; // 1 m = 100 px

export const CircleField: React.FC<CircleFieldProps> = ({
  throwType,
  landingPoint,
  infractions,
}) => {
  const { circleDiameter, fieldLength } = circleFieldDimsForThrowType(throwType);
  const overallWidthMeters = fieldLength + circleDiameter + 2.0; // meter of padding on each side
  const overallHeightMeters = 2.0 * fieldLength * Math.sin(34.92 / 2.0 * Math.PI / 180) + 2.0; // meter of padding on top and bottom
  const circleXPercent = (circleDiameter / 2.0 + 1.0) / overallWidthMeters;
  const circleXMeters = 0.0;
  const circleYMeters = 0.0;
  const circleRadiusMeters = circleDiameter / 2.0;
  const circleStrokeWidthMeters = 0.05;
  const lineStrokeWidthMeters = 0.05;
  const upperLineStartCoord = [circleRadiusMeters * Math.cos(34.92 / 2.0 * Math.PI / 180), circleRadiusMeters * Math.sin(34.92 / 2.0 * Math.PI / 180)];
  const upperLineEndCoord = [(circleRadiusMeters + fieldLength) * Math.cos(34.92 / 2.0 * Math.PI / 180), (circleRadiusMeters + fieldLength) * Math.sin(34.92 / 2.0 * Math.PI / 180)];
  const lowerLineStartCoord = [circleRadiusMeters * Math.cos(34.92 / 2.0 * Math.PI / 180), circleRadiusMeters * -Math.sin(34.92 / 2.0 * Math.PI / 180)];
  const lowerLineEndCoord = [(circleRadiusMeters + fieldLength) * Math.cos(34.92 / 2.0 * Math.PI / 180), (circleRadiusMeters + fieldLength) * -Math.sin(34.92 / 2.0 * Math.PI / 180)];

  const [circleXPx, circleYPx] = globalToSvgCoord(circleXMeters, circleYMeters, SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const circleRadiusPx = circleRadiusMeters * SCALE;
  const circleStrokeWidthPx = circleStrokeWidthMeters * SCALE;
  const upperLineStartPx = globalToSvgCoord(upperLineStartCoord[0], upperLineStartCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const upperLineEndPx = globalToSvgCoord(upperLineEndCoord[0], upperLineEndCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const lowerLineStartPx = globalToSvgCoord(lowerLineStartCoord[0], lowerLineStartCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const lowerLineEndPx = globalToSvgCoord(lowerLineEndCoord[0], lowerLineEndCoord[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters);
  const lineStrokeWidthPx = lineStrokeWidthMeters * SCALE;
  const landingPointPx = landingPoint ? globalToSvgCoord(landingPoint[0], landingPoint[1], SCALE, circleXPercent, overallWidthMeters, overallHeightMeters) : null;

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${overallWidthMeters * SCALE} ${overallHeightMeters * SCALE}`}
      preserveAspectRatio="xMidYMid meet"
      className="bg-green-500 rounded-lg"
    >
      {/* Circle */}
      <circle
        cx={circleXPx}
        cy={circleYPx}
        r={circleRadiusPx}
        fill="#none"
        stroke={infractions.includes(Infraction.CIRCLE) ? "#ff0000" : "#ffffff"}
        strokeWidth={circleStrokeWidthPx}
      />

      {/* Upper sector line */}
      <path
        d={`M ${upperLineStartPx[0]} ${upperLineStartPx[1]} L ${upperLineEndPx[0]} ${upperLineEndPx[1]}`}
        stroke={infractions.includes(Infraction.LEFT_SECTOR) ? "#ff0000" : "#ffffff"}
        strokeWidth={lineStrokeWidthPx}
        fill="none"
      />

      {/* Lower sector line */}
      <path
        d={`M ${lowerLineStartPx[0]} ${lowerLineStartPx[1]} L ${lowerLineEndPx[0]} ${lowerLineEndPx[1]}`}
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
