import { z } from "zod";

export enum Infraction {
  LEFT_SECTOR = "left_sector",
  RIGHT_SECTOR = "right_sector",
  CIRCLE = "circle",
}

export function displayNameForInfraction(infraction: Infraction) {
  switch (infraction) {
    case Infraction.LEFT_SECTOR:
    case Infraction.RIGHT_SECTOR:
      return "Sector Foul";
    case Infraction.CIRCLE:
      return "Circle Foul";
  }
}

export enum ThrowType {
  DISCUS = "discus",
  HAMMER = "hammer",
  JAVELIN = "javelin",
  SHOTPUT = "shotput",
}

export function circleFieldDimsForThrowType(throwType: ThrowType) {
  switch (throwType) {
    case ThrowType.DISCUS:
      return { circleDiameter: 2.5, fieldLength: 80 };
    case ThrowType.HAMMER:
      return { circleDiameter: 2.135, fieldLength: 90 };
    case ThrowType.SHOTPUT:
      return { circleDiameter: 2.135, fieldLength: 30 };
    case ThrowType.JAVELIN:
      return { circleDiameter: 16, fieldLength: 100 };
  }
}

export const infractionSchema = z
  .object({
    type: z.enum(Infraction),
    confidence: z.number().min(0).max(1),
  })
  .strict();

export const throwEventSchema = z
  .object({
    throwId: z.uuid(),
    timestamp: z.string().refine((val) => !isNaN(Date.parse(val)), {
      message: "Invalid timestamp",
    }),
    throwType: z.enum(ThrowType),
    distance: z.number().nonnegative(),
    infractions: z.array(infractionSchema),
    images: z.array(z.url()),
    landing_point: z.tuple([z.number(), z.number()]).optional(),
  })
  .strict();

export type ThrowEvent = z.infer<typeof throwEventSchema>;
