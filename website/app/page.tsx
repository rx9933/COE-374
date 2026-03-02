"use client";

import { useEffect, useState } from "react";
import { displayNameForInfraction, displayNameForThrowType, ThrowEvent, throwEventSchema, ThrowType } from "@/lib/schemas";
import { fetchThrowEvent } from "@/lib/api";
import { ImageGallery } from "@/app/components/image-gallery";
import { CircleField } from "./components/circle-field";
import { JavelinField } from "./components/javelin-field";

export default function Page() {
  const [currentThrow, setCurrentThrow] = useState<ThrowEvent | null>(null);
  const [status, setStatus] = useState<"waiting" | "received">("waiting");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const data = throwEventSchema.parse(await fetchThrowEvent());

        if (!cancelled) {
          setCurrentThrow(data);
          setStatus("received");
        }
      } catch (err: unknown) {
        console.error(err);
        if (!cancelled) setError("Failed to fetch/validate throw");
      } finally {
        if (!cancelled) setTimeout(poll, 1000);
      }
    }

    poll();

    return () => {
      cancelled = true;
    };
  }, []);


  return (
    <div className="min-h-screen bg-gray-900 text-white p-8 flex flex-col">
      {/* Title */}
      <h1 className="text-4xl font-extrabold mb-8 text-center">
        {status === "waiting" ? "Waiting for throw..." : "Throw received!"}
      </h1>

      {currentThrow && currentThrow.infractions.length === 0 && (
        <h2 className="text-2xl font-bold mb-4 text-center">
          Distance: {currentThrow.distance.toFixed(2)}m
        </h2>
      )}

      <h3 className="text-xl font-semibold mb-2 text-center">
        {currentThrow ? `Throw type: ${displayNameForThrowType(currentThrow.throwType)}` : "No throw data yet"}
      </h3>

      {/* Error */}
      {error && <p className="text-red-500 mb-4">{error}</p>}

      {/* Infractions */}
      {(currentThrow?.infractions?.length ?? 0) > 0 && (
        <div className="bg-red-800 border border-red-600 text-red-300 px-4 py-2 rounded mb-6">
          <strong>Infractions detected:</strong>
          <ul className="list-disc ml-5">
            {currentThrow!.infractions.map((i, idx) => (
              <li key={idx}>
                {currentThrow?.throwType === ThrowType.JAVELIN ? 'Foul Throw' : displayNameForInfraction(i.type)} (confidence: {(i.confidence * 100).toFixed(0)}%)
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Main grid: field + images */}
      <div className="grid grid-cols-12 flex-1 gap-x-8 items-start h-full">
        {/* Field */}
        <div className="col-span-9 flex justify-center">
          {currentThrow ? (
            currentThrow.throwType === ThrowType.JAVELIN ? (
                <JavelinField
                  landingPoint={currentThrow.landing_point}
                  infractions={currentThrow.infractions.map((i) => i.type)}
                />
              ) : (
                <CircleField
                  throwType={currentThrow.throwType}
                  landingPoint={currentThrow.landing_point}
                  infractions={currentThrow.infractions.map((i) => i.type)}
                />
              )
          ) : (
            <p className="text-gray-400">Waiting for throw...</p>
          )}
        </div>

        {/* Image gallery */}
        <div className="col-span-3 flex justify-end h-full">
          <ImageGallery images={currentThrow?.images ?? []} />
        </div>
      </div>
    </div>
  );
}
