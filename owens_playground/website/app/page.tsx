"use client";

import { useState } from "react";
import { displayNameForInfraction, displayNameForThrowType, ThrowEvent, throwEventSchema, ThrowType } from "@/lib/schemas";
import { fetchThrowEvent } from "@/lib/api";
import { ImageGallery } from "@/app/components/image-gallery";
import { CircleField } from "./components/circle-field";
import { JavelinField } from "./components/javelin-field";

const THROW_TYPE_OPTIONS = [
  ThrowType.DISCUS,
  ThrowType.HAMMER,
  ThrowType.JAVELIN,
  ThrowType.SHOTPUT,
];

const spinKeyframes = `@keyframes spin { to { transform: rotate(360deg); } }`;

export default function Page() {
  const [currentThrow, setCurrentThrow] = useState<ThrowEvent | null>(null);
  const [status, setStatus] = useState<"waiting" | "received">("waiting");
  const [error, setError] = useState<string | null>(null);
  const [selectedThrowType, setSelectedThrowType] = useState<ThrowType>(ThrowType.SHOTPUT);
  const [isLoadingCurrentThrow, setIsLoadingCurrentThrow] = useState(false);


  return (
    <div className="min-h-screen bg-gray-900 text-white p-8 flex flex-col">
      {/* Title */}
      <h1 className="text-4xl font-extrabold mb-8 text-center">
        {status === "waiting" ? "Waiting for throw..." : "Throw received!"}
      </h1>

      {/* Throw type dropdown */}
      <div className="flex justify-center mb-6">
        <label className="flex items-center gap-3 text-lg font-semibold">
          Throw Type:
          <select
            value={selectedThrowType}
            onChange={async (e) => {
              const newType = e.target.value as ThrowType;
              setSelectedThrowType(newType);
              await fetch("/throw-type", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ throwType: newType }),
              });
            }}
            className="bg-gray-800 border border-gray-600 text-white rounded px-3 py-2 text-base font-normal focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {THROW_TYPE_OPTIONS.map((type) => (
              <option key={type} value={type}>
                {displayNameForThrowType(type)}
              </option>
            ))}
          </select>
        </label>
        <button
          onClick={async () => {
            setIsLoadingCurrentThrow(true);
            try {
              const data = throwEventSchema.parse(await fetchThrowEvent(selectedThrowType));
              setCurrentThrow(data);
              setStatus("received");
            } catch (err: unknown) {
              console.error(err);
              setError("Failed to fetch/validate throw");
            } finally {
              setIsLoadingCurrentThrow(false);
            }
          }}
          disabled={isLoadingCurrentThrow}
          className={`bg-green-600 text-white px-4 py-2 rounded ml-6 flex items-center gap-2 ${isLoadingCurrentThrow ? "opacity-60 cursor-not-allowed" : "hover:bg-green-500 cursor-pointer"}`}
        >
          {isLoadingCurrentThrow && (
            <span
              style={{
                width: 16,
                height: 16,
                borderRadius: "50%",
                border: "2px solid white",
                borderTopColor: "transparent",
                display: "inline-block",
                animation: "spin 0.75s linear infinite",
              }}
            />
          )}
          {isLoadingCurrentThrow ? "Working..." : "Analyze Throw"}
        </button>
      </div>

      <style>{spinKeyframes}</style>

      {currentThrow && currentThrow.infractions.length === 0 && (
        <h2 className="text-2xl font-bold mb-4 text-center">
          Distance: {currentThrow.distance.toFixed(2)}m
        </h2>
      )}

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
            selectedThrowType === ThrowType.JAVELIN ? (
                <JavelinField
                  landingPoint={currentThrow.landing_point}
                  infractions={currentThrow.infractions.map((i) => i.type)}
                />
              ) : (
                <CircleField
                  throwType={selectedThrowType}
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
