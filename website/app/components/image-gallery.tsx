"use client";

import { Dialog, DialogPanel, DialogBackdrop } from '@headlessui/react'
import React, { useState } from "react";
import Image from "next/image";

interface ImageGalleryProps {
  images: string[];
}

export const ImageGallery: React.FC<ImageGalleryProps> = ({ images }) => {
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <>
      {/* Gallery */}
      <div className="w-full max-w-full h-full overflow-y-auto px-2 pb-2">
        {images.map((url, i) => (
          <div key={i} className="mb-3">
            <Image
              src={url}
              alt={`Throw image ${i}`}
              width={0}
              height={0}
              sizes="(max-width: 768px) 100vw, 25vw"
              className="w-full h-auto rounded cursor-pointer hover:brightness-70 transition duration-200"
              onClick={() => setSelected(url)}
            />
          </div>
        ))}
      </div>
      <Dialog open={!!selected} onClose={() => setSelected(null)} className="relative z-50">
        {/* Backdrop */}
        <DialogBackdrop className="fixed inset-0 bg-black/60" />
        {/* Centered modal */}
        <div className="fixed inset-0 flex items-center justify-center">
          <DialogPanel className="relative w-full max-w-[90vw] max-h-[90vh] rounded overflow-auto">
            {/* Close button */}
            <button
              onClick={() => setSelected(null)}
              className="absolute top-2 right-2 z-50 w-8 h-8 rounded-full bg-black/50 text-white flex items-center justify-center hover:bg-black/70 cursor-pointer transition duration-200"
            >
              ×
            </button>
            {selected && (<Image
              src={selected}
              alt="Selected throw image"
              width={0}
              height={0}
              className="w-full max-h-[90vh] h-auto"
              sizes="90vw"
              priority
            />)}
          </DialogPanel>
        </div>
      </Dialog>
    </>
  );
};
