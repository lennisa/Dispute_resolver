// hooks/useTilt.js
// Drives the "leans toward the cursor" effect on cards: hover the left half
// and the card tilts back on that side, hover the right half and it tilts
// the other way. Pure CSS transform driven by mouse position, no library.

import { useRef, useState, useCallback } from "react";

export default function useTilt({ maxDeg = 5, perspective = 1200 } = {}) {
  const ref = useRef(null);
  const [style, setStyle] = useState({
    transform: `perspective(${perspective}px) rotateY(0deg) rotateX(0deg)`,
    transition: "transform 400ms cubic-bezier(0.22, 1, 0.36, 1)",
  });

  const onMouseMove = useCallback(
    (e) => {
      const el = ref.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const relX = (e.clientX - rect.left) / rect.width; // 0 (left) .. 1 (right)
      const relY = (e.clientY - rect.top) / rect.height; // 0 (top) .. 1 (bottom)
      const rotateY = (relX - 0.5) * 2 * maxDeg; // left half -> negative, right half -> positive
      const rotateX = (0.5 - relY) * 2 * (maxDeg * 0.5);
      setStyle({
        transform: `perspective(${perspective}px) rotateY(${rotateY}deg) rotateX(${rotateX}deg)`,
        transition: "transform 60ms linear",
      });
    },
    [maxDeg, perspective]
  );

  const onMouseLeave = useCallback(() => {
    setStyle({
      transform: `perspective(${perspective}px) rotateY(0deg) rotateX(0deg)`,
      transition: "transform 450ms cubic-bezier(0.22, 1, 0.36, 1)",
    });
  }, [perspective]);

  return { ref, style, onMouseMove, onMouseLeave };
}
