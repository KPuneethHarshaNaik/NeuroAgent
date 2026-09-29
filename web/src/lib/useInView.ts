import { useEffect, useRef, useState } from "react";

/**
 * Fires once, when the element first enters the viewport. Used for the two deliberate
 * scroll-triggered moments (the stage sequence drawing, the verdict stamping) and nothing else --
 * the rest of the page is static, on purpose.
 */
export function useInView<T extends HTMLElement>(rootMargin = "-15% 0px -10% 0px") {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element || inView) return;
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.disconnect();
        }
      },
      { rootMargin, threshold: 0.1 },
    );
    observer.observe(element);
    // Fail open: if the observer callback is starved, anything already on screen still plays its
    // one moment rather than waiting for an event that may not arrive.
    const timer = window.setTimeout(() => {
      if (element.getBoundingClientRect().top < window.innerHeight) setInView(true);
    }, 1200);
    return () => {
      observer.disconnect();
      window.clearTimeout(timer);
    };
  }, [inView, rootMargin]);

  return { ref, inView };
}
