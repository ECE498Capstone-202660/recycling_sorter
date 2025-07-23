import { useEffect, useState } from "react";
import { getLatestResults } from "../services/api";

export function useLatestResults() {
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    async function fetchResults() {
      try {
        const data = await getLatestResults();
        if (mounted) setResults(Array.isArray(data) ? data : []);
      } finally {
        if (mounted) setLoading(false);
      }
    }

    fetchResults();
    const id = setInterval(fetchResults, 8000); // refresh every 8 s
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  return { results, loading };
}
