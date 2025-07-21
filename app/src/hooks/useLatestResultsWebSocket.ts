import { useEffect, useRef } from 'react';

// WebSocket URL for latest classification results
const WEBSOCKET_URL = 'ws://localhost:8080/model/ws/latest-results'; // Update as needed

export function useLatestResultsWebSocket(
  setResults: (results: any[]) => void,
  setLoading: (loading: boolean) => void
) {
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    setLoading(true);
    ws.current = new WebSocket(WEBSOCKET_URL);

    ws.current.onopen = () => setLoading(false);

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setResults(Array.isArray(data) ? data : []);
      } catch {
        setResults([]);
      }
    };

    ws.current.onerror = () => setLoading(false);
    ws.current.onclose = () => setLoading(false);

    return () => {
      ws.current && ws.current.close();
    };
  }, [setResults, setLoading]);
} 