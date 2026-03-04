const API_URL = process.env.EXPO_PUBLIC_API_URL ?? "http://localhost:8080";

type AuthErrorHandler = (status: number) => void;
let authErrorHandler: AuthErrorHandler | null = null;

export function setAuthErrorHandler(handler: AuthErrorHandler | null) {
  authErrorHandler = handler;
}

async function handleAuthError(res: Response) {
  if (res.status === 401) {
    authErrorHandler?.(res.status);
  }
}

export async function login(username: string, password: string) {
  const fd = new FormData();
  fd.append("username", username);
  fd.append("password", password);
  const res = await fetch(`${API_URL}/auth/login`, { method: "POST", body: fd });
  await handleAuthError(res);
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

export async function register(
  username: string,
  password: string,
  payload: { email: string; first_name: string; last_name: string }
) {
  const res = await fetch(`${API_URL}/auth/register`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      username,
      password,
      email: payload.email,
      first_name: payload.first_name,
      last_name: payload.last_name,
    }),
  });
  await handleAuthError(res);
  return res.json();
}

export async function getLatestResults(token: string) {
  const res = await fetch(`${API_URL}/classification/latest`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  await handleAuthError(res);
  return res.json();
}

export async function getHistory(token: string) {
  const res = await fetch(`${API_URL}/classification/history`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  await handleAuthError(res);
  return res.json();
}

export async function getUserMe(token: string) {
  const res = await fetch(`${API_URL}/user/me`, {
    headers: { Authorization: `Bearer ${token}`, "X-Activity": "1" },
  });
  await handleAuthError(res);
  return res.json();
}

export async function updateUserMe(
  token: string,
  payload: { email?: string | null; first_name?: string | null; last_name?: string | null }
) {
  const res = await fetch(`${API_URL}/user/me`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      "X-Activity": "1",
    },
    body: JSON.stringify(payload),
  });
  await handleAuthError(res);
  return res.json();
}

export async function getSortingStatus(token: string) {
  const res = await fetch(`${API_URL}/kiosk/status`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  await handleAuthError(res);
  return res.json();
}

export async function startRecycling(token: string) {
  const res = await fetch(`${API_URL}/kiosk/start`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  await handleAuthError(res);
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}

export async function stopRecycling(token: string) {
  const res = await fetch(`${API_URL}/kiosk/stop`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  await handleAuthError(res);
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, data };
}
