const API_URL = "http://localhost:8080";

export async function login(username: string, password: string) {
  const fd = new FormData();
  fd.append("username", username);
  fd.append("password", password);
  const res = await fetch(`${API_URL}/auth/login`, { method: "POST", body: fd });
  return res.json();
}

export async function register(username: string, password: string) {
  const fd = new FormData();
  fd.append("username", username);
  fd.append("password", password);
  const res = await fetch(`${API_URL}/auth/register`, { method: "POST", body: fd });
  return res.json();
}

export async function getRebates(token: string) {
  const res = await fetch(`${API_URL}/rebates`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.json();
}

export async function createRebate(token: string, title: string, amount: number) {
  const res = await fetch(`${API_URL}/rebates`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ title, amount }),
  });
  return res.json();
}

export async function getLatestResults() {
  const res = await fetch(`${API_URL}/classification/latest`);
  return res.json();
}

export async function getHistory() {
  const res = await fetch(`${API_URL}/classification/history`);
  return res.json();
} 