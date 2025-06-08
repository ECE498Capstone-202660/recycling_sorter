const API_URL = "http://192.168.0.185:8080";

export async function login(username: string, password: string) {
  const res = await fetch(`${API_URL}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  return res.json();
}

export async function register(username: string, password: string) {
  const res = await fetch(`${API_URL}/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
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