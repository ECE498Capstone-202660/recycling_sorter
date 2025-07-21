// const API_URL = "http://xxx.xxx.xxx.xxx:8080";
const API_URL = "http://localhost:8080";

export async function login(username: string, password: string) {
  const formData = new FormData();
  formData.append('username', username);
  formData.append('password', password);
  const res = await fetch(`${API_URL}/auth/login`, {
    method: "POST",
    body: formData,
  });
  return res.json();
}

export async function register(username: string, password: string) {
  const formData = new FormData();
  formData.append('username', username);
  formData.append('password', password);
  const res = await fetch(`${API_URL}/auth/register`, {
    method: "POST",
    body: formData,
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

export async function getLatestResult() {
  const res = await fetch(`${API_URL}/model/latest-result`);
  return res.json();
} 