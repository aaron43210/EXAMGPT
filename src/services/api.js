import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: `${API_URL}/api/v1`,
});

// Attach JWT token to every request
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Handle 401 responses (expired token)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ── Auth ─────────────────────────────────────────────────
export const authAPI = {
  login: (username, password) => {
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);
    return api.post('/auth/login', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    });
  },
  register: (data) => api.post('/auth/register', data),
  me: () => api.get('/auth/me'),
};

// ── Courses ──────────────────────────────────────────────
export const coursesAPI = {
  list: () => api.get('/courses/'),
  get: (id) => api.get(`/courses/${id}`),
  create: (data) => api.post('/courses/', data),
  delete: (id) => api.delete(`/courses/${id}`),
};

// ── Documents ────────────────────────────────────────────
export const documentsAPI = {
  list: (courseId) => api.get(`/courses/${courseId}/documents`),
  upload: (courseId, file, docType = 'notes') => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('doc_type', docType);
    return api.post(`/courses/${courseId}/documents`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
};

// ── Query ────────────────────────────────────────────────
export const queryAPI = {
  ask: (courseId, query) => api.post(`/courses/${courseId}/query`, { query }),
  history: (courseId) => api.get(`/courses/${courseId}/chat-history`),
};

// ── Study Plan ───────────────────────────────────────────
export const studyPlanAPI = {
  generate: (courseId, data) => api.post(`/courses/${courseId}/study-plan`, data),
  get: (courseId) => api.get(`/courses/${courseId}/study-plan`),
};

export default api;
