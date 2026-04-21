// iNaturalist integration: OAuth session lives on the backend (HTTP-only cookie).

export interface iNaturalistObservation {
  species: string;
  scientificName: string;
  confidence: number;
  count: number;
  imageFile: File;
  location?: {
    latitude: number;
    longitude: number;
  };
  notes?: string;
}

export interface iNaturalistAuthStatus {
  connected: boolean;
  expiresAt: number | null;
}

/**
 * Base URL for credentialed API calls (OAuth cookie + uploads).
 * - If `VITE_API_BASE_URL` is set at build time, use it (e.g. local dev: API on another port).
 * - Otherwise in the browser, use same origin (empty string → paths like `/api/...`), which
 *   matches production: Nginx serves the SPA and proxies `/api` to the backend.
 */
function getBackendBaseURL(): string | null {
  if (typeof import.meta !== "undefined" && (import.meta as any).env) {
    const raw = (import.meta as any).env.VITE_API_BASE_URL as string | undefined;
    if (raw != null && String(raw).trim()) {
      return String(raw).replace(/\/$/, "");
    }
  }
  if (typeof window !== "undefined") {
    return "";
  }
  return null;
}

class iNaturalistService {
  /** True when we can reach the API (explicit base URL or same-origin in the browser). */
  isBackendConfigured(): boolean {
    return getBackendBaseURL() !== null;
  }

  /** GET /api/auth/inat/status — sends session cookie. */
  async getAuthStatus(): Promise<iNaturalistAuthStatus> {
    const base = getBackendBaseURL();
    if (base === null) {
      return { connected: false, expiresAt: null };
    }
    const res = await fetch(`${base}/api/auth/inat/status`, {
      method: "GET",
      credentials: "include",
    });
    if (!res.ok) {
      console.warn("iNaturalist status failed:", res.status);
      return { connected: false, expiresAt: null };
    }
    const data = await res.json();
    return {
      connected: Boolean(data.connected),
      expiresAt: typeof data.expiresAt === "number" ? data.expiresAt : null,
    };
  }

  /** Full-page navigation to start OAuth (sets session cookie on redirect). */
  connectAccount(): void {
    const base = getBackendBaseURL();
    if (base === null) {
      throw new Error("Cannot connect to iNaturalist outside a browser session.");
    }
    window.location.href = `${base}/api/auth/inat/login`;
  }

  /** Clear server-side iNat tokens and session cookie. */
  async disconnect(): Promise<void> {
    const base = getBackendBaseURL();
    if (base === null) return;
    await fetch(`${base}/api/auth/inat/logout`, {
      method: "POST",
      credentials: "include",
    });
  }

  /**
   * POST /api/observations — session cookie identifies the user; do not send iNat tokens from JS.
   */
  async uploadObservation(observation: iNaturalistObservation): Promise<boolean> {
    const base = getBackendBaseURL();
    if (base === null) {
      throw new Error("Cannot upload observation outside a browser session.");
    }

    const formData = new FormData();
    formData.append("file", observation.imageFile);
    formData.append("species", observation.species);
    formData.append("scientificName", observation.scientificName);
    formData.append("confidence", observation.confidence.toString());
    formData.append("count", observation.count.toString());

    if (observation.location) {
      formData.append("latitude", observation.location.latitude.toString());
      formData.append("longitude", observation.location.longitude.toString());
    }
    if (observation.notes) {
      formData.append("notes", observation.notes);
    }

    const res = await fetch(`${base}/api/observations`, {
      method: "POST",
      credentials: "include",
      body: formData,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Upload failed (${res.status}): ${text}`);
    }
    await res.json();
    return true;
  }

  async getUserObservations(): Promise<unknown[]> {
    const base = getBackendBaseURL();
    if (base === null) return [];
    const res = await fetch(`${base}/api/observations`, { credentials: "include" });
    if (!res.ok) throw new Error("Failed to fetch observations");
    return await res.json();
  }

  async searchSpecies(query: string): Promise<unknown[]> {
    const base = getBackendBaseURL();
    if (base === null) return [];
    const url = new URL(`${base}/api/species/search`, window.location.origin);
    url.searchParams.set("q", query);
    const res = await fetch(url.toString());
    if (!res.ok) throw new Error("Search failed");
    return await res.json();
  }

  async getSpeciesDetails(scientificName: string): Promise<unknown> {
    const base = getBackendBaseURL();
    if (base === null) return null;
    const res = await fetch(
      `${base}/api/species/${encodeURIComponent(scientificName)}`,
    );
    if (!res.ok) throw new Error("Details fetch failed");
    return await res.json();
  }
}

export const iNaturalistAPI = new iNaturalistService();

export async function getCurrentLocation(): Promise<{ latitude: number; longitude: number } | null> {
  return new Promise((resolve) => {
    if (!navigator.geolocation) {
      resolve(null);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        resolve({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
        });
      },
      (error) => {
        console.warn("Failed to get location:", error);
        resolve(null);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 300000,
      },
    );
  });
}
