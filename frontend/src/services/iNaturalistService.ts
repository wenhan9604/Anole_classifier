// iNaturalist API Service
// This service handles integration with iNaturalist for uploading observations

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

export interface iNaturalistAuth {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

class iNaturalistService {
  // Optional backend base URL for local FastAPI proxy (set VITE_API_BASE_URL)
  private backendBaseURL: string | null = (typeof import.meta !== 'undefined' && (import.meta as any).env)
    ? (import.meta as any).env.VITE_API_BASE_URL || null
    : null;
  private auth: iNaturalistAuth | null = null;

  // Initialize authentication (this would typically be done through OAuth flow)
  async authenticate(): Promise<boolean> {
    try {
      // If backend is configured, get mock tokens from backend
      if (this.backendBaseURL) {
        const res = await fetch(`${this.backendBaseURL}/api/auth/mock-login`, {
          method: 'POST'
        });
        if (!res.ok) throw new Error('Backend auth failed');
        const data = await res.json();
        this.auth = {
          accessToken: data.accessToken,
          refreshToken: data.refreshToken,
          expiresAt: data.expiresAt
        };
        return true;
      }

      // Fallback: simulate authentication
      this.auth = {
        accessToken: 'mock_access_token',
        refreshToken: 'mock_refresh_token',
        expiresAt: Date.now() + 3600000 // 1 hour
      };
      return true;
    } catch (error) {
      console.error('iNaturalist authentication failed:', error);
      return false;
    }
  }

  // Upload observation to iNaturalist
  async uploadObservation(observation: iNaturalistObservation): Promise<boolean> {
    if (!this.auth) {
      const authenticated = await this.authenticate();
      if (!authenticated) {
        throw new Error('Failed to authenticate with iNaturalist');
      }
    }

    try {
      const formData = new FormData();
      formData.append('file', observation.imageFile);
      formData.append('species', observation.species);
      formData.append('scientificName', observation.scientificName);
      formData.append('confidence', observation.confidence.toString());
      formData.append('count', observation.count.toString());
      
      if (observation.location) {
        formData.append('latitude', observation.location.latitude.toString());
        formData.append('longitude', observation.location.longitude.toString());
      }
      if (observation.notes) {
        formData.append('notes', observation.notes);
      }

      if (this.backendBaseURL) {
        const res = await fetch(`${this.backendBaseURL}/api/observations`, {
          method: 'POST',
          body: formData
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`Backend error ${res.status}: ${text}`);
        }
        const data = await res.json();
        console.log('Observation uploaded via backend:', data);
        return true;
      }

      // Fallback simulation when backend is not configured
      await new Promise(resolve => setTimeout(resolve, 2000));
      console.log('Observation uploaded (simulated):', {
        species: observation.species,
        scientificName: observation.scientificName,
        confidence: observation.confidence,
        count: observation.count
      });
      return true;
    } catch (error) {
      console.error('Failed to upload observation to iNaturalist:', error);
      throw error;
    }
  }

  // Get user's observations
  async getUserObservations(): Promise<any[]> {
    if (!this.auth) {
      throw new Error('Not authenticated with iNaturalist');
    }

    try {
      if (this.backendBaseURL) {
        const res = await fetch(`${this.backendBaseURL}/api/observations`);
        if (!res.ok) throw new Error('Failed to fetch observations');
        return await res.json();
      }
      // Fallback
      return [];
    } catch (error) {
      console.error('Failed to fetch user observations:', error);
      throw error;
    }
  }

  // Search for species information
  async searchSpecies(query: string): Promise<any[]> {
    try {
      if (this.backendBaseURL) {
        const url = new URL(`${this.backendBaseURL}/api/species/search`);
        url.searchParams.set('q', query);
        const res = await fetch(url.toString());
        if (!res.ok) throw new Error('Search failed');
        return await res.json();
      }
      // Fallback
      return [];
    } catch (error) {
      console.error('Failed to search species:', error);
      throw error;
    }
  }

  // Get species details
  async getSpeciesDetails(scientificName: string): Promise<any> {
    try {
      if (this.backendBaseURL) {
        const res = await fetch(`${this.backendBaseURL}/api/species/${encodeURIComponent(scientificName)}`);
        if (!res.ok) throw new Error('Details fetch failed');
        return await res.json();
      }
      // Fallback
      return null;
    } catch (error) {
      console.error('Failed to get species details:', error);
      throw error;
    }
  }
}

export const iNaturalistAPI = new iNaturalistService();

// Helper function to get user's current location (for mobile devices)
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
          longitude: position.coords.longitude
        });
      },
      (error) => {
        console.warn('Failed to get location:', error);
        resolve(null);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 300000 // 5 minutes
      }
    );
  });
}
