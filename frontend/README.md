# Florida Anole Classifier - Frontend

A React web application for identifying and classifying Florida anole species from uploaded images. The app provides a user-friendly interface for species detection with confidence scoring and citizen science integration.

## What This Does

- **Species Classification**: Identifies 5 Florida anole species (Green Anole, Brown Anole, Crested Anole, Knight Anole, Bark Anole)
- **Multi-Detection**: Can detect multiple lizards in a single image
- **Confidence Scoring**: Shows confidence levels for each species prediction with visual indicators
- **Mobile Support**: Responsive design optimized for mobile devices
- **Citizen Science**: Integration with iNaturalist for contributing observations (currently simulated)

## Current Status

The application currently uses hardcoded mock predictions for testing the UI/UX flow. The prediction system randomly generates species classifications with confidence scores, and the iNaturalist upload is simulated. These will be replaced with actual machine learning model integration and real API calls.

## How to Launch

### Prerequisites
- Node.js 18 or higher
- npm or yarn

### Installation and Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The application will be available at `http://localhost:5173` (or the next available port).

### Building for Production
```bash
# Build the application
npm run build

# Preview the production build
npm run preview
```

## Project Structure

- `src/pages/LandingPage.tsx` - Welcome screen with features overview
- `src/pages/PredictionPage.tsx` - Main classification interface
- `src/services/iNaturalistService.ts` - iNaturalist API integration service
- `src/App.css` - Styling and mobile responsiveness

## Next Steps

- Integrate actual machine learning model for species classification
- Implement real iNaturalist API authentication and upload functionality
- Add support for all 5 Florida anole species in the prediction system
