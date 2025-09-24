# 🦎 Florida Anole Classifier

A comprehensive React application for identifying and classifying Florida anole species from images, with citizen science integration and mobile support.

## ✨ Features

### 🔍 Multi-Species Detection
- **Detects multiple lizards** in a single image
- **Identifies 5 Florida anole species**:
  - Green Anole (*Anolis carolinensis*)
  - Brown Anole (*Anolis sagrei*)
  - Crested Anole (*Anolis cristatellus*)
  - Knight Anole (*Anolis equestris*)
  - Bark Anole (*Anolis distichus*)

### 📊 Confidence Scoring
- **Real-time confidence levels** for each species prediction
- **Visual progress bars** showing prediction certainty
- **Color-coded confidence indicators** (High/Medium/Low)

### 🌐 iNaturalist Integration
- **Automatic upload** of classified observations to iNaturalist
- **Location tagging** (when GPS is available)
- **Citizen science contribution** with AI-detected species data
- **Batch upload** for multiple detections

### 📱 Mobile-First Design
- **Responsive design** optimized for iOS and Android
- **Touch-friendly interface** with proper touch targets
- **Camera integration** for direct photo capture
- **Offline-capable** with service worker support

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation
```bash
# Clone the repository
git clone https://github.com/codingFungus/anole_classification.git
cd anole_classification

# Install dependencies
npm install

# Start development server
npm run dev
```

### Building for Production
```bash
# Build the application
npm run build

# Preview production build
npm run preview
```

## 🏗️ Architecture

### Components
- **LandingPage**: Welcome screen with feature overview
- **PredictionPage**: Main classification interface
- **iNaturalistService**: API integration service

### Key Features Implementation
- **Multi-detection**: Handles multiple lizards per image
- **Confidence scoring**: Visual feedback on prediction certainty
- **Mobile optimization**: Responsive design with touch support
- **API integration**: Seamless iNaturalist upload workflow

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
VITE_INATURALIST_CLIENT_ID=your_client_id
VITE_INATURALIST_CLIENT_SECRET=your_client_secret
VITE_ML_API_URL=your_ml_model_endpoint
```

### iNaturalist Setup
1. Register your app at [iNaturalist Developer Portal](https://www.inaturalist.org/oauth/applications)
2. Configure OAuth redirect URLs
3. Add your client credentials to environment variables

## 📱 Mobile Features

### Touch Optimization
- **44px minimum touch targets** for accessibility
- **Swipe gestures** for navigation
- **Pinch-to-zoom** for image preview
- **Haptic feedback** on supported devices

### Camera Integration
- **Direct camera access** for field photography
- **Image compression** for faster uploads
- **EXIF data preservation** for location tagging

### Performance
- **Lazy loading** for images and components
- **Service worker** for offline functionality
- **Optimized bundle size** for mobile networks

## 🧪 Testing

### Manual Testing
1. **Upload test images** with various anole species
2. **Test mobile responsiveness** on different screen sizes
3. **Verify iNaturalist integration** with test observations
4. **Check confidence scoring** accuracy

### Automated Testing
```bash
# Run linting
npm run lint

# Run type checking
npm run type-check
```

## 🚀 Deployment

### Vercel (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

### Netlify
```bash
# Build and deploy
npm run build
# Upload dist/ folder to Netlify
```

### Mobile App (PWA)
The app is configured as a Progressive Web App and can be installed on mobile devices for native-like experience.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **iNaturalist** for citizen science platform
- **Florida anole research community** for species data
- **React and Vite** for the development framework
