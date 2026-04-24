import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { toast } from 'react-hot-toast';
import { AnoleDetectionService, type AnolePrediction } from '../services/AnoleDetectionService';
import { ResizableBoundingBox } from '../components/ResizableBoundingBox';
import { iNaturalistAPI, getCurrentLocation, type iNaturalistAuthStatus } from '../services/iNaturalistService';
import { PersistenceService } from '../services/PersistenceService';

// ============ DATA ============
const SPECIES_CONFIG = [
  { id: 'Green Anole',    name: 'Green Anole',    sci: 'Anolis carolinensis',  native: true,  color: '#7fa14a' },
  { id: 'Brown Anole',    name: 'Brown Anole',    sci: 'Anolis sagrei',        native: false, color: '#8b6a3e' },
  { id: 'Knight Anole',   name: 'Knight Anole',   sci: 'Anolis equestris',     native: false, color: '#4a6b2a' },
  { id: 'Bark Anole',     name: 'Bark Anole',     sci: 'Anolis distichus',     native: false, color: '#a8774a' },
  { id: 'Crested Anole',  name: 'Crested Anole',  sci: 'Anolis cristatellus',  native: false, color: '#6b8fa5' },
];

const RECENT_OBS_MOCK = [
  { id: 1, species: 'Green Anole', where: 'Gainesville, FL',  when: '2h ago',  conf: 0.94 },
  { id: 2, species: 'Brown Anole', where: 'Miami, FL',        when: '5h ago',  conf: 0.97 },
  { id: 3, species: 'Brown Anole', where: 'Tampa, FL',        when: '1d ago',  conf: 0.88 },
  { id: 4, species: 'Green Anole', where: 'Orlando, FL',      when: '2d ago',  conf: 0.91 },
  { id: 5, species: 'Knight Anole', where: 'Homestead, FL', when: '3d ago', conf: 0.82 },
];

// ============ ICONS ============
const Icon = {
  Upload: (p: { s?: number }) => <svg viewBox="0 0 24 24" width={p.s||18} height={p.s||18} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 4v12"/><path d="m7 9 5-5 5 5"/><path d="M4 20h16"/></svg>,
  Check:  (p: { s?: number }) => <svg viewBox="0 0 24 24" width={p.s||14} height={p.s||14} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m5 12 5 5 9-10"/></svg>,
  Arrow:  (p: { s?: number }) => <svg viewBox="0 0 24 24" width={p.s||14} height={p.s||14} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></svg>,
  Edit:   (p: { s?: number }) => <svg viewBox="0 0 24 24" width={p.s||14} height={p.s||14} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>,
};

// ============ COMPONENTS ============

function Rule({ label, align = 'left' }: { label: string, align?: 'left' | 'center' }) {
  return (
    <div style={{ display:'flex', alignItems:'center', gap:10, margin:'0 0 14px' }}>
      {align !== 'left' && <div style={{ flex:1, height:1, background:'var(--rule)' }}/>}
      <div className="mono" style={{ fontSize:10, letterSpacing:'0.18em', color:'var(--ink-3)', textTransform:'uppercase' }}>{label}</div>
      <div style={{ flex:1, height:1, background:'var(--rule)' }}/>
    </div>
  );
}

function CountUp({ to, duration = 1200 }: { to: number, duration?: number }) {
  const [n, setN] = useState(0);
  useEffect(() => {
    let raf: number, start: number;
    const step = (t: number) => {
      if (!start) start = t;
      const p = Math.min(1, (t - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setN(Math.round(to * eased));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [to, duration]);
  return <>{n}</>;
}

function SpeciesRibbon({ stats }: { stats: any }) {
  const total = stats?.observations || 100;
  // Mock species distribution based on total
  const speciesData = SPECIES_CONFIG.map((s, i) => ({
    ...s,
    count: Math.round(total * [0.35, 0.45, 0.1, 0.06, 0.04][i])
  }));

  return (
    <div>
      <div style={{ display:'flex', alignItems:'baseline', justifyContent:'space-between', marginBottom:10, gap:12, flexWrap:'wrap' }}>
        <div>
          <div className="serif" style={{ fontSize:32, fontWeight:600, letterSpacing:'-0.01em' }}>Which species?</div>
          <div style={{ fontSize:13, color:'var(--ink-3)' }}>{total} sightings tracked via iNaturalist</div>
        </div>
        <div className="mono" style={{ fontSize:11, color:'var(--ink-3)', letterSpacing:'0.1em' }}>N = {total}</div>
      </div>

      <div style={{
        display:'flex', width:'100%', height:44, borderRadius:4,
        overflow:'hidden', border:'1px solid var(--ink)', boxShadow:'0 1px 0 rgba(0,0,0,0.04)'
      }}>
        {speciesData.map((s, i) => (
          <div key={s.id} title={`${s.name}: ${s.count}`} style={{
            flex: s.count,
            background: s.color,
            borderRight: i < speciesData.length - 1 ? '1px solid rgba(0,0,0,0.15)' : 'none',
            position:'relative',
            display:'flex', alignItems:'center', justifyContent:'center',
          }}>
            {s.count / total > 0.08 && (
              <span className="mono" style={{ color:'#fff', fontSize:11, fontWeight:600, letterSpacing:'0.05em' }}>
                {Math.round(s.count/total*100)}%
              </span>
            )}
          </div>
        ))}
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit, minmax(210px, 1fr))', gap:'18px 24px', marginTop:18 }}>
        {speciesData.map(s => (
          <div key={s.id} style={{ display:'flex', alignItems:'flex-start', gap:10 }}>
            <div style={{
              width:10, height:10, marginTop:6, flexShrink:0,
              background:s.color,
              border:'1px solid var(--ink)',
              transform:'rotate(45deg)'
            }}/>
            <div style={{ flex:1, minWidth:0 }}>
              <div style={{ display:'flex', justifyContent:'space-between', gap:6, lineHeight:1.35 }}>
                <div style={{ fontSize:13, fontWeight:600, whiteSpace:'nowrap' }}>{s.name}</div>
                <div className="mono" style={{ fontSize:12, color:'var(--ink-2)' }}>{s.count}</div>
              </div>
              <div className="serif" style={{ fontStyle:'italic', fontSize:12, color:'var(--ink-3)', lineHeight:1.4, marginTop:2 }}>
                {s.sci} {s.native && <span style={{ color:'var(--ok)', fontStyle:'normal', fontSize:10, marginLeft:4 }}>● native</span>}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ActivityChart({ total }: { total: number }) {
  const data = [2,1,3,2,4,3,5,2,6,4,3,7,5,8,4,6,9,7,5,8,6,10,7,9,11,8,12,9,7,13];
  const max = Math.max(...data);
  const W = 100, H = 32, step = W / (data.length - 1);
  const points = data.map((v,i) => [i*step, H - (v/max) * (H-2) - 1]);
  const pathD = 'M ' + points.map(([x,y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' L ');
  const areaD = pathD + ` L ${W},${H} L 0,${H} Z`;

  return (
    <div style={{ border:'1px solid var(--rule)', background:'var(--paper)', padding:18, borderRadius:4 }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:6 }}>
        <div className="mono" style={{ fontSize:10, letterSpacing:'0.15em', color:'var(--ink-3)', textTransform:'uppercase' }}>30-day activity</div>
        <div className="mono" style={{ fontSize:10, color:'var(--ok)' }}>▲ 14%</div>
      </div>
      <div style={{ display:'flex', alignItems:'baseline', gap:8, marginBottom:10 }}>
        <div className="serif" style={{ fontSize:34, fontWeight:600, lineHeight:1 }}><CountUp to={total}/></div>
        <div style={{ fontSize:12, color:'var(--ink-3)' }}>observations</div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ width:'100%', height:48, display:'block' }}>
        <path d={areaD} fill="var(--moss)" fillOpacity="0.12"/>
        <path d={pathD} stroke="var(--moss)" strokeWidth="1.2" fill="none" vectorEffect="non-scaling-stroke"/>
        <circle cx={points[points.length-1][0]} cy={points[points.length-1][1]} r="1.5" fill="var(--dewlap)"/>
      </svg>
      <div style={{ display:'flex', justifyContent:'space-between', marginTop:6 }} className="mono">
        <span style={{ fontSize:10, color:'var(--ink-3)' }}>Last Month</span>
        <span style={{ fontSize:10, color:'var(--ink-3)' }}>today</span>
      </div>
    </div>
  );
}

function ClassifyPanel({ inatStatus, selectedFile, setSelectedFile, preview, setPreview }: { inatStatus: iNaturalistAuthStatus | null, selectedFile: File | null, setSelectedFile: (f: File | null) => void, preview: string | null, setPreview: (s: string | null) => void }) {
  const [state, setState] = useState<'idle' | 'dragging' | 'analyzing' | 'done'>('idle');
  const [result, setResult] = useState<any>(null);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const [reclassifyingIndex, setReclassifyingIndex] = useState<number | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawingBox, setDrawingBox] = useState<{ startX: number, startY: number, endX: number, endY: number } | null>(null);
  const [isRestored, setIsRestored] = useState(false);

  const fileRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const reclassifyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Restore persistence
  useEffect(() => {
    const restore = async () => {
      try {
        const [savedFile, savedResult] = await Promise.all([
          PersistenceService.loadImage(),
          PersistenceService.loadResult()
        ]);
        if (savedFile) {
          setSelectedFile(savedFile);
          setPreview(URL.createObjectURL(savedFile));
        }
        if (savedResult) {
          setResult(savedResult);
          setState('done');
        }
      } catch (e) {
        console.warn("Failed to restore persistence", e);
      } finally {
        setIsRestored(true);
      }
    };
    restore();
  }, [setSelectedFile, setPreview]);

  // Save persistence
  useEffect(() => {
    if (isRestored) {
      if (selectedFile) PersistenceService.saveImage(selectedFile);
      PersistenceService.saveResult(result);
    }
  }, [selectedFile, result, isRestored]);

  useEffect(() => {
    return () => {
      if (reclassifyTimeoutRef.current) clearTimeout(reclassifyTimeoutRef.current);
    };
  }, []);

  const handleClassify = async (file: File) => {
    if (!file) return;
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setState('analyzing');
    
    try {
      const data = await AnoleDetectionService.detect(file, 'auto');
      
      if (data.predictions && data.predictions.length > 0) {
        setResult(data);
      } else {
        setResult({ totalLizards: 0, predictions: [] });
      }
      setState('done');
    } catch (err) {
      console.error(err);
      toast.error("Classification failed. Please try again.");
      setState('idle');
    }
  };

  const handleUploadToInat = async () => {
    if (!result || !selectedFile || !inatStatus?.connected) {
      if (!inatStatus?.connected) toast.error("Please connect to iNaturalist first");
      return;
    }

    const topPred = result.predictions[0];
    if (!topPred) return;

    setIsUploading(true);
    const toastId = toast.loading("Uploading to iNaturalist...");

    try {
      const location = await getCurrentLocation();
      await iNaturalistAPI.uploadObservation({
        species: topPred.species,
        scientificName: topPred.scientificName,
        confidence: topPred.confidence,
        count: 1,
        imageFile: selectedFile,
        location: location || undefined
      });
      toast.success("Observation uploaded successfully!", { id: toastId });
    } catch (e) {
      console.error(e);
      toast.error("Failed to upload observation", { id: toastId });
    } finally {
      setIsUploading(false);
    }
  };

  const handleBoxResize = (index: number, newBox: { x: number; y: number; width: number; height: number }) => {
    if (!result || !selectedFile) return;

    const [x1, y1, x2, y2] = [
      newBox.x,
      newBox.y,
      newBox.x + newBox.width,
      newBox.y + newBox.height
    ];

    // Update locally immediately
    const updatedPredictions = [...result.predictions];
    updatedPredictions[index] = {
      ...updatedPredictions[index],
      boundingBox: [x1, y1, x2, y2],
    };
    setResult({ ...result, predictions: updatedPredictions });

    // Debounce re-classification
    if (reclassifyTimeoutRef.current) clearTimeout(reclassifyTimeoutRef.current);

    reclassifyTimeoutRef.current = setTimeout(async () => {
      try {
        setReclassifyingIndex(index);
        const classification = await AnoleDetectionService.classifyRegion(
          selectedFile,
          [x1, y1, x2, y2],
          'auto'
        );

        const finalPredictions = [...result.predictions];
        finalPredictions[index] = {
          ...finalPredictions[index],
          species: classification.species,
          scientificName: classification.scientificName,
          confidence: classification.confidence,
          boundingBox: [x1, y1, x2, y2],
          alternateConfidences: classification.alternateConfidences,
        };
        setResult({ ...result, predictions: finalPredictions });
      } catch (error) {
        console.error("Reclassification failed:", error);
      } finally {
        setReclassifyingIndex(null);
      }
    }, 600); // 600ms debounce
  };

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setImageDimensions({
      width: img.naturalWidth,
      height: img.naturalHeight
    });
  };

  const displayToNatural = useCallback((displayX: number, displayY: number): [number, number] => {
    if (!imageDimensions || !imageRef.current) return [0, 0];
    const scaleX = imageDimensions.width / imageRef.current.clientWidth;
    const scaleY = imageDimensions.height / imageRef.current.clientHeight;
    return [displayX * scaleX, displayY * scaleY];
  }, [imageDimensions]);

  const naturalToDisplay = useCallback((naturalX: number, naturalY: number): [number, number] => {
    if (!imageDimensions || !imageRef.current) return [0, 0];
    const scaleX = imageRef.current.clientWidth / imageDimensions.width;
    const scaleY = imageRef.current.clientHeight / imageDimensions.height;
    return [naturalX * scaleX, naturalY * scaleY];
  }, [imageDimensions]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!preview || !imageContainerRef.current || !imageRef.current || !imageDimensions || reclassifyingIndex !== null) return;
    
    // Prevent default browser behavior (dragging images, selecting text)
    e.preventDefault();
    
    const rect = imageRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Only start drawing if within image bounds
    if (x < 0 || y < 0 || x > imageRef.current.clientWidth || y > imageRef.current.clientHeight) return;
    
    setIsDrawing(true);
    const [naturalX, naturalY] = displayToNatural(x, y);
    setDrawingBox({ startX: naturalX, startY: naturalY, endX: naturalX, endY: naturalY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing || !drawingBox || !imageRef.current) return;
    
    const rect = imageRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, imageRef.current.clientWidth));
    const y = Math.max(0, Math.min(e.clientY - rect.top, imageRef.current.clientHeight));
    
    const [naturalX, naturalY] = displayToNatural(x, y);
    setDrawingBox({ ...drawingBox, endX: naturalX, endY: naturalY });
  };

  const handleMouseUp = async () => {
    if (!isDrawing || !drawingBox || !selectedFile || !imageDimensions) return;
    
    const x1 = Math.min(drawingBox.startX, drawingBox.endX);
    const y1 = Math.min(drawingBox.startY, drawingBox.endY);
    const x2 = Math.max(drawingBox.startX, drawingBox.endX);
    const y2 = Math.max(drawingBox.startY, drawingBox.endY);
    
    const width = x2 - x1;
    const height = y2 - y1;

    // Minimum box size
    if (width < 10 || height < 10) {
      setIsDrawing(false);
      setDrawingBox(null);
      return;
    }

    setIsDrawing(false);
    setDrawingBox(null);

    const finalBox: [number, number, number, number] = [x1, y1, x2, y2];
    
    try {
      toast.loading("Analyzing region...", { id: 'region-scan' });
      const classification = await AnoleDetectionService.classifyRegion(selectedFile, finalBox, 'auto');
      
      const newPrediction: any = {
        species: classification.species,
        scientificName: classification.scientificName,
        confidence: classification.confidence,
        boundingBox: finalBox,
        alternateConfidences: classification.alternateConfidences
      };

      if (result) {
        setResult({
          ...result,
          totalLizards: result.totalLizards + 1,
          predictions: [...result.predictions, newPrediction]
        });
      } else {
        setResult({
          totalLizards: 1,
          predictions: [newPrediction]
        });
        setState('done');
      }
      toast.success("Region classified", { id: 'region-scan' });
    } catch (error) {
      console.error(error);
      toast.error("Failed to classify region", { id: 'region-scan' });
    }
  };

  const handleFile = (file?: File) => {
    if (!file) return;
    handleClassify(file);
  };

  const reset = () => { 
    setPreview(null); 
    setResult(null); 
    setState('idle'); 
    setImageDimensions(null); 
    PersistenceService.clearAll();
  };

  return (
    <div>
      <div
        ref={imageContainerRef}
        onDragOver={e => { e.preventDefault(); setState(s => s === 'idle' ? 'dragging' : s); }}
        onDragLeave={() => setState(s => s === 'dragging' ? 'idle' : s)}
        onDrop={e => {
          e.preventDefault();
          setState('idle');
          handleFile(e.dataTransfer.files[0]);
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        style={{
          position:'relative',
          border: state==='dragging' ? '1.5px solid var(--moss)' : '1px dashed var(--ink-3)',
          background: state==='dragging' ? 'rgba(61,100,42,0.06)' : 'var(--paper-2)',
          borderRadius: 4,
          minHeight: preview ? 'auto' : 300,
          display:'flex', alignItems:'center', justifyContent:'center',
          overflow:'visible',
          transition:'all 200ms',
          cursor: preview ? 'crosshair' : 'default',
          userSelect: 'none'
        }}
      >
        {preview && (
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <img 
              ref={imageRef}
              src={preview} 
              alt="Preview" 
              onLoad={handleImageLoad}
              draggable="false"
              style={{
                display: 'block',
                maxWidth: '100%',
                height: 'auto',
                maxHeight: '500px',
                borderRadius: 4,
                filter: (state === 'analyzing' || reclassifyingIndex !== null) ? 'saturate(0.7) brightness(0.9)' : 'none',
                transition: 'filter 400ms',
                pointerEvents: 'none'
              }}
            />
            
            {/* Bounding Boxes Layer */}
            <div style={{ position: 'absolute', inset: 0, pointerEvents: 'auto' }}>
              {state === 'done' && result && imageDimensions && imageRef.current && (
                result.predictions.map((p: any, i: number) => {
                  if (!p.boundingBox) return null;
                  const [x1, y1, x2, y2] = p.boundingBox;
                  return (
                    <ResizableBoundingBox
                      key={`${i}-${p.boundingBox.join(',')}`}
                      x={x1}
                      y={y1}
                      width={x2 - x1}
                      height={y2 - y1}
                      label={`${p.species} (${(p.confidence * 100).toFixed(0)}%)`}
                      color={AnoleDetectionService.getSpeciesColor(p.species)}
                      imageNaturalWidth={imageDimensions.width}
                      imageNaturalHeight={imageDimensions.height}
                      imageDisplayWidth={imageRef.current!.clientWidth}
                      imageDisplayHeight={imageRef.current!.clientHeight}
                      onResize={(newBox) => handleBoxResize(i, newBox)}
                      disabled={reclassifyingIndex !== null || isDrawing}
                    />
                  );
                })
              )}
            </div>
          </div>
        )}
        
        {!preview && (
          <div style={{ textAlign:'center', color:'var(--ink-3)', padding: '40px 0' }}>
            <div style={{ display:'inline-flex', padding:14, border:'1px solid var(--rule)', borderRadius:'50%', marginBottom:12, background:'var(--paper)' }}>
              <Icon.Upload s={22}/>
            </div>
            <div className="serif" style={{ fontSize:17, color:'var(--ink)', marginBottom:4 }}>Drop a lizard photo here</div>
            <div style={{ fontSize:12 }}>or click to upload</div>
          </div>
        )}

        {/* Drawing Box Preview */}
        {isDrawing && drawingBox && (
          <div style={{
            position: 'absolute',
            left: Math.min(naturalToDisplay(drawingBox.startX, 0)[0], naturalToDisplay(drawingBox.endX, 0)[0]),
            top: Math.min(naturalToDisplay(0, drawingBox.startY)[1], naturalToDisplay(0, drawingBox.endY)[1]),
            width: Math.abs(naturalToDisplay(drawingBox.startX, 0)[0] - naturalToDisplay(drawingBox.endX, 0)[0]),
            height: Math.abs(naturalToDisplay(0, drawingBox.startY)[1] - naturalToDisplay(0, drawingBox.endY)[1]),
            border: '2px dashed var(--dewlap)',
            backgroundColor: 'rgba(216, 87, 42, 0.1)',
            pointerEvents: 'none',
            zIndex: 20
          }} />
        )}

        {state === 'analyzing' && (
          <>
            <div style={{
              position:'absolute', inset:0, pointerEvents:'none',
              background: 'linear-gradient(180deg, transparent 0%, rgba(216,87,42,0.35) 50%, transparent 100%)',
              height:'30%', animation:'scan 1.6s ease-in-out infinite'
            }}/>
            <div style={{
              position:'absolute', left:12, top:12,
              background:'var(--ink)', color:'var(--paper)',
              padding:'6px 10px', borderRadius:3,
              display:'flex', alignItems:'center', gap:8
            }}>
              <div style={{ width:8, height:8, borderRadius:'50%', background:'var(--dewlap)', animation:'pulse 1s ease-in-out infinite' }}/>
              <span className="mono" style={{ fontSize:10, letterSpacing:'0.12em' }}>CLASSIFYING...</span>
            </div>
          </>
        )}
      </div>

      {state === 'done' ? (
        <ResultCard 
          result={result} 
          onReset={reset} 
          onCorrect={(newSpecies) => {
            const updated = {...result};
            updated.predictions[0] = {
              ...updated.predictions[0],
              species: newSpecies,
              scientificName: SPECIES_CONFIG.find(s => s.name === newSpecies)?.sci || '',
              confidence: 1.0,
              isManualCorrection: true
            };
            setResult(updated);
            toast.success(`Updated to ${newSpecies}`);
          }}
          onUpload={handleUploadToInat}
          isUploading={isUploading}
          inatConnected={inatStatus?.connected || false}
        />
      ) : (
        <div style={{ display:'flex', gap:10, marginTop:12 }}>
          <button
            onClick={() => fileRef.current?.click()}
            disabled={state==='analyzing'}
            style={{
              flex:1,
              border:'1px solid var(--ink)', background:'var(--ink)', color:'var(--paper)',
              padding:'11px 14px', borderRadius:3, fontSize:13, fontWeight:500,
              display:'flex', alignItems:'center', justifyContent:'center', gap:8,
              opacity: state==='analyzing' ? 0.5 : 1
            }}>
            <Icon.Upload s={14}/> Upload photo
          </button>
          <input ref={fileRef} type="file" accept="image/*" onChange={e => handleFile(e.target.files?.[0])} style={{ display:'none' }}/>
        </div>
      )}
      <style>{`
        @keyframes scan { 0% { transform: translateY(-100%); } 100% { transform: translateY(400%); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
      `}</style>
    </div>
  );
}

function ResultCard({ result, onReset, onCorrect, onUpload, isUploading, inatConnected }: { result: any, onReset: () => void, onCorrect: (s: string) => void, onUpload: () => void, isUploading: boolean, inatConnected: boolean }) {
  const [isCorrecting, setIsCorrecting] = useState(false);
  
  if (!result || result.totalLizards === 0) {
    return (
      <div style={{ marginTop:14, padding: 20, border:'1px solid var(--ink)', background:'var(--paper)', borderRadius:4, textAlign: 'center' }}>
        <div className="serif" style={{ fontSize:18, marginBottom: 10 }}>No lizards detected</div>
        <button onClick={onReset} style={{ border:'1px solid var(--ink)', background:'var(--ink)', color:'var(--paper)', padding:'8px 16px', borderRadius:3 }}>Try another photo</button>
      </div>
    );
  }

  const topPred = result.predictions[0];
  const topConfig = SPECIES_CONFIG.find(s => s.name === topPred.species) || SPECIES_CONFIG[0];

  return (
    <div style={{ marginTop:14, border:'1px solid var(--ink)', background:'var(--paper)', borderRadius:4, overflow:'hidden' }}>
      <div style={{ padding:'14px 16px', borderBottom:'1px solid var(--rule)', background:'var(--paper-2)', display:'flex', alignItems:'center', justifyContent:'space-between' }}>
        <div className="mono" style={{ fontSize:10, letterSpacing:'0.15em', color:'var(--ink-3)', textTransform:'uppercase' }}>
          Classification result · {result.totalLizards} lizard{result.totalLizards > 1 ? 's' : ''} found
        </div>
        {!topPred.isManualCorrection && (
           <div className="mono" style={{ fontSize:10, color: topPred.confidence > 0.8 ? 'var(--ok)' : 'var(--warn)', display:'flex', alignItems:'center', gap:5 }}>
           {topPred.confidence > 0.8 ? <Icon.Check s={12}/> : null} {topPred.confidence > 0.8 ? 'HIGH CONFIDENCE' : 'LOW CONFIDENCE'}
         </div>
        )}
      </div>
      <div style={{ padding:'16px 18px' }}>
        <div className="serif" style={{ fontSize:26, fontWeight:600, letterSpacing:'-0.01em' }}>{topPred.species}</div>
        <div className="serif" style={{ fontStyle:'italic', fontSize:14, color:'var(--ink-3)', marginBottom:14 }}>
          {topPred.scientificName} {topConfig.native && <span style={{ color:'var(--ok)', fontStyle:'normal', fontSize:11, marginLeft:6 }}>● NATIVE</span>}
        </div>

        {/* Confidence bars */}
        <div style={{ display:'flex', flexDirection:'column', gap:6 }}>
          <ConfidenceRow species={topPred.species} conf={topPred.confidence} top/>
          {(topPred.alternateConfidences || []).slice(0, 3).map((o: any) => (
            <ConfidenceRow key={o.species} species={o.species} conf={o.confidence}/>
          ))}
        </div>

        {!isCorrecting ? (
          <div style={{ display:'flex', gap:8, marginTop:16, position: 'relative' }}>
            <button 
              onClick={onUpload}
              disabled={isUploading || !inatConnected}
              style={{
                flex:1, border:'1px solid var(--ink)', background: inatConnected ? 'var(--ink)' : 'var(--rule)', color:'var(--paper)',
                padding:'10px 14px', borderRadius:3, fontSize:12, fontWeight:500, letterSpacing:'0.03em',
                display:'flex', alignItems:'center', justifyContent:'center', gap:6,
                cursor: inatConnected ? 'pointer' : 'not-allowed'
              }}>
              {isUploading ? 'Uploading...' : 'Submit to iNaturalist'} <Icon.Arrow s={12}/>
            </button>
            {!inatConnected && (
              <div style={{ position: 'absolute', bottom: -18, left: 0, right: 0, textAlign: 'center', fontSize: 9, color: 'var(--ink-3)' }}>
                Connect iNaturalist to upload
              </div>
            )}
            <button onClick={() => setIsCorrecting(true)} style={{
              border:'1px solid var(--rule)', background:'transparent', color:'var(--ink-2)',
              padding:'10px 14px', borderRadius:3, fontSize:12, display: 'flex', alignItems: 'center', gap: 6
            }}>
              <Icon.Edit s={12}/> Correct
            </button>
            <button onClick={onReset} style={{
              border:'1px solid var(--rule)', background:'transparent', color:'var(--ink-2)',
              padding:'10px 14px', borderRadius:3, fontSize:12
            }}>
              Reset
            </button>
          </div>
        ) : (
          <div style={{ marginTop: 16, borderTop: '1px solid var(--rule)', paddingTop: 12 }}>
            <div className="mono" style={{ fontSize: 10, marginBottom: 8, textTransform: 'uppercase' }}>Select correct species:</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
              {SPECIES_CONFIG.map(s => (
                <button 
                  key={s.id} 
                  onClick={() => { onCorrect(s.name); setIsCorrecting(false); }}
                  style={{ 
                    padding: '6px 10px', fontSize: 11, textAlign: 'left', borderRadius: 3,
                    border: '1px solid var(--rule)', background: topPred.species === s.name ? 'var(--paper-2)' : 'white'
                  }}
                >
                  {s.name}
                </button>
              ))}
              <button onClick={() => setIsCorrecting(false)} style={{ padding: '6px 10px', fontSize: 11, borderRadius: 3, border: '1px solid var(--ink)', background: 'var(--ink)', color: 'white' }}>Cancel</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ConfidenceRow({ species, conf, top }: { species: string, conf: number, top?: boolean }) {
  const sp = SPECIES_CONFIG.find(s => s.name === species);
  if (!sp) return null;
  return (
    <div style={{ display:'grid', gridTemplateColumns:'minmax(90px, 120px) 1fr 44px', gap:10, alignItems:'center' }}>
      <div style={{ fontSize: top ? 12 : 11, fontWeight: top ? 600 : 400, color: top ? 'var(--ink)' : 'var(--ink-3)' }}>
        {sp.name}
      </div>
      <div style={{ height: top ? 8 : 4, background:'var(--paper-3)', borderRadius: 2, overflow:'hidden' }}>
        <div style={{
          width: `${conf*100}%`, height:'100%',
          background: top ? 'var(--moss)' : 'var(--ink-3)',
          transition:'width 700ms cubic-bezier(.2,.8,.2,1)'
        }}/>
      </div>
      <div className="mono" style={{ fontSize:11, textAlign:'right', color: top ? 'var(--ink)' : 'var(--ink-3)', fontWeight: top ? 600 : 400 }}>
        {(conf*100).toFixed(1)}%
      </div>
    </div>
  );
}

// ============ MAIN PAGE ============

export default function NewDesignPage() {
  const [stats, setStats] = useState<any>(null);
  const [inatStatus, setInatStatus] = useState<iNaturalistAuthStatus | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const refreshInatStatus = useCallback(async () => {
    if (!iNaturalistAPI.isBackendConfigured()) return;
    try {
      const s = await iNaturalistAPI.getAuthStatus();
      setInatStatus(s);
    } catch (e) {
      console.warn("iNaturalist status:", e);
    }
  }, []);

  useEffect(() => {
    refreshInatStatus();
  }, [refreshInatStatus]);

  useEffect(() => {
    // Inject fonts
    const link = document.createElement('link');
    link.href = 'https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,100..900;1,9..144,100..900&family=Inter:wght@100..900&family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap';
    link.rel = 'stylesheet';
    document.head.appendChild(link);

    // Fetch stats
    const fetchStats = async () => {
      try {
        const res = await fetch('/api/metrics/stats');
        if (res.ok) {
          const data = await res.json();
          setStats(data);
        }
      } catch (e) {
        console.error("Failed to fetch stats", e);
      }
    };
    fetchStats();

    return () => {
      document.head.removeChild(link);
    };
  }, []);

  return (
    <div style={{ 
      backgroundColor: '#f4efe4', 
      minHeight: '100vh',
      color: '#1f231c',
      fontFamily: "'Inter', sans-serif",
      position: 'relative'
    }} className="new-design-root">
      <style>{`
        :root {
          --paper: #f4efe4;
          --paper-2: #ebe4d3;
          --paper-3: #e2d9c3;
          --ink: #1f231c;
          --ink-2: #3a3f34;
          --ink-3: #6a6f5f;
          --rule: #cfc6ad;
          --moss: #3d5a2a;
          --dewlap: #d8572a;
          --ok: #4a7a3d;
          --warn: #b8802a;
        }
        .serif { font-family: 'Fraunces', serif; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .ll-main { max-width: 1240px; margin: 0 auto; padding: 40px 48px 80px; }
        .ll-hero { display: grid; grid-template-columns: 1fr 1fr; gap: 60px; align-items: start; margin-bottom: 44px; }
        @media (max-width: 860px) {
          .ll-hero { grid-template-columns: 1fr; gap: 36px; }
          .ll-main { padding: 20px 20px 60px; }
        }
      `}</style>
      
      {/* Header */}
      <header style={{ padding: '22px 48px', borderBottom: '1px solid var(--rule)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'white' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ fontSize: 24 }}>🦎</div>
          <div>
            <div className="serif" style={{ fontSize: 20, fontWeight: 700, lineHeight: 1 }}>Lizard Lens</div>
            <div className="mono" style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--ink-3)' }}>Field Classifier</div>
          </div>
        </div>
        <nav style={{ display: 'flex', gap: 24, alignItems: 'center' }} className="mono">
          {inatStatus?.connected ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--ok)' }} />
              <span style={{ fontSize: 10, color: 'var(--ink-2)' }}>iNAT CONNECTED</span>
              <button onClick={() => iNaturalistAPI.disconnect().then(refreshInatStatus)} style={{ border: 'none', background: 'none', color: 'var(--ink-3)', fontSize: 10, cursor: 'pointer', textDecoration: 'underline' }}>Disconnect</button>
            </div>
          ) : (
            <button onClick={() => iNaturalistAPI.connectAccount()} style={{ border: '1px solid var(--ink)', background: 'var(--ink)', color: 'white', fontSize: 10, padding: '4px 10px', borderRadius: 4, cursor: 'pointer' }}>CONNECT iNATURALIST</button>
          )}
          <div style={{ fontSize: 11, textTransform: 'uppercase', color: 'var(--ink-3)' }}>{new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' }).toUpperCase()}</div>
        </nav>
      </header>

      <main className="ll-main">
        <div className="ll-hero">
          <div>
            <div className="mono" style={{ fontSize: 10, letterSpacing: '0.2em', color: 'var(--dewlap)', textTransform: 'uppercase', marginBottom: 16 }}>
              ● Citizen science
            </div>
            <h1 className="serif" style={{ fontSize: 54, fontWeight: 500, letterSpacing: '-0.02em', margin: '0 0 18px', lineHeight: 1.1 }}>
              Identify Florida<br/>Anoles instantly.
            </h1>
            <p style={{ fontSize: 16, color: 'var(--ink-2)', maxWidth: 460, lineHeight: 1.6 }}>
              Upload a photo of an anole. Our AI will classify the species and help you contribute to research efforts across the sunshine state.
            </p>
            <div style={{ display: 'flex', gap: 24, marginTop: 32 }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>5 species</div>
                <div className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>CURRENTLY SUPPORTED</div>
              </div>
              <div>
                <div style={{ fontSize: 14, fontWeight: 600 }}>Real-time</div>
                <div className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>ON-DEVICE IDENTIFICATION</div>
              </div>
            </div>
          </div>

          <div>
            <Rule label="01 · Classify specimen"/>
            <ClassifyPanel 
              inatStatus={inatStatus} 
              selectedFile={selectedFile} 
              setSelectedFile={setSelectedFile} 
              preview={preview} 
              setPreview={setPreview}
            />
          </div>
        </div>

        <div style={{ marginTop: 60 }}>
          <Rule label="02 · Community Impact" />
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 40 }}>
            <section>
              <SpeciesRibbon stats={stats} />
            </section>
            <section>
              <ActivityChart total={stats?.observations || 0} />
              <div style={{ marginTop: 20, border:'1px solid var(--rule)', background:'var(--paper)', padding:18, borderRadius:4 }}>
                 <div className="mono" style={{ fontSize:10, textTransform:'uppercase', color:'var(--ink-3)', marginBottom: 8 }}>Top observers</div>
                 <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {[{n:'m.chen', c:34}, {n:'r.patel', c:28}, {n:'j.alvarez', c:22}].map((u, i) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                        <span className="mono">{u.n}</span>
                        <span className="mono" style={{ fontWeight: 600 }}>{u.c}</span>
                      </div>
                    ))}
                 </div>
              </div>
            </section>
          </div>
        </div>

        <footer style={{ marginTop: 80, paddingTop: 20, borderTop: '1px solid var(--rule)', display: 'flex', justifyContent: 'space-between' }} className="mono">
           <div style={{ fontSize: 10, color: 'var(--ink-3)' }}>LIZARD LENS v0.4</div>
           <div style={{ fontSize: 10, color: 'var(--ink-3)' }}>POWERED BY STROUD LAB, GEORGIA TECH</div>
        </footer>
      </main>
    </div>
  );
}
