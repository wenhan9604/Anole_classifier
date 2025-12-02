import { useState, useRef, useEffect } from 'react';

interface ResizableBoundingBoxProps {
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
  label: string;
  onResize: (newBox: { x: number; y: number; width: number; height: number }) => void;
  imageNaturalWidth: number;
  imageNaturalHeight: number;
  imageDisplayWidth: number;
  imageDisplayHeight: number;
  disabled?: boolean;
}

export function ResizableBoundingBox({
  x,
  y,
  width,
  height,
  color,
  label,
  onResize,
  imageNaturalWidth,
  imageNaturalHeight,
  imageDisplayWidth,
  imageDisplayHeight,
  disabled = false,
}: ResizableBoundingBoxProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState<string | null>(null);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [boxStart, setBoxStart] = useState({ x: 0, y: 0, width: 0, height: 0 });
  const boxRef = useRef<HTMLDivElement>(null);

  // Convert natural image coordinates to display coordinates
  const toDisplayCoords = (naturalX: number, naturalY: number): [number, number] => {
    const scaleX = imageDisplayWidth / imageNaturalWidth;
    const scaleY = imageDisplayHeight / imageNaturalHeight;
    return [naturalX * scaleX, naturalY * scaleY];
  };

  const getImageRelativeCoords = (clientX: number, clientY: number): [number, number] => {
    if (!boxRef.current) return [0, 0];
    // Get the container (parent) element
    const container = boxRef.current.parentElement;
    if (!container) return [0, 0];
    const containerRect = container.getBoundingClientRect();
    
    // Convert viewport coordinates to container-relative coordinates
    const containerX = clientX - containerRect.left;
    const containerY = clientY - containerRect.top;
    
    // Convert container coordinates (display) to natural image coordinates
    const scaleX = imageNaturalWidth / imageDisplayWidth;
    const scaleY = imageNaturalHeight / imageDisplayHeight;
    
    return [containerX * scaleX, containerY * scaleY];
  };

  const handleMouseDown = (e: React.MouseEvent, type: 'drag' | 'resize', corner?: string) => {
    if (disabled) return;
    e.preventDefault();
    e.stopPropagation();

    const [startX, startY] = getImageRelativeCoords(e.clientX, e.clientY);

    if (type === 'drag') {
      setIsDragging(true);
      setDragStart({ x: startX, y: startY });
      setBoxStart({ x, y, width, height });
    } else if (type === 'resize' && corner) {
      setIsResizing(corner);
      setDragStart({ x: startX, y: startY });
      setBoxStart({ x, y, width, height });
    }
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!boxRef.current) return;
      
      const container = boxRef.current.parentElement;
      if (!container) return;
      const containerRect = container.getBoundingClientRect();
      
      // Convert viewport coordinates to container-relative coordinates
      const containerX = e.clientX - containerRect.left;
      const containerY = e.clientY - containerRect.top;
      
      // Convert container coordinates (display) to natural image coordinates
      const scaleX = imageNaturalWidth / imageDisplayWidth;
      const scaleY = imageNaturalHeight / imageDisplayHeight;
      const currentX = containerX * scaleX;
      const currentY = containerY * scaleY;
      
      if (isDragging) {
        const deltaX = currentX - dragStart.x;
        const deltaY = currentY - dragStart.y;
        
        const newX = Math.max(0, Math.min(boxStart.x + deltaX, imageNaturalWidth - boxStart.width));
        const newY = Math.max(0, Math.min(boxStart.y + deltaY, imageNaturalHeight - boxStart.height));
        
        onResize({ x: newX, y: newY, width: boxStart.width, height: boxStart.height });
      } else if (isResizing) {
        const deltaX = currentX - dragStart.x;
        const deltaY = currentY - dragStart.y;
        
        let newX = boxStart.x;
        let newY = boxStart.y;
        let newWidth = boxStart.width;
        let newHeight = boxStart.height;
        
        const minSize = 20; // Minimum box size in natural coordinates
        
        switch (isResizing) {
          case 'nw':
            newX = Math.max(0, boxStart.x + deltaX);
            newY = Math.max(0, boxStart.y + deltaY);
            newWidth = Math.max(minSize, boxStart.width - deltaX);
            newHeight = Math.max(minSize, boxStart.height - deltaY);
            break;
          case 'ne':
            newY = Math.max(0, boxStart.y + deltaY);
            newWidth = Math.max(minSize, boxStart.width + deltaX);
            newHeight = Math.max(minSize, boxStart.height - deltaY);
            break;
          case 'sw':
            newX = Math.max(0, boxStart.x + deltaX);
            newWidth = Math.max(minSize, boxStart.width - deltaX);
            newHeight = Math.max(minSize, boxStart.height + deltaY);
            break;
          case 'se':
            newWidth = Math.max(minSize, boxStart.width + deltaX);
            newHeight = Math.max(minSize, boxStart.height + deltaY);
            break;
        }
        
        // Ensure box stays within image bounds
        if (newX + newWidth > imageNaturalWidth) {
          newWidth = imageNaturalWidth - newX;
        }
        if (newY + newHeight > imageNaturalHeight) {
          newHeight = imageNaturalHeight - newY;
        }
        
        onResize({ x: newX, y: newY, width: newWidth, height: newHeight });
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      setIsResizing(null);
    };

    if (isDragging || isResizing) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, isResizing, dragStart, boxStart, imageNaturalWidth, imageNaturalHeight, imageDisplayWidth, imageDisplayHeight, onResize]);

  // Convert natural coordinates to display coordinates
  const [displayX, displayY] = toDisplayCoords(x, y);
  const [displayWidth, displayHeight] = toDisplayCoords(width, height);

  const handleSize = 8;
  const borderWidth = 3;

  return (
    <div
      ref={boxRef}
      style={{
        position: 'absolute',
        left: `${displayX}px`,
        top: `${displayY}px`,
        width: `${displayWidth}px`,
        height: `${displayHeight}px`,
        border: `${borderWidth}px solid ${color}`,
        borderRadius: '4px',
        boxSizing: 'border-box',
        cursor: isDragging ? 'grabbing' : 'grab',
        pointerEvents: disabled ? 'none' : 'auto',
        zIndex: 5,
      }}
      onMouseDown={(e) => handleMouseDown(e, 'drag')}
    >
      {/* Label above box */}
      <div
        style={{
          position: 'absolute',
          left: '0px',
          top: `${-25}px`,
          backgroundColor: color,
          color: 'white',
          padding: '2px 6px',
          borderRadius: '4px',
          fontSize: '11px',
          fontWeight: 'bold',
          whiteSpace: 'nowrap',
          pointerEvents: 'none',
          zIndex: 10,
        }}
      >
        {label}
      </div>

      {/* Resize handles */}
      {!disabled && (
        <>
          {/* Northwest */}
          <div
            style={{
              position: 'absolute',
              left: `${-handleSize / 2}px`,
              top: `${-handleSize / 2}px`,
              width: `${handleSize}px`,
              height: `${handleSize}px`,
              backgroundColor: color,
              border: '2px solid white',
              borderRadius: '50%',
              cursor: 'nwse-resize',
              zIndex: 15,
            }}
            onMouseDown={(e) => handleMouseDown(e, 'resize', 'nw')}
          />
          {/* Northeast */}
          <div
            style={{
              position: 'absolute',
              right: `${-handleSize / 2}px`,
              top: `${-handleSize / 2}px`,
              width: `${handleSize}px`,
              height: `${handleSize}px`,
              backgroundColor: color,
              border: '2px solid white',
              borderRadius: '50%',
              cursor: 'nesw-resize',
              zIndex: 15,
            }}
            onMouseDown={(e) => handleMouseDown(e, 'resize', 'ne')}
          />
          {/* Southwest */}
          <div
            style={{
              position: 'absolute',
              left: `${-handleSize / 2}px`,
              bottom: `${-handleSize / 2}px`,
              width: `${handleSize}px`,
              height: `${handleSize}px`,
              backgroundColor: color,
              border: '2px solid white',
              borderRadius: '50%',
              cursor: 'nesw-resize',
              zIndex: 15,
            }}
            onMouseDown={(e) => handleMouseDown(e, 'resize', 'sw')}
          />
          {/* Southeast */}
          <div
            style={{
              position: 'absolute',
              right: `${-handleSize / 2}px`,
              bottom: `${-handleSize / 2}px`,
              width: `${handleSize}px`,
              height: `${handleSize}px`,
              backgroundColor: color,
              border: '2px solid white',
              borderRadius: '50%',
              cursor: 'nwse-resize',
              zIndex: 15,
            }}
            onMouseDown={(e) => handleMouseDown(e, 'resize', 'se')}
          />
        </>
      )}
    </div>
  );
}

