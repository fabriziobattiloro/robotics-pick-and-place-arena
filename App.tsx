/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { generateText } from 'ai';
import type { LanguageModelV1ProviderMetadata } from '@ai-sdk/provider';
import JSZip from 'jszip';
import { getModel, getModelOption, MODEL_OPTIONS } from './models';
import { AlertCircle, Loader2, X } from 'lucide-react';
import loadMujoco from 'mujoco_wasm';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { v4 as uuidv4 } from 'uuid';
import { MujocoSim, SavedCubeConfig } from './MujocoSim';
import { RobotSelector } from './components/RobotSelector';
import { Toolbar } from './components/Toolbar';
import { UnifiedSidebar } from './components/UnifiedSidebar';
import { getDetectionPromptParts, getSpatialSegmentationPromptParts } from './prompts/detectionPrompts';
import { DetectedItem, DetectType, LogEntry, MujocoModule } from './types';
import { tlog } from './utils/terminalLog';

interface RunAllResult {
  modelId: string;
  provider: string;
  type: DetectType;
  prompt: string;
  params: {
    temperature: number;
    thinking: boolean;
    twoPass: boolean;
    optimised: boolean;
  };
  screenshot: string;
  detections: DetectedItem[];
  error?: string;
}

interface LogOverlayProps {
  log: LogEntry;
}

const SAVED_CUBES_STORAGE_KEY = 'saved-cubes-layout-v1';
type EnvironmentMode = 'free' | 'saved';

/**
 * LogOverlay
 * Draws Gemini detection results (boxes/points) over an image.
 * Uses a normalized 1000x1000 coordinate system.
 */
export function LogOverlay({ log }: LogOverlayProps) {
  if (!log.result || !Array.isArray(log.result)) return null;
  
  const results = log.result as DetectedItem[];
  const shapes = results.map((item, idx) => {
    if (item.box_2d) {
      const [ymin, xmin, ymax, xmax] = item.box_2d;
      return (
        <rect 
          key={idx} x={xmin} y={ymin} width={xmax - xmin} height={ymax - ymin} 
          fill="rgba(79, 70, 229, 0.15)" stroke="#4f46e5" strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
      );
    } else if (item.point) {
      const [y, x] = item.point;
      // Using vector-effect="non-scaling-stroke" ensures the circle border is visible even in small miniatures.
      // cx/cy are normalized 0-1000.
      return <circle key={idx} cx={x} cy={y} r="10" fill="#4f46e5" stroke="white" strokeWidth="2" vectorEffect="non-scaling-stroke" />;
    }
    return null;
  });

  return (
    <svg 
      viewBox="0 0 1000 1000" 
      preserveAspectRatio="none" 
      className="absolute inset-0 pointer-events-none w-full h-full z-10"
    >
      {shapes}
    </svg>
  );
}

/**
 * Main Application Component
 */
export function App() {
  const containerRef = useRef<HTMLDivElement>(null); 
  const simRef = useRef<MujocoSim | null>(null);      
  const isMounted = useRef(true);                     
  const mujocoModuleRef = useRef<MujocoModule | null>(null);          

  const [isLoading, setIsLoading] = useState(true);
  const [loadingStatus, setLoadingStatus] = useState("Initializing Spatial Engine...");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [mujocoReady, setMujocoReady] = useState(false); 
  
  const [isPaused, setIsPaused] = useState(false);
  // Initialize sidebar based on screen width (hidden on mobile by default)
  const [showSidebar, setShowSidebar] = useState(() => window.innerWidth >= 660); 
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [environmentMode, setEnvironmentMode] = useState<EnvironmentMode>('free');
  const [savedCubesConfig, setSavedCubesConfig] = useState<SavedCubeConfig | null>(() => {
    if (typeof window === 'undefined') return null;
    try {
      const raw = window.localStorage.getItem(SAVED_CUBES_STORAGE_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw) as SavedCubeConfig;
      if (!parsed || !Array.isArray(parsed.cubes) || parsed.cubes.length === 0) return null;
      return parsed;
    } catch {
      return null;
    }
  });
  const freeCubesConfigRef = useRef<SavedCubeConfig | null>(null);
  
  const [erLoading, setErLoading] = useState(false);
  const [logs, setLogs] = useState<Array<LogEntry>>([]);
  const [expandedLogId, setExpandedLogId] = useState<string | null>(null);
  const [flash, setFlash] = useState(false); 
  const detectedTargets = useRef<Array<{pos: THREE.Vector3, markerId: number}>>([]); 
  const [detectedCount, setDetectedCount] = useState(0); 
  
  const [isPickingUp, setIsPickingUp] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [runAllLoading, setRunAllLoading] = useState(false);
  const [runAllProgress, setRunAllProgress] = useState<{ current: number; total: number; modelId: string } | null>(null);
  const [runAllResults, setRunAllResults] = useState<RunAllResult[]>([]);
  const [saveFeedback, setSaveFeedback] = useState<string | null>(null);
  const [environmentFeedback, setEnvironmentFeedback] = useState<string | null>(null);

  const [gizmoStats, setGizmoStats] = useState<{pos: string, rot: string} | null>(null);

  // Deriving activeLog directly from the latest logs state ensures UI reactivity
  const activeLog = expandedLogId ? logs.find(l => l.id === expandedLogId) : null;

  useEffect(() => {
    isMounted.current = true;
    loadMujoco({
      locateFile: (path: string) => path.endsWith('.wasm') ? "https://unpkg.com/mujoco-js@0.0.7/dist/mujoco_wasm.wasm" : path,
      printErr: (text: string) => { 
        if (text.includes("Aborted") && isMounted.current) {
            setLoadError(prev => prev ? prev : "Simulation crashed. Reload page."); 
        }
      }
    }).then((inst: unknown) => { 
      if (isMounted.current) { 
        mujocoModuleRef.current = inst as MujocoModule; 
        setMujocoReady(true); 
      } 
    }).catch((err: Error) => { 
      if (isMounted.current) { 
        setLoadError(err.message || "Failed to init spatial simulation"); 
        setIsLoading(false); 
      } 
    });
    return () => { isMounted.current = false; simRef.current?.dispose(); };
  }, []);

  useEffect(() => {
      if (!mujocoReady || !containerRef.current || !mujocoModuleRef.current) return;
      setIsLoading(true); 
      setLoadError(null); 
      setIsPaused(false);
      
      simRef.current?.dispose();
      
      try {
          simRef.current = new MujocoSim(containerRef.current, mujocoModuleRef.current);
          simRef.current.renderSys.setDarkMode(isDarkMode);
          
          simRef.current.init("franka_panda_stack", "scene.xml", (msg) => {
             if (isMounted.current) setLoadingStatus(msg);
          })
             .then(() => {
                 if (isMounted.current) {
                     setEnvironmentMode('free');
                     simRef.current?.setIkEnabled(false);
                     setIsLoading(false);
                 }
             })
             .catch(err => { 
                 if (isMounted.current) { 
                     setLoadError(err.message); 
                     setIsLoading(false); 
                 } 
             });
             
      } catch (err: unknown) { 
          if (isMounted.current) { setLoadError((err as Error).message); setIsLoading(false); } 
      }
  }, [mujocoReady]);

  useEffect(() => {
    try {
      if (savedCubesConfig && savedCubesConfig.cubes.length > 0) {
        window.localStorage.setItem(SAVED_CUBES_STORAGE_KEY, JSON.stringify(savedCubesConfig));
      } else {
        window.localStorage.removeItem(SAVED_CUBES_STORAGE_KEY);
      }
    } catch {
      // ignore storage failures
    }
  }, [savedCubesConfig]);

  // Effect to move camera when sidebar toggles
  useEffect(() => {
    if (isLoading || !simRef.current || erLoading) return;
    
    // Standard view when sidebar is closed
    const standardPos = new THREE.Vector3(2.2, -1.2, 2.2);
    const standardTarget = new THREE.Vector3(0, 0, 0);
    
    // Offset view to shift robot left when sidebar is open
    const offsetPos = new THREE.Vector3(2.35, -0.7, 2.2);
    const offsetTarget = new THREE.Vector3(0.15, 0.4, 0.05);
    
    // Only offset camera on desktop/tablet (width >= 660px). On mobile, keep centered.
    if (showSidebar && window.innerWidth >= 660) {
      simRef.current.renderSys.moveCameraTo(offsetPos, offsetTarget, 1000);
    } else {
      simRef.current.renderSys.moveCameraTo(standardPos, standardTarget, 1000);
    }
  }, [showSidebar, isLoading, erLoading]);

  useEffect(() => {
      if (isLoading) return;
      let animId: number;
      const uiLoop = () => {
          if (simRef.current) {
              const s = simRef.current.getGizmoStats();
              setGizmoStats(s ? { 
                  pos: `X: ${s.pos.x.toFixed(2)}, Y: ${s.pos.y.toFixed(2)}, Z: ${s.pos.z.toFixed(2)}`, 
                  rot: `X: ${s.rot.x.toFixed(2)}, Y: ${s.rot.y.toFixed(2)}, Z: ${s.rot.z.toFixed(2)}` 
              } : null);
          }
          animId = requestAnimationFrame(uiLoop);
      };
      uiLoop();
      return () => cancelAnimationFrame(animId);
  }, [isLoading]);

  const toggleDarkMode = () => {
    const next = !isDarkMode;
    setIsDarkMode(next);
    simRef.current?.renderSys.setDarkMode(next);
  };

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
        if (simRef.current && !isLoading && !erLoading) {
            const markerPos = simRef.current.renderSys.checkMarkerClick(e.clientX, e.clientY);
            if (markerPos) {
                simRef.current.moveIkTargetTo(markerPos, 2000);
                simRef.current.setIkEnabled(true);
            }
        }
    };
    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [isLoading, erLoading]);

  function parseJsonResponse(text: string | undefined): unknown[] {
      if (!text) return [];
      let jsonText = text.replace(/```json|```/g, '').trim();
      const firstBracket = jsonText.indexOf('[');
      const lastBracket = jsonText.lastIndexOf(']');
      if (firstBracket !== -1 && lastBracket !== -1) {
          jsonText = jsonText.substring(firstBracket, lastBracket + 1);
      }
      try { return JSON.parse(jsonText); }
      catch { tlog('ERR', 'JSON parse failed', { rawText: text.substring(0, 500) }); return []; }
  }

  function normalizePointFromPixels(point: number[], width: number, height: number): number[] {
      if (point.length < 2) return point;
      const [x, y] = point;
      if (typeof x !== 'number' || typeof y !== 'number') return point;
      const safeW = width > 0 ? width : 1;
      const safeH = height > 0 ? height : 1;
      const xNorm = (x / safeW) * 1000;
      const yNorm = (y / safeH) * 1000;
      const clamp = (v: number) => Math.min(1000, Math.max(0, v));
      return [clamp(yNorm), clamp(xNorm)];
  }

  function normalizeResultPointsFromPixels(items: DetectedItem[], width: number, height: number): DetectedItem[] {
      return items.map((item) => {
          if (!item.point || !Array.isArray(item.point)) return item;
          return { ...item, point: normalizePointFromPixels(item.point as number[], width, height) };
      });
  }

  function normalizeBoxFromPixels(box: number[], width: number, height: number): number[] {
      if (box.length < 4) return box;
      const [ymin, xmin, ymax, xmax] = box;
      if ([ymin, xmin, ymax, xmax].some(v => typeof v !== 'number')) return box;
      const safeW = width > 0 ? width : 1;
      const safeH = height > 0 ? height : 1;
      const yMinNorm = (ymin / safeH) * 1000;
      const yMaxNorm = (ymax / safeH) * 1000;
      const xMinNorm = (xmin / safeW) * 1000;
      const xMaxNorm = (xmax / safeW) * 1000;
      const clamp = (v: number) => Math.min(1000, Math.max(0, v));
      return [clamp(yMinNorm), clamp(xMinNorm), clamp(yMaxNorm), clamp(xMaxNorm)];
  }

  function normalizeResultBoxesFromPixels(items: DetectedItem[], width: number, height: number): DetectedItem[] {
      return items.map((item) => {
          if (!item.box_2d || !Array.isArray(item.box_2d)) return item;
          return { ...item, box_2d: normalizeBoxFromPixels(item.box_2d as number[], width, height) };
      });
  }

  function deduplicateByProximity(items: { item: DetectedItem; point3d: THREE.Vector3 }[], threshold: number = 0.04) {
      const kept: typeof items = [];
      for (const entry of items) {
          const isDupe = kept.some(k => k.point3d.distanceTo(entry.point3d) < threshold);
          if (!isDupe) kept.push(entry);
      }
      return kept;
  }

  const handleErSend = async (prompt: string, type: DetectType, temperature: number, enableThinking: boolean, modelId: string, enableSpatialSegmentation: boolean = false, optimisedPrompts: boolean = false) => {
      if (!simRef.current || erLoading) return;
      tlog('BOT', 'Detection pipeline started', { prompt, type, model: modelId, temperature, thinking: enableThinking, twoPass: enableSpatialSegmentation, optimisedPrompts });
      setErLoading(true);
      simRef.current.renderSys.clearErMarkers();
      detectedTargets.current = [];
      setDetectedCount(0);
      setIsPickingUp(false);
      setPlaybackSpeed(1);

      const savedState = simRef.current.renderSys.getCameraState();
      const topPos = new THREE.Vector3(0, -0.01, 2.0);
      const target = new THREE.Vector3(0, 0, 0);
      tlog('BOT', 'Moving camera to top-down view');
      await simRef.current.renderSys.moveCameraTo(topPos, target, 1500);
      await new Promise(r => setTimeout(r, 100));

      tlog('BOT', 'Camera flash — capturing snapshot');
      setFlash(true);
      setTimeout(() => setFlash(false), 100);

      // Dynamic Resizing: Limit max dimension to 640px while preserving aspect ratio.
      const canvas = simRef.current.renderSys.renderer.domElement;
      const width = canvas.width;
      const height = canvas.height;
      const scaleFactor = Math.min(640 / width, 640 / height);
      const snapshotWidth = Math.floor(width * scaleFactor);
      const snapshotHeight = Math.floor(height * scaleFactor);

      // Serialization: Convert to PNG.
      const imageBase64 = simRef.current.renderSys.getCanvasSnapshot(snapshotWidth, snapshotHeight, 'image/png');
      tlog('BOT', 'Image captured', { originalSize: `${width}x${height}`, snapshotSize: `${snapshotWidth}x${snapshotHeight}`, base64Length: imageBase64.length });
      // Payload Preparation: Strip data URI prefix.
      const base64Data = imageBase64.replace('data:image/png;base64,', '');

      const modelOption = getModelOption(modelId);
      const provider = modelOption?.provider || 'google';
      const promptProvider = enableThinking && !optimisedPrompts ? 'google' : provider;
      const promptOptimised = optimisedPrompts;
      const expectsPixelPoints = provider === 'openai' && promptOptimised && type === 'Points';
      const expectsPixelBoxes = provider === 'openai' && promptOptimised && type === '2D bounding boxes';
      const parts = getDetectionPromptParts(promptProvider, type, promptOptimised);
      const subject = prompt.trim() || parts[1];
      const textPrompt = `${parts[0]} ${subject}${parts[2]}`;
      tlog('BOT', 'Prompt constructed', { textPrompt });
      const providerOptions: LanguageModelV1ProviderMetadata = {};
      if (modelOption?.provider === 'google' && !enableThinking) {
          providerOptions.google = { thinkingConfig: { thinkingBudget: 0 } };
      }
      if (modelOption?.provider === 'anthropic' && enableThinking) {
          providerOptions.anthropic = { thinking: { type: 'enabled', budgetTokens: 10000 } };
      }

      const requestLogData = {
          model: modelId,
          provider: modelOption?.provider,
          prompt: textPrompt,
          temperature,
          providerOptions,
      };

      const logId = uuidv4();
      const newLog: LogEntry = {
          id: logId,
          timestamp: new Date(),
          imageSrc: imageBase64,
          prompt,
          fullPrompt: textPrompt,
          type,
          result: null, 
          requestData: requestLogData
      };
      setLogs(prev => [newLog, ...prev]);

      tlog('BOT', 'Restoring camera to original position');
      await simRef.current.renderSys.moveCameraTo(savedState.position, savedState.target, 1500);

      try {
          const model = getModel(modelId);
          const apiCallOptions = Object.keys(providerOptions).length > 0 ? { providerOptions } : {};

          if (enableSpatialSegmentation) {
              // ===== TWO-PASS DETECTION =====

              // --- PASS 1: Coarse region detection ---
              tlog('BOT', 'PASS 1: Detecting regions of interest...');
              const regionPromptParts = getSpatialSegmentationPromptParts(promptProvider, promptOptimised);
              const regionSubject = prompt.trim() || regionPromptParts[1];
              const regionPrompt = `${regionPromptParts[0]} ${regionSubject}${regionPromptParts[2]}`;

              const pass1Start = performance.now();
              const regionResponse = await generateText({
                  model,
                  messages: [{
                      role: 'user' as const,
                      content: [
                          { type: 'image' as const, image: base64Data, mimeType: 'image/png' },
                          { type: 'text' as const, text: regionPrompt },
                      ],
                  }],
                  temperature,
                  ...apiCallOptions,
              });
              tlog('OK', `PASS 1 responded in ${(performance.now() - pass1Start).toFixed(0)}ms`);

              let regions = parseJsonResponse(regionResponse.text) as DetectedItem[];
              const MAX_REGIONS = 4;
              if (regions.length > MAX_REGIONS) {
                  tlog('BOT', `Capping regions from ${regions.length} to ${MAX_REGIONS}`);
                  regions = regions.slice(0, MAX_REGIONS);
              }
              tlog('BOT', `PASS 1: Found ${regions.length} regions`, { regions: regions.map(r => r.label) });

              if (regions.length === 0) {
                  tlog('WARN', 'No regions detected — falling back to single-pass');
              }

              // --- PASS 2: Fine detection per region ---
              const allDetections: { item: DetectedItem; point3d: THREE.Vector3 }[] = [];

              for (let i = 0; i < regions.length; i++) {
                  const region = regions[i];
                  if (!region.box_2d) continue;
                  tlog('BOT', `PASS 2: Scanning region ${i + 1}/${regions.length} — "${region.label}"`);

                  const zoomedCam = simRef.current.renderSys.regionToZoomedCamera(
                      region.box_2d as [number, number, number, number], topPos, target
                  );

                  // Move camera to zoomed position
                  await simRef.current.renderSys.moveCameraTo(zoomedCam.position, zoomedCam.target, 1000);
                  await new Promise(r => setTimeout(r, 100));

                  // Flash and capture zoomed screenshot
                  setFlash(true);
                  setTimeout(() => setFlash(false), 100);
                  const zoomedSnapshot = simRef.current.renderSys.getCanvasSnapshot(snapshotWidth, snapshotHeight, 'image/png');
                  const zoomedBase64 = zoomedSnapshot.replace('data:image/png;base64,', '');

                  // Fine detection with normal prompt
                  const pass2Start = performance.now();
                  const fineResponse = await generateText({
                      model,
                      messages: [{
                          role: 'user' as const,
                          content: [
                              { type: 'image' as const, image: zoomedBase64, mimeType: 'image/png' },
                              { type: 'text' as const, text: textPrompt },
                          ],
                      }],
                      temperature,
                      ...apiCallOptions,
                  });
                  tlog('OK', `PASS 2 region ${i + 1} responded in ${(performance.now() - pass2Start).toFixed(0)}ms`);

                  const fineResult = parseJsonResponse(fineResponse.text) as DetectedItem[];
                  const normalizedFineResult = expectsPixelPoints || expectsPixelBoxes
                      ? normalizeResultBoxesFromPixels(
                          normalizeResultPointsFromPixels(fineResult, snapshotWidth, snapshotHeight),
                          snapshotWidth,
                          snapshotHeight
                        )
                      : fineResult;
                  tlog('BOT', `Region ${i + 1}: detected ${fineResult.length} items`);

                  // Project detections to 3D using the ZOOMED camera
                  for (const item of normalizedFineResult) {
                      let center2d: { x: number; y: number } | null = null;
                      if (item.box_2d) {
                          const [ymin, xmin, ymax, xmax] = item.box_2d;
                          center2d = { x: (xmin + xmax) / 2, y: (ymin + ymax) / 2 };
                      } else if (item.point) {
                          const [y, x] = item.point;
                          center2d = { x, y };
                      }
                      if (center2d) {
                          const projection = simRef.current?.renderSys.project2DTo3D(
                              center2d.x, center2d.y, zoomedCam.position, zoomedCam.target
                          );
                          if (projection) {
                              allDetections.push({ item, point3d: projection.point });
                          }
                      }
                  }

                  // Restore camera between regions
                  await simRef.current.renderSys.moveCameraTo(savedState.position, savedState.target, 800);
              }

              // Deduplicate and place markers
              const deduped = deduplicateByProximity(allDetections);
              tlog('BOT', `Merged ${allDetections.length} detections → ${deduped.length} after dedup`);

              deduped.forEach((entry, idx) => {
                  const markerId = Date.now() + Math.random();
                  simRef.current?.renderSys.addErMarker(entry.point3d, entry.item.label || '', markerId);
                  detectedTargets.current.push({ pos: entry.point3d, markerId });
                  tlog('BOT', `Marker #${idx + 1} placed`, { label: entry.item.label, x: entry.point3d.x.toFixed(3), y: entry.point3d.y.toFixed(3), z: entry.point3d.z.toFixed(3) });
              });

              const displayResults = deduped.map(d => d.item);
              setLogs(prev => prev.map(l => l.id === logId ? { ...l, result: displayResults } : l));
              tlog('OK', `Two-pass detection complete — ${deduped.length} markers placed`);
              setDetectedCount(deduped.length);

          } else {
              // ===== SINGLE-PASS DETECTION (existing logic) =====
              tlog('BOT', `Calling ${modelOption?.provider} API...`, { model: modelId, temperature, thinking: enableThinking });
              const apiStartTime = performance.now();
              const response = await generateText({
                  model,
                  messages: [
                      {
                          role: 'user' as const,
                          content: [
                              { type: 'image' as const, image: base64Data, mimeType: 'image/png' },
                              { type: 'text' as const, text: textPrompt },
                          ],
                      },
                  ],
                  temperature,
                  ...apiCallOptions,
              });

              const apiDuration = (performance.now() - apiStartTime).toFixed(0);
              const text = response.text;
              tlog('OK', `${modelOption?.provider} API responded in ${apiDuration}ms`, { responseLength: text?.length ?? 0 });
              if (!text) throw new Error("No response text returned.");

              let result: unknown[] = parseJsonResponse(text);
              if ((expectsPixelPoints || expectsPixelBoxes) && Array.isArray(result)) {
                  result = normalizeResultBoxesFromPixels(
                      normalizeResultPointsFromPixels(result as DetectedItem[], snapshotWidth, snapshotHeight),
                      snapshotWidth,
                      snapshotHeight
                  );
              }
              tlog('BOT', 'Parsed detection result', { itemCount: Array.isArray(result) ? result.length : 0 });

              // Remove absolute duplicates
              if (Array.isArray(result)) {
                  const seen = new Set();
                  result = result.filter((item: unknown) => {
                      const serialized = JSON.stringify(item);
                      if (seen.has(serialized)) return false;
                      seen.add(serialized);
                      return true;
                  });
              }

              setLogs(prev => prev.map(l => l.id === logId ? { ...l, result } : l));

              if (Array.isArray(result)) {
                  tlog('BOT', `Processing ${result.length} detected items — projecting to 3D`);
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  result.forEach((item: any, idx: number) => {
                      let center2d: {x: number, y: number} | null = null;
                      if (item.box_2d) {
                          const [ymin, xmin, ymax, xmax] = item.box_2d;
                          center2d = { x: (xmin + xmax) / 2, y: (ymin + ymax) / 2 };
                      } else if (item.point) {
                          const [y, x] = item.point;
                          center2d = { x, y };
                      }

                      if (center2d) {
                          const projection = simRef.current?.renderSys.project2DTo3D(center2d.x, center2d.y, topPos, target);
                          if (projection) {
                              const markerId = Date.now() + Math.random();
                              simRef.current?.renderSys.addErMarker(projection.point, item.label, markerId);
                              detectedTargets.current.push({ pos: projection.point, markerId });
                              tlog('BOT', `Marker #${idx + 1} placed`, { label: item.label, x: projection.point.x.toFixed(3), y: projection.point.y.toFixed(3), z: projection.point.z.toFixed(3) });
                          }
                      }
                  });
                  tlog('OK', `Detection complete — ${detectedTargets.current.length} markers placed`);
                  setDetectedCount(detectedTargets.current.length);
              }
          }
      } catch (error: unknown) {
          tlog('ERR', 'API Error', { error: (error as Error).message });
          const errorMsg = (error as Error).message || "Unknown error";
          setLogs(prev => prev.map(l => l.id === logId && l.result === null ? { ...l, result: { error: errorMsg } } : l));
      } finally {
          tlog('BOT', 'Detection pipeline finished');
          setErLoading(false);
      }
  };

  const handleRunAll = async (prompt: string, _type: DetectType, temperature: number, enableThinking: boolean, enableSpatialSegmentation: boolean, optimisedPrompts: boolean) => {
      if (!simRef.current || erLoading || runAllLoading) return;

      const typesToRun: DetectType[] = ['2D bounding boxes', 'Points'];
      const modelsToRun = MODEL_OPTIONS;
      const totalRuns = modelsToRun.length * typesToRun.length;

      tlog('BOT', `Run All started — ${modelsToRun.length} models × ${typesToRun.length} types = ${totalRuns} runs`, { prompt, temperature, thinking: enableThinking, optimised: optimisedPrompts });
      setRunAllLoading(true);
      setRunAllResults([]);
      simRef.current.renderSys.clearErMarkers();
      detectedTargets.current = [];
      setDetectedCount(0);

      const savedState = simRef.current.renderSys.getCameraState();
      const topPos = new THREE.Vector3(0, -0.01, 2.0);
      const target = new THREE.Vector3(0, 0, 0);

      // Move to top-down and capture ONE snapshot for all models
      await simRef.current.renderSys.moveCameraTo(topPos, target, 1500);
      await new Promise(r => setTimeout(r, 100));
      setFlash(true);
      setTimeout(() => setFlash(false), 100);

      const canvas = simRef.current.renderSys.renderer.domElement;
      const width = canvas.width;
      const height = canvas.height;
      const scaleFactor = Math.min(640 / width, 640 / height);
      const snapshotWidth = Math.floor(width * scaleFactor);
      const snapshotHeight = Math.floor(height * scaleFactor);
      const imageBase64 = simRef.current.renderSys.getCanvasSnapshot(snapshotWidth, snapshotHeight, 'image/png');
      const base64Data = imageBase64.replace('data:image/png;base64,', '');

      const results: RunAllResult[] = [];
      let runIndex = 0;

      for (const type of typesToRun) {
        for (let i = 0; i < modelsToRun.length; i++) {
          const modelOption = modelsToRun[i];
          const provider = modelOption.provider;
          runIndex++;
          setRunAllProgress({ current: runIndex, total: totalRuns, modelId: modelOption.id });
          tlog('BOT', `Run All [${runIndex}/${totalRuns}] — ${modelOption.id} / ${type} ${enableSpatialSegmentation ? '(two-pass)' : ''}`);

          const promptProvider = enableThinking && !optimisedPrompts ? 'google' : provider;
          const promptOptimised = optimisedPrompts;
          const parts = getDetectionPromptParts(promptProvider, type, promptOptimised);
          const subject = prompt.trim() || parts[1];
          const textPrompt = `${parts[0]} ${subject}${parts[2]}`;

          const providerOptions: LanguageModelV1ProviderMetadata = {};
          if (provider === 'google' && !enableThinking) {
              providerOptions.google = { thinkingConfig: { thinkingBudget: 0 } };
          }
          if (provider === 'anthropic' && enableThinking) {
              providerOptions.anthropic = { thinking: { type: 'enabled', budgetTokens: 10000 } };
          }

          const expectsPixelPoints = provider === 'openai' && promptOptimised && type === 'Points';
          const expectsPixelBoxes = provider === 'openai' && promptOptimised && type === '2D bounding boxes';

          const logId = uuidv4();
          const newLog: LogEntry = {
              id: logId,
              timestamp: new Date(),
              imageSrc: imageBase64,
              prompt,
              fullPrompt: textPrompt,
              type,
              result: null,
              requestData: {
                  model: modelOption.id,
                  provider,
                  prompt: textPrompt,
                  temperature,
                  providerOptions,
              },
          };
          setLogs(prev => [newLog, ...prev]);

          try {
              const model = getModel(modelOption.id);
              const apiCallOptions = Object.keys(providerOptions).length > 0 ? { providerOptions } : {};

              let detections: DetectedItem[] = [];

              if (enableSpatialSegmentation) {
                  // ===== TWO-PASS =====
                  const regionPromptParts = getSpatialSegmentationPromptParts(promptProvider, promptOptimised);
                  const regionSubject = prompt.trim() || regionPromptParts[1];
                  const regionPrompt = `${regionPromptParts[0]} ${regionSubject}${regionPromptParts[2]}`;

                  // Ensure camera is at top-down for Pass 1 snapshot baseline
                  // (base64Data was captured at topPos at the start of handleRunAll)
                  const pass1Start = performance.now();
                  const regionResponse = await generateText({
                      model,
                      messages: [{
                          role: 'user' as const,
                          content: [
                              { type: 'image' as const, image: base64Data, mimeType: 'image/png' },
                              { type: 'text' as const, text: regionPrompt },
                          ],
                      }],
                      temperature,
                      ...apiCallOptions,
                  });
                  tlog('OK', `PASS 1 (${modelOption.id}) responded in ${(performance.now() - pass1Start).toFixed(0)}ms`);

                  let regions = parseJsonResponse(regionResponse.text) as DetectedItem[];
                  const MAX_REGIONS = 4;
                  if (regions.length > MAX_REGIONS) regions = regions.slice(0, MAX_REGIONS);
                  tlog('BOT', `PASS 1: ${regions.length} regions`, { regions: regions.map(r => r.label) });

                  const allDetections: { item: DetectedItem; point3d: THREE.Vector3 }[] = [];

                  for (let rIdx = 0; rIdx < regions.length; rIdx++) {
                      const region = regions[rIdx];
                      if (!region.box_2d) continue;

                      const zoomedCam = simRef.current.renderSys.regionToZoomedCamera(
                          region.box_2d as [number, number, number, number], topPos, target
                      );
                      await simRef.current.renderSys.moveCameraTo(zoomedCam.position, zoomedCam.target, 800);
                      await new Promise(r => setTimeout(r, 100));
                      setFlash(true);
                      setTimeout(() => setFlash(false), 100);

                      const zoomedSnapshot = simRef.current.renderSys.getCanvasSnapshot(snapshotWidth, snapshotHeight, 'image/png');
                      const zoomedBase64 = zoomedSnapshot.replace('data:image/png;base64,', '');

                      const pass2Start = performance.now();
                      const fineResponse = await generateText({
                          model,
                          messages: [{
                              role: 'user' as const,
                              content: [
                                  { type: 'image' as const, image: zoomedBase64, mimeType: 'image/png' },
                                  { type: 'text' as const, text: textPrompt },
                              ],
                          }],
                          temperature,
                          ...apiCallOptions,
                      });
                      tlog('OK', `PASS 2 region ${rIdx + 1}/${regions.length} responded in ${(performance.now() - pass2Start).toFixed(0)}ms`);

                      const fineResult = parseJsonResponse(fineResponse.text) as DetectedItem[];
                      const normalizedFine = expectsPixelPoints || expectsPixelBoxes
                          ? normalizeResultBoxesFromPixels(
                              normalizeResultPointsFromPixels(fineResult, snapshotWidth, snapshotHeight),
                              snapshotWidth, snapshotHeight
                            )
                          : fineResult;

                      for (const item of normalizedFine) {
                          let center2d: { x: number; y: number } | null = null;
                          if (item.box_2d) {
                              const [ymin, xmin, ymax, xmax] = item.box_2d;
                              center2d = { x: (xmin + xmax) / 2, y: (ymin + ymax) / 2 };
                          } else if (item.point) {
                              const [y, x] = item.point;
                              center2d = { x, y };
                          }
                          if (center2d) {
                              const projection = simRef.current?.renderSys.project2DTo3D(
                                  center2d.x, center2d.y, zoomedCam.position, zoomedCam.target
                              );
                              if (projection) {
                                  allDetections.push({ item, point3d: projection.point });
                              }
                          }
                      }
                  }

                  const deduped = deduplicateByProximity(allDetections);
                  tlog('BOT', `Merged ${allDetections.length} → ${deduped.length} after dedup`);

                  // Return camera to top-down for the screenshot
                  await simRef.current.renderSys.moveCameraTo(topPos, target, 800);
                  await new Promise(r => setTimeout(r, 100));

                  simRef.current?.renderSys.clearErMarkers();
                  deduped.forEach((entry) => {
                      const markerId = Date.now() + Math.random();
                      simRef.current?.renderSys.addErMarker(entry.point3d, entry.item.label || '', markerId);
                  });

                  detections = deduped.map(d => d.item);

              } else {
                  // ===== SINGLE-PASS =====
                  const apiStart = performance.now();
                  const response = await generateText({
                      model,
                      messages: [{
                          role: 'user' as const,
                          content: [
                              { type: 'image' as const, image: base64Data, mimeType: 'image/png' },
                              { type: 'text' as const, text: textPrompt },
                          ],
                      }],
                      temperature,
                      ...apiCallOptions,
                  });
                  tlog('OK', `${modelOption.id} responded in ${(performance.now() - apiStart).toFixed(0)}ms`);

                  if (!response.text) throw new Error("No response text returned.");
                  let result = parseJsonResponse(response.text) as DetectedItem[];

                  if ((expectsPixelPoints || expectsPixelBoxes) && Array.isArray(result)) {
                      result = normalizeResultBoxesFromPixels(
                          normalizeResultPointsFromPixels(result, snapshotWidth, snapshotHeight),
                          snapshotWidth, snapshotHeight
                      ) as DetectedItem[];
                  }

                  const seen = new Set<string>();
                  result = result.filter(item => {
                      const key = JSON.stringify(item);
                      if (seen.has(key)) return false;
                      seen.add(key);
                      return true;
                  });

                  detections = result;

                  simRef.current?.renderSys.clearErMarkers();
                  detections.forEach((item) => {
                      let center2d: { x: number; y: number } | null = null;
                      if (item.box_2d) {
                          const [ymin, xmin, ymax, xmax] = item.box_2d;
                          center2d = { x: (xmin + xmax) / 2, y: (ymin + ymax) / 2 };
                      } else if (item.point) {
                          const [y, x] = item.point;
                          center2d = { x, y };
                      }
                      if (center2d) {
                          const projection = simRef.current?.renderSys.project2DTo3D(center2d.x, center2d.y, topPos, target);
                          if (projection) {
                              const markerId = Date.now() + Math.random();
                              simRef.current?.renderSys.addErMarker(projection.point, item.label || '', markerId);
                          }
                      }
                  });
              }

              // Wait for render, then capture screenshot with markers
              await new Promise(r => setTimeout(r, 200));
              simRef.current?.renderSys.renderNow();
              const screenshot = simRef.current?.renderSys.getCanvasSnapshot(canvas.width, canvas.height, 'image/png') || '';

              tlog('OK', `${modelOption.id}/${type} — ${detections.length} detections, screenshot captured`);
              results.push({
                  modelId: modelOption.id, provider, type, prompt,
                  params: { temperature, thinking: enableThinking, twoPass: enableSpatialSegmentation, optimised: optimisedPrompts },
                  screenshot, detections,
              });
              setLogs(prev => prev.map(l => l.id === logId ? { ...l, result: detections } : l));

          } catch (error: unknown) {
              const errorMsg = (error as Error).message || "Unknown error";
              tlog('ERR', `${modelOption.id} failed`, { error: errorMsg });
              results.push({
                  modelId: modelOption.id, provider, type, prompt,
                  params: { temperature, thinking: enableThinking, twoPass: enableSpatialSegmentation, optimised: optimisedPrompts },
                  screenshot: '', detections: [], error: errorMsg,
              });
              setLogs(prev => prev.map(l => l.id === logId && l.result === null ? { ...l, result: { error: errorMsg } } : l));
          }

          try { simRef.current?.renderSys.clearErMarkers(); } catch { /* ignore */ }
        }
      }

      // Restore camera
      await simRef.current.renderSys.moveCameraTo(savedState.position, savedState.target, 1500);

      setRunAllResults(results);
      setRunAllProgress(null);
      setRunAllLoading(false);
      tlog('OK', `Run All complete — ${results.length} models processed`);
  };

  const handleDownloadResults = async () => {
      if (runAllResults.length === 0) return;

      const typeLabel = (t: DetectType) => t === '2D bounding boxes' ? 'Boxes' : t;
      const buildSuffix = (result: RunAllResult) => [
          typeLabel(result.type),
          result.prompt.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-]/g, ''),
          result.params.optimised ? 'optimised' : 'base',
          result.params.thinking ? 'thinking-on' : 'thinking-off',
          result.params.twoPass ? 'twopass-on' : 'twopass-off',
          `temp-${result.params.temperature}`,
      ].join('_');

      const zip = new JSZip();
      for (const result of runAllResults) {
          const suffix = buildSuffix(result);
          const folder = zip.folder(`${result.modelId}/${typeLabel(result.type)}`)!;
          if (result.screenshot) {
              const b64 = result.screenshot.replace(/^data:image\/\w+;base64,/, '');
              folder.file(`${result.modelId}_${suffix}.png`, b64, { base64: true });
          }
          if (result.error) {
              folder.file(`${result.modelId}_${suffix}_error.txt`, result.error);
          }
          folder.file(`${result.modelId}_${suffix}_detections.json`, JSON.stringify(result.detections, null, 2));
      }

      const r0 = runAllResults[0];
      const zipSuffix = [
          r0.prompt.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-]/g, ''),
          r0.params.optimised ? 'optimised' : 'base',
          r0.params.thinking ? 'thinking-on' : 'thinking-off',
          r0.params.twoPass ? 'twopass-on' : 'twopass-off',
          `temp-${r0.params.temperature}`,
      ].join('_');

      const blob = await zip.generateAsync({ type: 'blob' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `run-all_${zipSuffix}_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.zip`;
      a.click();
      URL.revokeObjectURL(url);
  };

  const handlePickup = () => {
    if (simRef.current) {
        // If already picking up, this button acts as a speed toggle
        if (isPickingUp) {
            let nextSpeed = 1;
            if (playbackSpeed === 1) nextSpeed = 2;
            else if (playbackSpeed === 2) nextSpeed = 5;
            else if (playbackSpeed === 5) nextSpeed = 10;
            else if (playbackSpeed === 10) nextSpeed = 20;
            else if (playbackSpeed === 20) nextSpeed = 2;

            tlog('BOT', `Speed changed to ${nextSpeed}x`);
            setPlaybackSpeed(nextSpeed);
            simRef.current.setSpeedMultiplier(nextSpeed);
            return;
        }

        // Otherwise start the pickup sequence
        if (detectedTargets.current.length > 0) {
            tlog('BOT', `Pickup sequence started — ${detectedTargets.current.length} items to pick`);
            setIsPickingUp(true);
            setPlaybackSpeed(1);
            const positions = detectedTargets.current.map(t => t.pos);
            const markerIds = detectedTargets.current.map(t => t.markerId);

            simRef.current.pickupItems(positions, markerIds, () => {
                // On Finished
                tlog('OK', 'Pickup sequence complete — all items picked');
                setIsPickingUp(false);
                setPlaybackSpeed(1);
                setDetectedCount(0);
                detectedTargets.current = [];
                simRef.current?.setSpeedMultiplier(1);
            });
        }
    }
  };

  const handleReset = () => {
    tlog('BOT', 'Simulation reset');
    simRef.current?.reset();
    if (environmentMode === 'saved' && savedCubesConfig?.cubes?.length) {
      simRef.current?.applyCubeConfig(savedCubesConfig);
    }
    setLogs([]);
    setDetectedCount(0);
    setIsPickingUp(false);
    setPlaybackSpeed(1);
    detectedTargets.current = [];
  };

  const handleSaveCubes = () => {
    if (!simRef.current) return;
    const snapshot = simRef.current.captureCubeConfig();
    if (!snapshot.cubes.length) {
      tlog('WARN', 'No cubes found to save');
      setSaveFeedback('Nessun cubo da salvare');
      setTimeout(() => setSaveFeedback(null), 1800);
      return;
    }
    setSavedCubesConfig(snapshot);
    tlog('OK', `Cubes layout saved (${snapshot.cubes.length} cubes)`);
    setSaveFeedback(`Configurazione salvata (${snapshot.cubes.length} cubi)`);
    setTimeout(() => setSaveFeedback(null), 1800);
  };

  const handleToggleEnvironment = () => {
    if (!simRef.current) return;

    if (environmentMode === 'free') {
      if (!savedCubesConfig?.cubes?.length) {
        tlog('WARN', 'Nessuna configurazione salvata: usa Save prima di entrare in modalità salvata');
        return;
      }
      freeCubesConfigRef.current = simRef.current.captureCubeConfig();
      const applied = simRef.current.applyCubeConfig(savedCubesConfig);
      if (applied) {
        setEnvironmentMode('saved');
        tlog('BOT', 'Ambiente salvato attivato');
        setEnvironmentFeedback('Modalità salvata attiva');
        setTimeout(() => setEnvironmentFeedback(null), 1600);
      }
      return;
    }

    if (freeCubesConfigRef.current?.cubes?.length) {
      simRef.current.applyCubeConfig(freeCubesConfigRef.current);
    } else {
      simRef.current.reset();
    }
    setEnvironmentMode('free');
    tlog('BOT', 'Ambiente libero attivato');
    setEnvironmentFeedback('Modalità libera attiva');
    setTimeout(() => setEnvironmentFeedback(null), 1600);
  };

  return (
    <div className={`w-full h-full relative overflow-hidden font-sans transition-colors duration-500 ${isDarkMode ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-800'}`}>
      {/* 3D Container */}
      <div ref={containerRef} className="w-full h-full absolute inset-0 bg-slate-200" />
      
      {/* Robot Info Overlay */}
      {!loadError && <RobotSelector gizmoStats={gizmoStats} isDarkMode={isDarkMode} />}
      
      {/* Loading Screen */}
      {isLoading && (
          <div className={`absolute inset-0 flex flex-col items-center justify-center z-50 backdrop-blur-md px-6 ${isDarkMode ? 'bg-slate-950/40' : 'bg-slate-50/20'}`}>
              <div className="flex flex-col min-[660px]:flex-row gap-8 max-w-4xl w-full items-stretch">
                  <div className={`glass-panel p-12 rounded-[3rem] flex-1 flex flex-col justify-center shadow-2xl transition-colors ${isDarkMode ? 'bg-slate-900/70 border-white/10' : 'bg-white/70 border-white/80'}`}>
                    <h3 className={`text-sm font-bold uppercase tracking-widest mb-4 ${isDarkMode ? 'text-indigo-400' : 'text-indigo-600'}`}>System Overview</h3>
                    <p className={`text-sm leading-relaxed mb-6 ${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>
                      A browser-based embodied-reasoning demo for the Franka Emika Panda arm. Pick your model — <strong>Gemini</strong>, <strong>Claude</strong>, or <strong>GPT</strong> — and it detects objects in the scene and drives the arm to pick and place them.
                    </p>
                    <ul className={`text-[13px] space-y-3 list-disc list-inside ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                        <li>Real-time MuJoCo physics simulation</li>
                        <li>Analytical Inverse Kinematics for Franka Panda</li>
                        <li>Multi-provider detection (Gemini, Claude, GPT) with a Run All benchmarking mode</li>
                    </ul>
                  </div>

                  <div className={`glass-panel p-10 rounded-[3rem] flex flex-col items-center justify-center shrink-0 min-[660px]:w-[260px] shadow-2xl transition-colors ${isDarkMode ? 'bg-slate-900/70 border-white/10' : 'bg-white/70 border-white/80'}`}>
                      <div className="w-16 h-16 rounded-2xl bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-100/20 animate-pulse-soft mb-6">
                        <Loader2 className="w-8 h-8 text-white animate-spin" />
                      </div>
                      <h2 className={`text-base font-bold text-center px-2 ${isDarkMode ? 'text-slate-100' : 'text-slate-800'}`}>{loadingStatus}</h2>
                  </div>
              </div>
          </div>
      )}
      
      {/* Flash Effect */}
      {flash && <div className="absolute inset-0 bg-white z-[60] pointer-events-none opacity-50" />}

      {/* Environment Status Badge */}
      {!isLoading && !loadError && (
        <div className="absolute top-6 right-6 z-[70] pointer-events-none">
          <div className={`glass-panel px-4 py-2 rounded-xl border text-xs font-bold tracking-wide shadow-xl ${isDarkMode ? 'bg-slate-900/90 border-white/10 text-slate-100' : 'bg-white/90 border-slate-200 text-slate-700'}`}>
            <span className={`inline-block w-2 h-2 rounded-full mr-2 ${environmentMode === 'saved' ? 'bg-indigo-500' : 'bg-emerald-500'}`} />
            Ambiente: {environmentMode === 'saved' ? 'Salvato' : 'Libero'}
          </div>
        </div>
      )}

      {/* Save Feedback Toast */}
      {saveFeedback && (
        <div className="absolute top-6 left-1/2 -translate-x-1/2 z-[70] pointer-events-none">
          <div className={`glass-panel px-4 py-2 rounded-xl border text-sm font-semibold shadow-xl ${isDarkMode ? 'bg-slate-900/90 border-white/10 text-emerald-300' : 'bg-white/90 border-slate-200 text-emerald-700'}`}>
            {saveFeedback}
          </div>
        </div>
      )}

      {/* Environment Switch Toast */}
      {environmentFeedback && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 z-[70] pointer-events-none">
          <div className={`glass-panel px-4 py-2 rounded-xl border text-sm font-semibold shadow-xl ${isDarkMode ? 'bg-slate-900/90 border-white/10 text-indigo-300' : 'bg-white/90 border-slate-200 text-indigo-700'}`}>
            {environmentFeedback}
          </div>
        </div>
      )}
      
      {/* Error State */}
      {loadError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/40 backdrop-blur-xl z-50">
              <div className="glass-panel p-10 rounded-[2.5rem] border-red-100 max-w-md text-center">
                  <div className="w-16 h-16 bg-red-50 text-red-600 rounded-full flex items-center justify-center mx-auto mb-6">
                    <AlertCircle className="w-8 h-8" />
                  </div>
                  <h3 className="text-2xl text-slate-800 font-bold mb-2">Simulation Halted</h3>
                  <p className="text-slate-500 mb-8 leading-relaxed">{loadError}</p>
                  <button 
                    onClick={() => window.location.reload()} 
                    className="w-full py-4 bg-slate-900 text-white rounded-2xl font-bold hover:bg-black transition-all shadow-xl active:scale-95"
                  >
                    Restart System
                  </button>
              </div>
          </div>
      )}
      
      {/* Main UI Controls */}
      {!isLoading && !loadError && (
        <>
          <Toolbar 
            isPaused={isPaused} 
            togglePause={() => setIsPaused(simRef.current?.togglePause() ?? false)} 
            onReset={handleReset} 
            showSidebar={showSidebar}
            toggleSidebar={() => setShowSidebar(!showSidebar)}
            isDarkMode={isDarkMode}
            toggleDarkMode={toggleDarkMode}
            onSaveCubes={handleSaveCubes}
            hasSavedCubes={Boolean(savedCubesConfig?.cubes?.length)}
            isSavedEnvironment={environmentMode === 'saved'}
            onToggleEnvironment={handleToggleEnvironment}
          />
          
          <UnifiedSidebar
            isOpen={showSidebar}
            onClose={() => setShowSidebar(false)}
            onSend={handleErSend}
            onRunAll={handleRunAll}
            onDownloadResults={handleDownloadResults}
            onPickup={handlePickup}
            isLoading={erLoading}
            runAllLoading={runAllLoading}
            runAllProgress={runAllProgress}
            hasRunAllResults={runAllResults.length > 0}
            hasDetectedItems={detectedCount > 0}
            logs={logs}
            onOpenLog={(log) => setExpandedLogId(log.id)}
            isDarkMode={isDarkMode}
            isPickingUp={isPickingUp}
            playbackSpeed={playbackSpeed}
          />

          {/* Expanded View Modal - Overlay everything */}
          {activeLog && (
            <div className="fixed inset-0 z-[100] flex items-center justify-center min-[660px]:p-10 bg-slate-950/20 backdrop-blur-xl animate-in fade-in" onClick={() => setExpandedLogId(null)}>
              <div className={`glass-panel overflow-hidden flex flex-col shadow-2xl transition-colors fixed top-4 bottom-4 left-4 right-4 rounded-[2.5rem] min-[660px]:relative min-[660px]:inset-auto min-[660px]:w-full min-[660px]:max-w-6xl min-[660px]:max-h-[90vh] ${isDarkMode ? 'bg-slate-900 border-white/10 text-slate-100' : 'bg-white border-white/80 text-slate-800'}`} onClick={e => e.stopPropagation()}>
                 <div className={`p-8 border-b flex justify-between items-center shrink-0 ${isDarkMode ? 'border-white/5 bg-white/5' : 'border-slate-100 bg-white/40'}`}>
                    <div>
                      <h3 className="text-xl font-bold">API Call</h3>
                      <p className={`text-xs font-medium ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>{activeLog.timestamp.toLocaleString()}</p>
                    </div>
                    <button onClick={() => setExpandedLogId(null)} className={`w-10 h-10 flex items-center justify-center rounded-full shadow-sm border transition-colors ${isDarkMode ? 'bg-slate-800 border-white/10 text-slate-400 hover:text-slate-200' : 'bg-white border-slate-100 text-slate-400 hover:text-slate-600'}`}>
                      <X className="w-5 h-5" />
                    </button>
                 </div>
                 <div className="flex-1 flex max-[659px]:flex-col max-[659px]:overflow-y-auto custom-scrollbar min-[660px]:flex-row min-[660px]:overflow-hidden">
                    <div className={`flex items-center justify-center border-b min-[660px]:border-b-0 min-[660px]:border-r min-[660px]:flex-1 min-[660px]:p-8 min-[660px]:overflow-hidden max-[659px]:shrink-0 max-[659px]:p-6 ${isDarkMode ? 'bg-slate-950/50 border-white/5' : 'bg-slate-50/30 border-slate-100'}`}>
                       <div className={`relative rounded-2xl overflow-hidden shadow-lg border-2 flex items-center justify-center min-[660px]:w-auto min-[660px]:h-auto min-[660px]:max-w-full min-[660px]:max-h-full max-[659px]:w-full max-[659px]:h-auto ${isDarkMode ? 'border-white/10 bg-black/20' : 'border-white bg-black/5'}`}>
                          <img src={activeLog.imageSrc} className={`block w-full h-auto min-[660px]:max-w-full min-[660px]:max-h-full`} alt="Detailed log" />
                          <LogOverlay log={activeLog} />
                       </div>
                    </div>
                    <div className={`min-[660px]:w-[420px] p-8 flex flex-col gap-6 min-[660px]:overflow-y-auto min-[660px]:custom-scrollbar ${isDarkMode ? 'bg-white/5' : 'bg-white/20'}`}>
                       <div className="space-y-1">
                          <h4 className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">User Prompt</h4>
                          <p className="text-sm font-bold leading-tight">{activeLog.prompt}</p>
                       </div>
                       <div className="space-y-1 flex-1 min-h-0">
                          <h4 className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">Full Prompt</h4>
                          <div className={`text-[12px] font-mono p-4 rounded-xl leading-relaxed border whitespace-pre-wrap overflow-y-auto h-full ${isDarkMode ? 'bg-slate-950 border-white/5 text-slate-400' : 'bg-slate-50 border-slate-200/50 text-slate-500'}`}>
                            {activeLog.fullPrompt}
                          </div>
                       </div>
                       <div className="space-y-3 flex flex-col flex-1 min-h-0">
                          <div className="flex items-center justify-between">
                            <h4 className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">API Call Results</h4>
                            <button
                              onClick={() => {
                                if (!activeLog?.result) return;
                                const text = JSON.stringify(activeLog.result, null, 2);
                                navigator.clipboard?.writeText(text);
                              }}
                              className={`text-[10px] font-semibold px-2 py-1 rounded-md border transition-colors ${isDarkMode ? 'border-white/10 text-slate-300 hover:text-white hover:border-white/20' : 'border-slate-200 text-slate-500 hover:text-slate-800 hover:border-slate-300'}`}
                            >
                              Copy JSON
                            </button>
                          </div>
                          <div className={`p-4 rounded-xl font-mono text-[12px] border overflow-y-auto shadow-inner flex-1 min-h-0 max-[659px]:h-96 ${isDarkMode ? 'bg-slate-950 border-white/5 text-indigo-400' : 'bg-slate-50/50 border-slate-100 text-indigo-600'}`}>
                            {activeLog.result === null ? (
                                <div className="h-full flex flex-col items-center justify-center gap-3 text-indigo-400 animate-pulse">
                                    <Loader2 className="w-6 h-6 animate-spin" />
                                    <span className="font-sans font-bold text-[8px] uppercase tracking-widest">Processing...</span>
                                </div>
                            ) : (
                                <pre className="whitespace-pre-wrap break-all leading-relaxed">{JSON.stringify(activeLog.result, null, 2)}</pre>
                            )}
                          </div>
                       </div>
                       <div className="min-[660px]:hidden h-8 shrink-0" />
                    </div>
                 </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
