import type { DetectType } from '../types';

export const basePromptParts: Record<DetectType, [string, string, string]> = {
  '2D bounding boxes': [
    'Detect',
    'items',
    ', with no more than 25 items. DO NOT detect items that only match the description partially. Output a json list where each entry contains the 2D bounding box in “box_2d” and a text label in “label”.',
  ],
  'Segmentation masks': [
    'Give the segmentation masks for',
    'all objects',
    '. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key “box_2d”, the segmentation mask in key “mask”, and the text label in the key “label”. Use descriptive labels.',
  ],
  'Points': [
    'Identify ',
    'items',
    ' in the scene and mark them with points. DO NOT mark items that only match the description partially. Follow the JSON format: [{“point”: [y, x], “label”: “label”}, ...]. The points are in [y, x] format normalized to 0-1000.',
  ],
};

export const optimisedPromptParts: Record<string, Record<DetectType, [string, string, string]>> = {
  google: {
    '2D bounding boxes': [
      'You are an object detection system. Look at this image and detect',
      'items',
      '. Only detect items that fully match the description. Definition of “cube”: A 3D block with a visible square top face and visible edges; solid colored faces (e.g., red, green, yellow, cyan). Ignore robot, base/grid, shadows, blue/gray dots/markers, and any non-cube shapes. Output ONLY a JSON array (no other text) where each entry has “box_2d” as [ymin, xmin, ymax, xmax] with values normalized to 0-1000 scale (0=top/left edge, 1000=bottom/right edge), and “label” as “cube”. If there are no cubes, return [] exactly.',
    ],
    'Segmentation masks': [
      'You are an object detection system. Look at this image and identify',
      'all objects',
      '. Output ONLY a JSON array (no other text) where each entry has “box_2d” as [ymin, xmin, ymax, xmax] with values normalized to 0-1000 scale (0=top/left edge, 1000=bottom/right edge), “mask” as the segmentation mask, and “label” as a descriptive text string.',
    ],
    'Points': [
      'You are a precise visual annotator. Your task is to find all and only the colored cubes in the image.',
      'items',
      ' Definition of “cube”: A 3D block with a visible square top face and visible edges; solid colored faces (e.g., red, green, yellow, cyan). Ignore robot, base/grid, shadows, blue/gray dots/markers, and any non-cube shapes. Point definition: point = the center of the top face of the cube in 2D; if the top face is not visible, use the center of the visible cube area. Output rules: Return ONLY a JSON array of objects with exactly {“point”: [y, x], “label”: “cube”}. The points are in [y, x] format normalized to 0-1000. If there are no cubes, return [] exactly. No extra text, no commentary, no markdown.',
    ],
  },
  openai: {
    '2D bounding boxes': [
      'You are an object detection system. Look at this image and detect',
      'items',
      '. Only detect items that fully match the description. Definition of “cube”: A 3D block with a visible square top face and visible edges; solid colored faces (e.g., red, green, yellow, cyan). Ignore robot, base/grid, shadows, blue/gray dots/markers, and any non-cube shapes. Output ONLY a JSON array (no other text) where each entry has “box_2d” as [ymin, xmin, ymax, xmax] in PIXELS (origin top-left), and “label” as “cube”. Coordinates must be integers (round to nearest integer). If there are no cubes, return [] exactly.',
    ],
    'Segmentation masks': [
      'You are an object detection system. Look at this image and identify',
      'all objects',
      '. Output ONLY a JSON array (no other text) where each entry has “box_2d” as [ymin, xmin, ymax, xmax] with values normalized to 0-1000 scale (0=top/left edge, 1000=bottom/right edge), “mask” as the segmentation mask, and “label” as a descriptive text string.',
    ],
    'Points': [
      'You are a precise visual annotator. Your task is to find all and only the colored cubes in the image.',
      'items',
      'Definition of “cube”: A 3D block with a visible square top face and visible edges; solid colored faces (e.g., red, green, yellow, cyan). Ignore robot, base/grid, shadows, blue/gray dots/markers, and any non-cube shapes. Point definition: point = the center of the top face of the cube in 2D; if the top face is not visible, use the center of the visible cube area. Output rules: Return ONLY a JSON array of objects with exactly {“point”:[x,y], “label”:”cube”}. x and y are integers (round to nearest integer). Origin (0,0) is the top-left corner of the image, coordinates in pixels. If there are no cubes, return [] exactly. No extra text, no commentary, no markdown.',
    ],
  },
  anthropic: {
    '2D bounding boxes': [
      'You are a high-precision robotic vision system. The image is a 1000x1000 normalized grid where [500,500] is the exact center. Output ONLY a raw JSON array. Inside EACH object, you MUST generate keys in this exact order: 1) "perceive" (brief visual description), 2) "box_2d" (the [ymin, xmin, ymax, xmax] coordinates normalized 0-1000 tightly fitting the ENTIRE object), 3) "label" (e.g. "red cube", use spaces). CRITICAL CONSTRAINTS: Ignore the large gray square target. Only detect colored cubes. Ignore any purple dots, markers, or UI overlays. Then detect ',
      'items',
      '. You MUST output a JSON array starting strictly with [ and ending with ]. Do not use markdown backticks or any text outside the array.',
    ],
    'Segmentation masks': [
      'You are a high-precision robotic vision system. Output ONLY a raw JSON array. Inside EACH object, you MUST generate keys in this exact order: 1) "perceive" (brief visual description), 2) "mask" (the segmentation data), 3) "label" (concise description). Then identify ',
      'all objects',
      '. You MUST output a JSON array starting strictly with [ and ending with ]. Do not use markdown backticks.',
    ],
    'Points': [
      'You are a high-precision robotic vision system. The image is a 1000x1000 normalized grid where [500,500] is the exact center. Output ONLY a raw JSON array. For EACH detected object, generate keys in this EXACT order: 1) "perceive" (describe the exact pixel-level location relative to center), 2) "point" (CRITICAL: exact [y, x] center of the top visible face normalized 0-1000), 3) "label" (e.g. "red cube", use spaces). CRITICAL CONSTRAINTS: Ignore the large gray square target. Only detect colored cubes with a visible top face. Ignore any purple dots, markers, or UI overlays. Then target ALL matching items among: ',
      'items',
      '. Output ONLY a JSON array starting with [ and ending with ].',
    ],
  },
};

export const baseSpatialPromptParts: [string, string, string] = [
  'Look at this top-down view of a robotic workspace. Identify up to 4 distinct spatial regions where',
  'items',
  ' might be located. Return a JSON array where each entry has "box_2d" as [ymin, xmin, ymax, xmax] and "label" describing what is in that region. Focus on spatial clusters, NOT individual objects. Maximum 4 regions.',
];

export const optimisedSpatialPromptParts: [string, string, string] = [
  'You are a spatial analysis system. Look at this top-down image of a robotic workspace and identify up to 4 rectangular regions where',
  'items',
  ' might be clustered. Output ONLY a JSON array (no other text) where each entry has "box_2d" as [ymin, xmin, ymax, xmax] with values normalized to 0-1000 scale (0=top/left, 1000=bottom/right), and "label" describing that region. Focus on spatial clusters, NOT individual objects. Maximum 4 regions. Example: [{"box_2d": [100, 200, 400, 500], "label": "cluster of red cubes"}]',
];

export function getDetectionPromptParts(provider: string, type: DetectType, optimised: boolean = false): [string, string, string] {
  if (optimised) {
    const providerPrompts = optimisedPromptParts[provider] || optimisedPromptParts.google;
    return providerPrompts[type];
  }
  return basePromptParts[type];
}

export function getSpatialSegmentationPromptParts(_provider: string, optimised: boolean = false): [string, string, string] {
  if (optimised) {
    return optimisedSpatialPromptParts;
  }
  return baseSpatialPromptParts;
}
