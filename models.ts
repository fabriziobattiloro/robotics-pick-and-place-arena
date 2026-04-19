/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createOpenAI } from '@ai-sdk/openai';

export type Provider = 'google' | 'anthropic' | 'openai';

export interface ModelOption {
  id: string;
  provider: Provider;
  supportsSegmentation: boolean;
}

export const MODEL_OPTIONS: ModelOption[] = [
  { id: 'gemini-robotics-er-1.5-preview', provider: 'google', supportsSegmentation: true },
  { id: 'gemini-3.1-pro-preview', provider: 'google', supportsSegmentation: false },
  { id: 'gemini-3-flash-preview', provider: 'google', supportsSegmentation: false },
  { id: 'claude-sonnet-4-6', provider: 'anthropic', supportsSegmentation: false },
  { id: 'claude-opus-4-6', provider: 'anthropic', supportsSegmentation: false },
  { id: 'gpt-5.4', provider: 'openai', supportsSegmentation: false },
];

const google = createGoogleGenerativeAI({ apiKey: process.env.GOOGLE_API_KEY || '' });
const anthropic = createAnthropic({
  apiKey: process.env.ANTHROPIC_API_KEY || '',
  headers: { 'anthropic-dangerous-direct-browser-access': 'true' },
});
const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY || '' });

export function getModel(modelId: string) {
  const option = MODEL_OPTIONS.find(m => m.id === modelId);
  if (!option) throw new Error(`Unknown model: ${modelId}`);

  switch (option.provider) {
    case 'google': return google(modelId);
    case 'anthropic': return anthropic(modelId);
    case 'openai': return openai(modelId);
  }
}

export function getModelOption(modelId: string): ModelOption | undefined {
  return MODEL_OPTIONS.find(m => m.id === modelId);
}
