/**
 * Preset model registry data paths.
 */

import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const currentDir = dirname(fileURLToPath(import.meta.url));

/** Path to the default models JSON file. */
export const DEFAULT_MODELS_PATH = join(currentDir, 'defaultModels.json');
