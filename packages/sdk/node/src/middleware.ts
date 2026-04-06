/**
 * Express middleware for TryAii-DRE.
 *
 * Adds routing headers to responses so downstream consumers know which
 * model was selected and its confidence score.
 *
 * Usage:
 *   import express from 'express';
 *   import { dreMiddleware } from 'tryaii-dre-sdk/middleware';
 *
 *   const app = express();
 *   app.use(express.json());
 *   app.use(dreMiddleware({ priorities: { quality: 5, cost: 1, speed: 3 } }));
 */

import type { Request, Response, NextFunction, RequestHandler } from 'express';
import type { DREMiddlewareOptions } from './types.js';
import { DREClient } from './client.js';

/**
 * Create Express middleware that classifies the incoming request body
 * and attaches routing headers to the response.
 *
 * Headers added (default prefix "X-DRE"):
 *   - X-DRE-Model: the recommended model ID
 *   - X-DRE-Score: the model's final score (0-1)
 */
export function dreMiddleware(options?: DREMiddlewareOptions): RequestHandler {
  const prefix = options?.headerPrefix ?? 'X-DRE';
  const promptField = options?.promptField ?? 'prompt';

  const client = new DREClient({
    priorities: options?.priorities,
  });

  return (req: Request, res: Response, next: NextFunction): void => {
    try {
      const body = req.body as Record<string, unknown> | undefined;
      const prompt = body?.[promptField];

      if (typeof prompt === 'string' && prompt.length > 0) {
        const result = client.route(prompt, {
          priorities: options?.priorities,
        });

        res.setHeader(`${prefix}-Model`, result.bestModel);
        res.setHeader(`${prefix}-Score`, String(result.bestScore));
      }
    } catch {
      // Routing failure should not block the request pipeline
    }

    next();
  };
}
